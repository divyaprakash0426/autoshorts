"""Utility for automatically generating short video clips using GPU acceleration.

This module processes gameplay videos and creates resized clips
that fit common short-video aspect ratios. It leverages NVIDIA GPU
(CUDA) for scene detection, audio/video analysis, image filtering,
and video encoding to maximize performance.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import os
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Optional

import numpy as np
from dotenv import load_dotenv
from moviepy import CompositeVideoClip, VideoFileClip
import cupy as cp
import cupyx.scipy.ndimage
import torch
import torchaudio
import decord
from decord import VideoReader, cpu, gpu
from tqdm import tqdm

# Load environment variables from a .env file if present.
load_dotenv()

# Configure basic logging.
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Ensure decord uses the correct backend
decord.bridge.set_bridge('torch')


def _get_env_int(name: str, default: int) -> int:
    """Read an int environment variable with a default and basic validation."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except Exception:
        logging.warning("Env var %s=%r is not a valid int. Using default %s.", name, value, default)
        return default


def _get_env_float(name: str, default: float) -> float:
    """Read a float environment variable with a default and basic validation."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except Exception:
        logging.warning(
            "Env var %s=%r is not a valid float. Using default %s.", name, value, default
        )
        return default


@dataclass(frozen=True)
class ProcessingConfig:
    """Configuration values used throughout the processing pipeline."""

    target_ratio_w: int = 1
    target_ratio_h: int = 1
    scene_limit: int = 6
    x_center: float = 0.5
    y_center: float = 0.5
    max_error_depth: int = 3
    min_short_length: int = 15
    max_short_length: int = 179
    max_combined_scene_length: int = 300

    @property
    def middle_short_length(self) -> float:
        """Return the mid point between min and max short lengths."""
        return (self.min_short_length + self.max_short_length) / 2


class _SecondsTime:
    """Lightweight stand-in for scene time objects using seconds."""

    def __init__(self, seconds: float):
        self._seconds = float(seconds)

    def get_seconds(self) -> float:
        return self._seconds

    def get_timecode(self) -> str:
        return f"{self._seconds:.2f}"

    def get_frames(self) -> int:
        return int(self._seconds * 30)


def detect_video_scenes_gpu(video_path: Path, threshold: float = 27.0) -> List[Tuple[_SecondsTime, _SecondsTime]]:
    """Detect scenes in the provided video file using GPU acceleration.

    Computes frame-to-frame difference on GPU and applies a threshold
    to identify scene cuts.

    Parameters
    ----------
    video_path: Path
        Path to the video file.
    threshold: float
        Threshold for scene change detection (scaled to pixel difference).

    Returns
    -------
    List[Tuple[_SecondsTime, _SecondsTime]]
        List of ``(start, end)`` timecodes for each detected scene.
    """

    # Check for GPU availability for decord
    ctx = gpu(0) if torch.cuda.is_available() else cpu(0)

    try:
        # Determine size on CPU first
        vr_cpu = VideoReader(str(video_path), ctx=cpu(0))
        h, w, _ = vr_cpu[0].shape
        del vr_cpu

        # Resize factor 4
        w_new = max(1, w // 4)
        h_new = max(1, h // 4)

        vr = VideoReader(str(video_path), ctx=ctx, width=w_new, height=h_new)
    except Exception as e:
        logging.warning(f"Failed to load video on GPU: {e}. Falling back to CPU/Standard load.")
        # Ensure fallback also uses resizing to avoid processing full 4K frames on CPU
        try:
            vr = VideoReader(str(video_path), ctx=cpu(0), width=w_new, height=h_new)
        except NameError:
             # If w_new/h_new calculation failed above
             vr = VideoReader(str(video_path), ctx=cpu(0))
        ctx = cpu(0)

    duration = len(vr) / vr.get_avg_fps()
    fps = vr.get_avg_fps()
    frame_count = len(vr)

    # Process in batches to avoid OOM
    batch_size = 16
    scene_cuts = [0.0]

    # Progress bar
    total_batches = (frame_count + batch_size - 1) // batch_size
    pbar = tqdm(total=total_batches, desc=f"Detect scenes GPU: {video_path.name}", unit="batch")

    prev_frame_tensor = None

    # Ideally, we should use a scene detection library but we are rewriting for GPU.
    # Simple algorithm: L1 distance between consecutive frames.

    # Normalize threshold: standard scenedetect uses 0-255 average diff.
    # We will compute mean absolute difference per pixel.

    cut_indices = [0]

    # To speed up, we can skip frames if FPS is high, but we might miss cuts.
    # Let's try to batch read.

    for i in range(0, frame_count, batch_size):
        # Read batch
        end_idx = min(i + batch_size, frame_count)

        # Optimized: Reading batch is faster
        frames = vr.get_batch(range(i, end_idx))

        # Convert to float
        frames_small = frames.float()

        # Convert to grayscale: 0.299*R + 0.587*G + 0.114*B
        # shape: (B, H, W, 3)
        gray = (frames_small[..., 0] * 0.299 +
                frames_small[..., 1] * 0.587 +
                frames_small[..., 2] * 0.114)

        # gray shape: (B, H, W)

        if prev_frame_tensor is not None:
            # Concatenate prev frame for diff
            curr_batch = gray
            prev_batch = torch.cat([prev_frame_tensor.unsqueeze(0), gray[:-1]])
        else:
            curr_batch = gray[1:]
            prev_batch = gray[:-1]

        diff = torch.abs(curr_batch - prev_batch)
        # Mean over H, W
        score = diff.mean(dim=(1, 2))

        # Find cuts
        # score is a 1D tensor of differences

        # Identify peaks
        cuts = (score > threshold).nonzero(as_tuple=True)[0]

        # Map back to global frame index
        # If prev_frame_tensor was not None, index 0 in score corresponds to frame i
        # If it was None, index 0 corresponds to frame i+1 vs i.

        offset = i if prev_frame_tensor is not None else i + 1

        for cut_idx in cuts:
            global_frame = offset + cut_idx.item()
            cut_indices.append(global_frame)

        prev_frame_tensor = gray[-1]

        # Free memory
        del frames, frames_small, gray, curr_batch, prev_batch, diff, score
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Progress update
        if pbar:
            pbar.update(1)

    # Finish progress bar
    if pbar:
        pbar.close()

    cut_indices.append(frame_count)
    cut_indices = sorted(list(set(cut_indices)))

    scenes = []
    for k in range(len(cut_indices) - 1):
        start_frame = cut_indices[k]
        end_frame = cut_indices[k+1]

        # Filter extremely short scenes (e.g. < 0.5s) to avoid jitter
        if (end_frame - start_frame) / fps < 0.5:
            continue

        start_time = start_frame / fps
        end_time = end_frame / fps
        scenes.append((_SecondsTime(start_time), _SecondsTime(end_time)))

    return scenes


def blur(image: np.ndarray) -> np.ndarray:
    """Return a blurred version of ``image`` using GPU acceleration.

    Transfers the image to GPU, applies Gaussian blur via CuPy,
    and transfers it back.
    """
    # Convert numpy to cupy
    image_gpu = cp.asarray(image)

    # Separate channels (H, W, 3) -> (3, H, W) for faster processing or process as is?
    # Gaussian filter in cupyx supports ndarray.
    # We want to blur spatial dims (0, 1) and not color (2) if it's separate?
    # Actually, gaussian_filter on a 3D array with sigma=8 blurs all axes by default.
    # We usually only want to blur H and W, not across channels (mixing colors).
    # So we should use sigma=(8, 8, 0).

    blurred_gpu = cupyx.scipy.ndimage.gaussian_filter(image_gpu.astype(float), sigma=(8, 8, 0))

    # Convert back to numpy
    return cp.asnumpy(blurred_gpu).astype(image.dtype)


# --- Audio-based action scoring (GPU) -------------------------------------------

def compute_audio_action_profile(
    video_path: Path,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute audio-based "action score" on GPU.

    Returns:
      times  - array of times (seconds) for each feature frame
      score  - combined action score (loudness + spectral "roughness")
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load audio using torchaudio
    # normalize=True loads as float32 in [-1, 1]
    waveform, sample_rate = torchaudio.load(str(video_path), normalize=True)

    # Move to GPU
    waveform = waveform.to(device)

    # Mix to mono if necessary
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Remove batch dim for processing
    y = waveform.squeeze(0)

    # RMS (Loudness)
    # We can use unfold to create frames
    # shape: (n_frames, frame_length)
    if y.shape[0] < frame_length:
        # Pad if too short
        y = torch.nn.functional.pad(y, (0, frame_length - y.shape[0]))

    # RMS calculation with progress bar
    # Unfold creates slices: (n_frames, frame_length)
    windows = y.unfold(0, frame_length, hop_length)
    n_frames = windows.shape[0]

    rms_chunks = []
    batch_size = 4096 if n_frames > 4096 else max(1, n_frames)
    total_batches = (n_frames + batch_size - 1) // batch_size
    pbar_rms = tqdm(total=total_batches, desc=f"Audio RMS: {video_path.name}", unit="batch")
    for i in range(0, n_frames, batch_size):
        chunk = windows[i : i + batch_size]
        rms_chunk = torch.sqrt(torch.mean(chunk**2, dim=1))
        rms_chunks.append(rms_chunk)
        pbar_rms.update(1)
    pbar_rms.close()
    rms = torch.cat(rms_chunks) if len(rms_chunks) > 0 else torch.tensor([], device=device)

    # Spectral Flux
    # STFT on GPU
    window = torch.hann_window(2048).to(device)
    stft = torch.stft(y, n_fft=2048, hop_length=hop_length, window=window, return_complex=True)
    magnitude = torch.abs(stft)  # shape: (freq_bins, n_frames)

    # Diff along time axis (dim 1) with progress bar
    t_frames = magnitude.shape[1]
    if t_frames <= 0:
        spectral_flux = torch.tensor([], device=device)
    else:
        # We'll compute differences in chunks along time axis (excluding the first frame)
        spectral_parts = [torch.tensor([0.0], device=device)]  # pad first value
        time_batch = 4096 if t_frames > 4096 else max(1, t_frames)
        total_time_batches = ((t_frames - 1) + time_batch - 1) // time_batch if t_frames > 1 else 0
        pbar_flux = tqdm(total=total_time_batches, desc=f"Audio spectral flux: {video_path.name}", unit="batch")
        for s in range(1, t_frames, time_batch):
            L = min(time_batch, t_frames - s)
            curr = magnitude[:, s : s + L]
            prev = magnitude[:, s - 1 : s + L - 1]
            diff = curr - prev
            flux_chunk = torch.sqrt(torch.sum(diff**2, dim=0))
            spectral_parts.append(flux_chunk)
            pbar_flux.update(1)
        pbar_flux.close()
        spectral_flux = torch.cat(spectral_parts, dim=0)

    # Match lengths: STFT and unfold might have slight size mismatch due to padding
    min_len = min(rms.shape[0], spectral_flux.shape[0])
    rms = rms[:min_len]
    spectral_flux = spectral_flux[:min_len]

    # Normalization
    rms_mean = rms.mean() if rms.numel() > 0 else torch.tensor(0.0, device=device)
    rms_std = (rms.std() + 1e-8) if rms.numel() > 0 else torch.tensor(1.0, device=device)
    rms_norm = (rms - rms_mean) / rms_std if rms.numel() > 0 else rms

    flux_mean = spectral_flux.mean() if spectral_flux.numel() > 0 else torch.tensor(0.0, device=device)
    flux_std = (spectral_flux.std() + 1e-8) if spectral_flux.numel() > 0 else torch.tensor(1.0, device=device)
    flux_norm = (spectral_flux - flux_mean) / flux_std if spectral_flux.numel() > 0 else spectral_flux

    # Smoothing (convolution)
    def smooth_gpu(x: torch.Tensor, win: int = 21) -> torch.Tensor:
        if x.numel() == 0:
            return x
        if win > x.shape[0]:
            win = x.shape[0]
        if win % 2 == 0:
            win += 1
        padding = win // 2
        # kernel
        kernel = torch.ones(win, device=device) / win
        # Add batch/channel dims for conv1d: (1, 1, seq_len)
        x_reshaped = x.view(1, 1, -1)
        kernel_reshaped = kernel.view(1, 1, -1)
        out = torch.nn.functional.conv1d(x_reshaped, kernel_reshaped, padding=padding)
        return out.view(-1)

    rms_smooth = smooth_gpu(rms_norm, win=21)
    flux_smooth = smooth_gpu(flux_norm, win=21)

    score = 0.6 * rms_smooth + 0.4 * flux_smooth if rms_smooth.numel() > 0 and flux_smooth.numel() > 0 else (
        rms_smooth if flux_smooth.numel() == 0 else flux_smooth
    )

    # Convert times to CPU numpy
    num_frames = score.shape[0]
    times = torch.arange(num_frames, device=device) * hop_length / sample_rate if num_frames > 0 else torch.tensor([], device=device)

    return times.cpu().numpy(), score.cpu().numpy()


def compute_video_action_profile(
    video_path: Path,
    fps: int = 6,
    downscale_factor: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute video-based "action score" on GPU.

    Uses Decord to read frames directly to GPU memory and computes
    mean absolute pixel difference.
    """

    ctx = gpu(0) if torch.cuda.is_available() else cpu(0)

    try:
        # First, read metadata on CPU to determine size without loading to GPU
        vr_cpu = VideoReader(str(video_path), ctx=cpu(0))
        h, w, _ = vr_cpu[0].shape
        del vr_cpu

        # Calculate new dimensions
        w_new = max(1, w // downscale_factor)
        h_new = max(1, h // downscale_factor)

        # Load directly to GPU with resize
        vr = VideoReader(str(video_path), ctx=ctx, width=w_new, height=h_new)
    except Exception:
        logging.warning("Failed to load video for action profile.", exc_info=True)
        return np.array([]), np.array([])

    duration = len(vr) / vr.get_avg_fps()
    orig_fps = vr.get_avg_fps()
    eff_fps = min(float(fps), float(orig_fps))
    if eff_fps <= 0:
        eff_fps = max(1.0, float(fps))

    # Calculate indices to sample
    step = max(1, int(orig_fps / eff_fps))
    indices = list(range(0, len(vr), step))

    if not indices:
        return np.array([]), np.array([])

    motions = []
    times = []

    # Process in batches - reduced size to save memory
    batch_size = 2
    prev_batch_last = None

    total_batches = (len(indices) + batch_size - 1) // batch_size
    pbar = tqdm(total=total_batches, desc=f"Video action GPU: {video_path.name}", unit="batch")

    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]
        frames = vr.get_batch(batch_indices) # (B, H_new, W_new, C)

        # Frames are already resized. Convert to float.
        frames_small = frames.float()

        # Grayscale
        gray = (frames_small[..., 0] * 0.299 +
                frames_small[..., 1] * 0.587 +
                frames_small[..., 2] * 0.114)

        # Diff
        if prev_batch_last is not None:
             diffs = torch.abs(gray - torch.cat([prev_batch_last.unsqueeze(0), gray[:-1]]))
        else:
             # For the very first frame, diff is 0
             diffs = torch.abs(gray - torch.cat([gray[0:1], gray[:-1]]))
             diffs[0] = 0.0

        # Mean diff per frame
        batch_motions = diffs.mean(dim=(1, 2))

        motions.append(batch_motions)

        # Timestamps
        batch_times = torch.tensor(batch_indices, device=gray.device).float() / orig_fps
        times.append(batch_times)

        prev_batch_last = gray[-1]

        del frames, frames_small, gray, diffs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pbar.update(1)

    pbar.close()

    motions = torch.cat(motions)
    times = torch.cat(times)

    # Normalize and smooth (similar to audio)
    if motions.std() == 0:
        motions_norm = motions
    else:
        motions_norm = (motions - motions.mean()) / (motions.std() + 1e-8)

    # Smooth
    def smooth_gpu(x, win):
        if win > x.shape[0]: win = x.shape[0]
        if win < 2: return x
        kernel = torch.ones(win, device=x.device) / win
        x_reshaped = x.view(1, 1, -1)
        kernel_reshaped = kernel.view(1, 1, -1)
        out = torch.nn.functional.conv1d(x_reshaped, kernel_reshaped, padding=win//2)
        return out.view(-1)[:x.shape[0]]

    score = smooth_gpu(motions_norm, win=int(eff_fps))

    return times.cpu().numpy(), score.cpu().numpy()


def scene_action_score(
    scene: Tuple,
    audio_times: np.ndarray,
    audio_score: np.ndarray,
    video_times: np.ndarray | None = None,
    video_score: np.ndarray | None = None,
    w_audio: float = 0.6,
    w_video: float = 0.4,
) -> float:
    """Return total (summed) action score within the scene."""

    start_sec = scene[0].get_seconds()
    end_sec = scene[1].get_seconds()

    if end_sec <= start_sec:
        return 0.0

    def _segment_sum(times: np.ndarray, score: np.ndarray) -> float:
        if times.size == 0 or score.size == 0:
            return 0.0
        mask = (times >= start_sec) & (times < end_sec)
        if not np.any(mask):
            return 0.0
        return float(score[mask].sum())

    audio_val = _segment_sum(audio_times, audio_score)

    if video_times is None or video_score is None:
        return audio_val

    video_val = _segment_sum(video_times, video_score)

    return w_audio * audio_val + w_video * video_val


def _best_window_single(
    scene: Tuple,
    window_length: float,
    times: np.ndarray,
    score: np.ndarray,
) -> float:
    """Helper to find best window on a single profile."""

    start_sec = float(scene[0].get_seconds())
    end_sec = float(scene[1].get_seconds())

    if not math.isfinite(start_sec) or not math.isfinite(end_sec) or end_sec <= start_sec:
        return start_sec

    max_allowed_start = end_sec - float(window_length)
    if max_allowed_start <= start_sec:
        return max(start_sec, min(start_sec, end_sec - float(window_length)))

    mask = (times >= start_sec) & (times <= end_sec)
    if not np.any(mask):
        return start_sec

    t_seg = times[mask]
    s_seg = score[mask]

    if len(t_seg) < 2:
        return start_sec

    dt = float(np.median(np.diff(t_seg)))
    if not math.isfinite(dt) or dt <= 0:
        return start_sec

    n_win = int(max(1, round(float(window_length) / dt)))
    if len(s_seg) < n_win:
        return start_sec

    csum = np.cumsum(np.concatenate(([0.0], s_seg)))
    window_sums = csum[n_win:] - csum[:-n_win]
    best_idx = int(np.argmax(window_sums))

    best_start_time = float(t_seg[best_idx])
    best_start_time = max(start_sec, min(best_start_time, max_allowed_start))

    return best_start_time


def best_action_window_start(
    scene: Tuple,
    window_length: float,
    audio_times: np.ndarray,
    audio_score: np.ndarray,
    video_times: np.ndarray | None = None,
    video_score: np.ndarray | None = None,
    w_audio: float = 0.6,
    w_video: float = 0.4,
) -> float:
    """Find the start of the window inside the scene maximizing combined action."""

    if (
        video_times is None
        or video_score is None
        or len(video_times) == 0
        or len(video_score) == 0
    ):
        return _best_window_single(scene, window_length, audio_times, audio_score)

    start_sec = float(scene[0].get_seconds())
    end_sec = float(scene[1].get_seconds())

    if not math.isfinite(start_sec) or not math.isfinite(end_sec) or end_sec <= start_sec:
        return start_sec

    a_mask = (audio_times >= start_sec) & (audio_times <= end_sec)
    if not np.any(a_mask):
        return _best_window_single(scene, window_length, video_times, video_score)

    t_a_seg = audio_times[a_mask]
    s_a_seg = audio_score[a_mask]

    if len(t_a_seg) < 2:
        return _best_window_single(scene, window_length, video_times, video_score)

    if len(video_times) > 1:
        order = np.argsort(video_times)
        v_interp = np.interp(t_a_seg, video_times[order], video_score[order])
    else:
        v_interp = np.full_like(t_a_seg, float(video_score[0]), dtype=float)

    combined_seg = w_audio * s_a_seg + w_video * v_interp

    dt = float(np.median(np.diff(t_a_seg)))
    if not math.isfinite(dt) or dt <= 0:
        return _best_window_single(scene, window_length, audio_times, audio_score)

    max_allowed_start = end_sec - float(window_length)
    if max_allowed_start <= start_sec:
        return max(start_sec, min(start_sec, end_sec - float(window_length)))

    n_win = int(max(1, round(float(window_length) / dt)))
    if len(combined_seg) < n_win:
        return _best_window_single(scene, window_length, audio_times, audio_score)

    csum = np.cumsum(np.concatenate(([0.0], combined_seg)))
    window_sums = csum[n_win:] - csum[:-n_win]
    best_idx = int(np.argmax(window_sums))

    best_start_time = float(t_a_seg[best_idx])
    best_start_time = max(start_sec, min(best_start_time, max_allowed_start))

    return best_start_time


def crop_clip(
    clip: VideoFileClip,
    ratio_w: int,
    ratio_h: int,
    x_center: float,
    y_center: float,
):
    """Crop ``clip`` to the desired aspect ratio."""
    width, height = clip.size
    current_ratio = width / height
    target_ratio = ratio_w / ratio_h

    if current_ratio > target_ratio:
        new_width = round(height * ratio_w / ratio_h)
        return clip.cropped(
            width=new_width,
            height=height,
            x_center=width * x_center,
            y_center=height * y_center,
        )

    new_height = round(width / ratio_w * ratio_h)
    return clip.cropped(
        width=width,
        height=new_height,
        x_center=width * x_center,
        y_center=height * y_center,
    )


def render_video(
    clip: VideoFileClip,
    video_file_name: Path,
    output_dir: Path,
    depth: int = 0,
    max_error_depth: int = 3,
) -> None:
    """Render ``clip`` to ``output_dir`` using NVIDIA NVENC hardware encoding."""

    try:
        # Use NVENC for hardware acceleration
        clip.write_videofile(
            str(output_dir / video_file_name.name),
            codec="h264_nvenc",
            audio_codec="aac",
            fps=min(getattr(clip, "fps", 60), 60),
            # h264_nvenc supports presets like 'p1' to 'p7' (p4 is default/medium)
            # We can also set bitrate if needed.
            ffmpeg_params=["-preset", "p4"]
        )
    except Exception as e:
        logging.warning(f"NVENC rendering failed: {e}. Trying fallback to libx264...")
        try:
            clip.write_videofile(
                str(output_dir / video_file_name.name),
                codec="libx264",
                audio_codec="aac",
                fps=min(getattr(clip, "fps", 60), 60),
            )
        except Exception:
            if depth < max_error_depth:
                logging.exception("Rendering failed, retrying...")
                render_video(
                    clip,
                    video_file_name,
                    output_dir,
                    depth + 1,
                    max_error_depth,
                )
            else:
                logging.exception("Rendering failed after multiple attempts.")
                raise


def select_background_resolution(width: int) -> Tuple[int, int]:
    """Choose an output resolution based on the clip width."""
    if width < 840:
        return 720, 1280
    if width < 1020:
        return 900, 1600
    if width < 1320:
        return 1080, 1920
    if width < 1680:
        return 1440, 2560
    if width < 2040:
        return 1800, 3200
    return 2160, 3840


def get_final_clip(
    clip: VideoFileClip,
    start_point: float,
    final_clip_length: float,
    config: ProcessingConfig,
) -> VideoFileClip:
    """Prepare a clip ready for rendering."""

    result_clip = clip.subclipped(start_point, start_point + final_clip_length)

    width, height = result_clip.size
    target_ratio = config.target_ratio_w / config.target_ratio_h
    if width / height > target_ratio:
        result_clip = crop_clip(
            result_clip,
            config.target_ratio_w,
            config.target_ratio_h,
            config.x_center,
            config.y_center,
        )

    width, height = result_clip.size
    bg_w, bg_h = select_background_resolution(width)
    result_clip = result_clip.resized(width=bg_w)

    if width >= height:
        background_clip = clip.subclipped(start_point, start_point + final_clip_length)
        background_clip = crop_clip(background_clip, 1, 1, config.x_center, config.y_center)
        background_clip = background_clip.resized(width=720, height=720)
        # Apply GPU-accelerated blur
        background_clip = background_clip.image_transform(blur)
        background_clip = background_clip.resized(width=bg_w, height=bg_w)
        result_clip = CompositeVideoClip([background_clip, result_clip.with_position("center")])
    elif width / 9 < height / 16:
        background_clip = clip.subclipped(start_point, start_point + final_clip_length)
        background_clip = crop_clip(background_clip, 9, 16, config.x_center, config.y_center)
        background_clip = background_clip.resized(width=720, height=1280)
        # Apply GPU-accelerated blur
        background_clip = background_clip.image_transform(blur)
        background_clip = background_clip.resized(width=bg_w, height=bg_h)
        result_clip = CompositeVideoClip([background_clip, result_clip.with_position("center")])

    return result_clip


def combine_scenes(scene_list: Sequence[Tuple], config: ProcessingConfig) -> List[List]:
    """Combine adjacent scenes while preserving content."""

    if not scene_list:
        return []

    def is_small(scene) -> bool:
        return (scene[1].get_seconds() - scene[0].get_seconds()) < config.min_short_length

    n = len(scene_list)
    out: List[List] = []

    # Initialize first run
    run_start_idx = 0
    run_type_small = is_small(scene_list[0])
    run_start_time = scene_list[0][0]
    run_end_time = scene_list[0][1]

    for i in range(1, n):
        current_small = is_small(scene_list[i])
        if current_small == run_type_small:
            # Same-type run continues; extend end.
            run_end_time = scene_list[i][1]

            if run_type_small:
                run_duration = run_end_time.get_seconds() - run_start_time.get_seconds()
                if run_duration > config.max_combined_scene_length:
                    prev_end_time = scene_list[i - 1][1]
                    out.append([run_start_time, prev_end_time])
                    run_start_idx = i
                    run_start_time = scene_list[i][0]
                    run_end_time = scene_list[i][1]
                elif run_duration == config.max_combined_scene_length:
                    is_last_scene = (i == n - 1)
                    if is_last_scene:
                        prev_end_time = scene_list[i - 1][1]
                        out.append([run_start_time, prev_end_time])
                        run_start_idx = i
                        run_start_time = scene_list[i][0]
                        run_end_time = scene_list[i][1]
                    else:
                        out.append([run_start_time, run_end_time])
                        run_start_idx = i + 1
                        run_start_time = scene_list[i][1]
                        run_end_time = scene_list[i][1]
        else:
            run_end_idx = i - 1
            run_duration = run_end_time.get_seconds() - run_start_time.get_seconds()
            is_boundary = (run_start_idx == 0) or (run_end_idx == n - 1)
            threshold = config.middle_short_length if is_boundary else config.min_short_length

            if run_duration >= threshold:
                out.append([run_start_time, run_end_time])
                run_start_idx = i
                run_type_small = current_small
                run_start_time = scene_list[i][0]
                run_end_time = scene_list[i][1]
            else:
                if is_boundary and run_start_idx == 0:
                    run_start_idx = i
                    run_type_small = current_small
                    run_start_time = scene_list[i][0]
                    run_end_time = scene_list[i][1]
                else:
                    run_type_small = current_small
                    run_end_time = scene_list[i][1]

    final_duration = run_end_time.get_seconds() - run_start_time.get_seconds()
    is_boundary = True
    threshold = config.middle_short_length if is_boundary else config.min_short_length
    if final_duration >= threshold:
        out.append([run_start_time, run_end_time])

    return out


def split_overlong_scenes(combined_scene_list: List[List], config: ProcessingConfig) -> List[List]:
    """Split scenes longer than 4 * max_short_length into n equal parts."""
    result: List[List] = []
    threshold = 4 * config.max_short_length
    for scene in combined_scene_list:
        start_s = scene[0].get_seconds()
        end_s = scene[1].get_seconds()
        duration = end_s - start_s

        if duration > threshold:
            n = int(math.floor(duration / (2 * config.max_short_length)))
            if n <= 1:
                result.append(scene)
                continue

            part_len = duration / n
            for i in range(n):
                part_start = start_s + i * part_len
                part_end = start_s + (i + 1) * part_len
                result.append([_SecondsTime(part_start), _SecondsTime(part_end)])
        else:
            result.append(scene)

    return result


def process_video(video_file: Path, config: ProcessingConfig, output_dir: Path) -> None:
    """Process a single video file and generate short clips."""

    logging.info("\nProcess: %s", video_file.name)

    logging.info("Detecting scenes (GPU)...")
    scene_list = detect_video_scenes_gpu(video_file)

    # Explicitly clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logging.info("Computing audio action profile (GPU)...")
    audio_times, audio_score = compute_audio_action_profile(video_file)

    # Explicitly clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logging.info("Computing video action profile (GPU)...")
    video_times, video_score = compute_video_action_profile(
        video_file,
        fps=4,
        downscale_factor=6,
    )

    # Explicitly clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    processed_scene_list = combine_scenes(scene_list, config)
    processed_scene_list = split_overlong_scenes(processed_scene_list, config)

    logging.info("Scenes list with action scores:")
    for i, scene in enumerate(processed_scene_list, start=1):
        duration = scene[1].get_seconds() - scene[0].get_seconds()
        score_val = scene_action_score(scene, audio_times, audio_score, video_times, video_score)
        logging.info(
            "    Scene %2d: Duration %5.1f s, ActionScore %7.3f,"
            " Start %s / Frame %d, End %s / Frame %d",
            i,
            duration,
            score_val,
            scene[0].get_timecode(),
            scene[0].get_frames(),
            scene[1].get_timecode(),
            scene[1].get_frames(),
        )

    sorted_processed_scene_list = sorted(
        processed_scene_list,
        key=lambda s: scene_action_score(s, audio_times, audio_score, video_times, video_score),
        reverse=True,
    )

    logging.info("Sorted scenes list (by action score):")
    for i, scene in enumerate(sorted_processed_scene_list, start=1):
        duration = scene[1].get_seconds() - scene[0].get_seconds()
        score_val = scene_action_score(scene, audio_times, audio_score, video_times, video_score)
        logging.info(
            "    Scene %2d: ActionScore %7.3f, Duration %5.1f s,"
            " Start %s / Frame %d, End %s / Frame %d",
            i,
            score_val,
            duration,
            scene[0].get_timecode(),
            scene[0].get_frames(),
            scene[1].get_timecode(),
            scene[1].get_frames(),
        )

    video_clip = VideoFileClip(str(video_file))
    truncated_list = sorted_processed_scene_list[: config.scene_limit]

    if truncated_list:
        for i, scene in enumerate(truncated_list):
            duration = math.floor(scene[1].get_seconds() - scene[0].get_seconds())
            short_length = random.randint(
                config.min_short_length, min(config.max_short_length, duration)
            )

            best_start = best_action_window_start(
                scene,
                float(short_length),
                audio_times,
                audio_score,
                video_times,
                video_score,
            )
            logging.info(
                "Selected start %.2f for scene %d with window %ds",
                best_start,
                i,
                short_length,
            )

            final_clip = get_final_clip(
                video_clip,
                best_start,
                short_length,
                config,
            )

            render_file_name = f"{video_file.stem} scene-{i}{video_file.suffix}"
            render_video(
                final_clip,
                Path(render_file_name),
                output_dir,
                max_error_depth=config.max_error_depth,
            )
    else:
        short_length = random.randint(
            config.min_short_length, config.max_short_length
        )

        if video_clip.duration < config.max_short_length:
            adapted_short_length = min(math.floor(video_clip.duration), short_length)
        else:
            adapted_short_length = short_length

        min_start_point = min(10, math.floor(video_clip.duration) - adapted_short_length)
        max_start_point = math.floor(video_clip.duration - adapted_short_length)
        final_clip = get_final_clip(
            video_clip,
            random.randint(min_start_point, max_start_point),
            adapted_short_length,
            config,
        )
        render_video(
            final_clip,
            video_file,
            output_dir,
            max_error_depth=config.max_error_depth,
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the shorts generator."""
    parser = argparse.ArgumentParser(description="Generate short clips from gameplay footage using GPU.")
    return parser.parse_args()


def config_from_env() -> ProcessingConfig:
    """Build ProcessingConfig from environment variables."""
    return ProcessingConfig(
        target_ratio_w=_get_env_int("TARGET_RATIO_W", 1),
        target_ratio_h=_get_env_int("TARGET_RATIO_H", 1),
        scene_limit=_get_env_int("SCENE_LIMIT", 6),
        x_center=_get_env_float("X_CENTER", 0.5),
        y_center=_get_env_float("Y_CENTER", 0.5),
        max_error_depth=_get_env_int("MAX_ERROR_DEPTH", 3),
        min_short_length=_get_env_int("MIN_SHORT_LENGTH", 15),
        max_short_length=_get_env_int("MAX_SHORT_LENGTH", 179),
        max_combined_scene_length=_get_env_int("MAX_COMBINED_SCENE_LENGTH", 300),
    )


def main() -> None:
    """Entry point for command-line execution."""
    args = parse_args()
    config = config_from_env()
    output_dir = Path("generated")
    output_dir.mkdir(exist_ok=True)

    gameplay_dir = Path("gameplay")
    if not gameplay_dir.exists():
         logging.warning("No 'gameplay' directory found. Exiting.")
         return

    for video_file in gameplay_dir.iterdir():
        if video_file.is_file() and video_file.suffix.lower() in [".mp4", ".mkv", ".mov"]:
            process_video(video_file, config, output_dir)


if __name__ == "__main__":
    main()
