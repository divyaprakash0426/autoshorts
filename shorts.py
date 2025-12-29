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
import multiprocessing
try:
    import resource
except ImportError:
    resource = None
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Optional

import numpy as np
from dotenv import load_dotenv
# moviepy is removed from the rendering path.
# We only import it if strictly necessary for some legacy helper, but we try to avoid it.
# import moviepy.editor as mp # Removed

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


def log_memory_usage(tag: str = ""):
    """Log current memory usage (RAM and VRAM)."""
    usage_stats = []

    # RAM
    if resource:
        # ru_maxrss is in KB on Linux
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        ram_mb = rusage.ru_maxrss / 1024.0
        usage_stats.append(f"RAM: {ram_mb:.1f} MB")

    # VRAM
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        usage_stats.append(f"VRAM Alloc: {allocated:.1f} MB, Res: {reserved:.1f} MB")

    logging.info(f"[{tag}] Memory: {', '.join(usage_stats)}")


@dataclass
class RenderParams:
    """Parameters required to render the final clip."""
    source_path: Path
    start_time: float
    duration: float
    output_width: int
    output_height: int
    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int
    bg_width: int
    bg_height: int
    is_vertical_bg: bool  # True if 9:16 background, False if 1:1 background (resizing logic)


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
    """Detect scenes matching PySceneDetect ContentDetector, but with GPU-assisted I/O.

    This implementation replicates scenedetect.detectors.ContentDetector (v0.6.7)
    semantics to produce identical scene cuts:
      - Frames are downscaled to an effective width of ~256 px (like SceneManager.auto_downscale).
      - Frame score is the mean absolute difference between adjacent frames in HSV space
        (per-channel: hue, saturation, value), averaged with equal weights.
      - Cuts are produced via the same FlashFilter MERGE policy with min_scene_len=15 frames.
      - Scene list is generated exactly like SceneManager.get_scene_list(start_in_scene=False):
        if no cuts are found, returns an empty list.

    GPU usage: frames are decoded/resized with decord (GPU if available). HSV conversion uses
    OpenCV on CPU to match ContentDetector exactly. The difference/thresholding logic follows
    the original algorithm.
    """
    import cv2

    # 1) Determine original size (CPU), compute SceneDetect-like downscale factor.
    try:
        vr_probe = VideoReader(str(video_path), ctx=cpu(0))
        h0, w0, _ = vr_probe[0].shape
        fps = float(vr_probe.get_avg_fps())
        del vr_probe
    except Exception:
        # Fallback: open with default context just to probe
        vr_probe = VideoReader(str(video_path))
        h0, w0, _ = vr_probe[0].shape
        fps = float(vr_probe.get_avg_fps())
        del vr_probe

    # SceneManager.DEFAULT_MIN_WIDTH = 256
    TARGET_MIN_WIDTH = 256
    if w0 < TARGET_MIN_WIDTH:
        downscale = 1.0
    else:
        downscale = w0 / float(TARGET_MIN_WIDTH)

    w_eff = int(w0 / downscale)
    h_eff = int(h0 / downscale)
    w_eff = max(1, w_eff)
    h_eff = max(1, h_eff)

    # 2) Open reader at effective size. Prefer GPU context like SceneDetect's downscale.
    ctx = gpu(0) if torch.cuda.is_available() else cpu(0)
    try:
        vr = VideoReader(str(video_path), ctx=ctx, width=w_eff, height=h_eff)
    except Exception as e:
        logging.warning(f"VideoReader GPU open failed: {e}. Falling back to CPU.")
        vr = VideoReader(str(video_path), ctx=cpu(0), width=w_eff, height=h_eff)

    frame_count = len(vr)
    if frame_count == 0 or fps <= 0.0:
        return []

    # 3) FlashFilter (MERGE) identical logic to scenedetect.scene_detector.FlashFilter
    class _FlashFilterMerge:
        def __init__(self, length: int):
            self._filter_length = int(length)
            self._last_above: Optional[int] = None
            self._merge_enabled: bool = False
            self._merge_triggered: bool = False
            self._merge_start: Optional[int] = None

        @property
        def max_behind(self) -> int:
            return self._filter_length

        def filter(self, frame_num: int, above_threshold: bool) -> List[int]:
            if not (self._filter_length > 0):
                return [frame_num] if above_threshold else []
            if self._last_above is None:
                self._last_above = frame_num
            # MERGE path
            return self._filter_merge(frame_num, above_threshold)

        def _filter_merge(self, frame_num: int, above_threshold: bool) -> List[int]:
            min_length_met = (frame_num - self._last_above) >= self._filter_length
            if above_threshold:
                self._last_above = frame_num
            if self._merge_triggered:
                num_merged_frames = self._last_above - self._merge_start
                if min_length_met and (not above_threshold) and (num_merged_frames >= self._filter_length):
                    self._merge_triggered = False
                    return [self._last_above]
                return []
            if not above_threshold:
                return []
            if min_length_met:
                self._merge_enabled = True
                return [frame_num]
            if self._merge_enabled:
                self._merge_triggered = True
                self._merge_start = frame_num
            return []

    min_scene_len = 15  # ContentDetector default
    flash_filter = _FlashFilterMerge(length=min_scene_len)

    # 4) Iterate frames, compute HSV components & frame score like ContentDetector
    #    Score normalization: divide by sum(abs(weights)) = 3.
    batch_size = 16
    total_batches = (frame_count + batch_size - 1) // batch_size
    pbar = tqdm(total=total_batches, desc=f"Detect scenes", unit="batch")

    last_hsv: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    cut_indices: List[int] = []

    # Helper: ensure ndarray uint8 contiguous for cv2
    def _to_bgr_uint8_cpu(frames_tensor: torch.Tensor) -> List[np.ndarray]:
        # decord produces RGB order; convert to BGR for OpenCV
        frames_cpu = frames_tensor.detach().to('cpu').numpy()
        # frames_cpu: (B,H,W,3), uint8
        # Convert RGB -> BGR by reversing last axis
        frames_bgr = frames_cpu[..., ::-1]
        # Ensure contiguous arrays per frame for cv2
        return [np.ascontiguousarray(frames_bgr[i]) for i in range(frames_bgr.shape[0])]

    for i in range(0, frame_count, batch_size):
        end_idx = min(i + batch_size, frame_count)
        frames_t = vr.get_batch(range(i, end_idx))  # torch Tensor (on GPU if ctx=gpu)
        # Convert to CPU BGR uint8 numpy arrays
        frames_bgr_list = _to_bgr_uint8_cpu(frames_t)

        # Process each frame sequentially to exactly match CPU semantics
        for j, bgr in enumerate(frames_bgr_list):
            frame_num = i + j
            # OpenCV HSV conversion (exact semantics/hue range)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            hue, sat, val = cv2.split(hsv)

            if last_hsv is None:
                last_hsv = (hue, sat, val)
                # First score is 0.0 by design
                above = False
                # Prime flash filter state
                flash_filter.filter(frame_num, above_threshold=above)
                continue

            hue_prev, sat_prev, val_prev = last_hsv
            # Mean pixel distance per channel (match _mean_pixel_distance)
            # cast to int32 to avoid uint8 underflow
            dh = np.abs(hue.astype(np.int32) - hue_prev.astype(np.int32)).sum() / float(hue.size)
            ds = np.abs(sat.astype(np.int32) - sat_prev.astype(np.int32)).sum() / float(sat.size)
            dv = np.abs(val.astype(np.int32) - val_prev.astype(np.int32)).sum() / float(val.size)
            frame_score = (dh + ds + dv) / 3.0

            # Record and advance last_hsv
            last_hsv = (hue, sat, val)

            # Compare against threshold exactly like ContentDetector
            above = frame_score >= threshold
            emitted = flash_filter.filter(frame_num=frame_num, above_threshold=above)
            if emitted:
                cut_indices.extend(emitted)

        # release batch tensors ASAP
        del frames_t, frames_bgr_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pbar.update(1)

    pbar.close()

    # Build scenes like get_scenes_from_cuts, but align to detect_video_scenes default (start_in_scene=False)
    if not cut_indices:
        return []

    cut_indices = sorted(set(cut_indices))
    scenes: List[Tuple[_SecondsTime, _SecondsTime]] = []
    last_cut = 0
    for cut in cut_indices:
        start_time = last_cut / fps
        end_time = cut / fps
        scenes.append((_SecondsTime(start_time), _SecondsTime(end_time)))
        last_cut = cut
    # Last scene from last cut to end_pos (= frame_count, exclusive)
    scenes.append((_SecondsTime(last_cut / fps), _SecondsTime(frame_count / fps)))

    return scenes


def blur_gpu(image_tensor: torch.Tensor, sigma: float = 8.0) -> torch.Tensor:
    """Return a blurred version of ``image_tensor`` using CuPy.

    Args:
        image_tensor: (H, W, 3) torch tensor on GPU.
        sigma: Blur sigma.

    Returns:
        Blurred torch tensor (H, W, 3).
    """
    # Use DLPack to zero-copy transfer to Cupy
    # Note: image_tensor must be contiguous
    if not image_tensor.is_contiguous():
        image_tensor = image_tensor.contiguous()

    cupy_array = cp.from_dlpack(torch.to_dlpack(image_tensor))

    # Gaussian blur. sigma=(sigma, sigma, 0) means blur H and W, preserve channels.
    # Convert to float for precision, then back to uint8
    f_array = cupy_array.astype(float)
    blurred = cupyx.scipy.ndimage.gaussian_filter(f_array, sigma=(sigma, sigma, 0))
    blurred = blurred.astype(cupy_array.dtype)

    # Convert back to torch (use DLPack protocol, avoiding deprecated CuPy .toDlpack())
    return torch.utils.dlpack.from_dlpack(blurred)


# --- Audio-based action scoring (GPU) -------------------------------------------

def compute_audio_action_profile(
    video_path: Path,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute audio-based "action score" on GPU with memory-efficient batching.

    Returns:
      times  - array of times (seconds) for each feature frame
      score  - combined action score (loudness + spectral "roughness")
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load audio using torchaudio (remains on CPU initially)
    # normalize=True loads as float32 in [-1, 1]
    # We do NOT move the full waveform to GPU.
    try:
        waveform, sample_rate = torchaudio.load(str(video_path), normalize=True)
    except Exception:
        logging.error(f"Failed to load audio from {video_path}")
        return np.array([]), np.array([])
    
    # Mix to mono if necessary (remains on CPU)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Remove batch dim
    y_cpu = waveform.squeeze(0)  # Shape: (total_samples,)

    # --- RMS (Loudness) Calculation in Batches ---
    
    # Use a sliding window approach with batched chunks to avoid allocating 
    # the entire unfolded tensor on GPU, which causes OOM on large files.
    
    CHUNK_SIZE = 48000 * 60  # ~1 minute chunks
    total_samples = y_cpu.shape[0]
    
    rms_values = []
    
    # Pad the signal if it's shorter than one frame 
    if total_samples < frame_length:
        y_cpu = torch.nn.functional.pad(y_cpu, (0, frame_length - total_samples))
        total_samples = y_cpu.shape[0]

    # Process in chunks ensuring proper window alignment.
    # Calculate total expected frames based on total samples and hop length.
    
    num_frames_total = (total_samples - frame_length) // hop_length + 1
    if num_frames_total <= 0:
        num_frames_total = 0
        
    # Iterate over OUTPUT frames
    FRAMES_PER_BATCH = 4096 * 4  # Process ~16k frames at a time
    
    pbar_rms = tqdm(total=num_frames_total, desc="Audio RMS", unit="fr")
    
    current_frame = 0
    while current_frame < num_frames_total:
        end_frame = min(current_frame + FRAMES_PER_BATCH, num_frames_total)
        count = end_frame - current_frame
        
        # Calculate sample range required for this batch of frames
        start_sample = current_frame * hop_length
        end_sample = (end_frame - 1) * hop_length + frame_length
        
        chunk_tensor = y_cpu[start_sample:end_sample].to(device)
        
        # Unfold on GPU
        # shape: (count, frame_length)
        windows = chunk_tensor.unfold(0, frame_length, hop_length)
        
        # Compute RMS for the batch
        rms_chunk = torch.sqrt(torch.mean(windows**2, dim=1))
        rms_values.append(rms_chunk)
        
        current_frame += count
        pbar_rms.update(count)
        
    pbar_rms.close()
    
    rms = torch.cat(rms_values) if rms_values else torch.tensor([], device=device)
    
    
    # --- Spectral Flux (STFT) in Batches ---
    
    # STFT also requires large intermediate buffers. We process in batches to keep VRAM usage low.
    # To match the original behavior (n_fft=2048, hop_length=512, center=True), 
    # we manually pad the input on CPU and use center=False during batch processing.
    
    window = torch.hann_window(2048).to(device)
    
    flux_values = []
    
    # Reuse valid batch size
    STFT_FRAMES_PER_BATCH = 4096 * 2
    
    # Simulate center=True by padding the CPU array once before splitting.
    pad_amount = 2048 // 2
    # Reflect padding requires at least 2D input in some PyTorch versions
    y_padded = torch.nn.functional.pad(y_cpu.unsqueeze(0), (pad_amount, pad_amount), mode='reflect').squeeze(0)
    
    # Calculate total frames for STFT from padded signal
    num_stft_frames = (y_padded.shape[0] - 2048) // hop_length + 1
    
    pbar_flux = tqdm(total=num_stft_frames, desc="Audio Flux", unit="fr")
    
    current_frame = 0
    # Initialize 'last_mag_col' as zeros (freq_bins,)
    # n_fft=2048 -> freq_bins=1025
    last_mag_col = torch.zeros(2048 // 2 + 1, device=device)
    
    while current_frame < num_stft_frames:
        end_frame = min(current_frame + STFT_FRAMES_PER_BATCH, num_stft_frames)
        count = end_frame - current_frame
        
        start_sample = current_frame * hop_length
        end_sample = (end_frame - 1) * hop_length + 2048
        
        chunk_tensor = y_padded[start_sample:end_sample].to(device)
        
        # Run STFT on this chunk. center=False because we manually padded y_padded
        # STFT shape: (freq_bins, count)
        stft_chunk = torch.stft(
            chunk_tensor, 
            n_fft=2048, 
            hop_length=hop_length, 
            window=window, 
            center=False, 
            return_complex=True
        )
        
        mag_chunk = torch.abs(stft_chunk)
        
        # Calculate Flux
        # We need [prev_last, curr_0, curr_1, ...]
        # Concatenate last_mag_col to the front
        
        # mag_chunk: (F, T)
        combined = torch.cat([last_mag_col.unsqueeze(1), mag_chunk], dim=1)
        
        # Diff: (F, T)
        diff = combined[:, 1:] - combined[:, :-1]
        
        # Flux: sum(diff^2) over freq, then sqrt
        flux_chunk = torch.sqrt(torch.sum(diff**2, dim=0))
        
        flux_values.append(flux_chunk)
        
        # Update last_mag_col
        last_mag_col = mag_chunk[:, -1]
        
        current_frame += count
        pbar_flux.update(count)
        
        # Cleanup
        del chunk_tensor, stft_chunk, mag_chunk, combined, diff
    
    pbar_flux.close()
    
    spectral_flux = torch.cat(flux_values) if flux_values else torch.tensor([], device=device)
    
    # --- Post Processing (same as before) ---
    
    # Match lengths
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

    # Smoothing
    def smooth_gpu(x: torch.Tensor, win: int = 21) -> torch.Tensor:
        if x.numel() == 0:
            return x
        if win > x.shape[0]:
            win = x.shape[0]
        if win % 2 == 0:
            win += 1
        padding = win // 2
        kernel = torch.ones(win, device=device) / win
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

    Robust to DECORD EOF issues: wraps get_batch with retries/chunking and
    allows configuring DECORD_EOF_RETRY_MAX via environment.
    """

    # Ensure a sufficiently high EOF retry limit before creating any VideoReader.
    try:
        max_retry = _get_env_int('DECORD_EOF_RETRY_MAX', 65536)
        os.environ['DECORD_EOF_RETRY_MAX'] = str(max_retry)
    except Exception:
        pass
    skip_tail = _get_env_int('DECORD_SKIP_TAIL_FRAMES', 0)

    ctx = gpu(0) if torch.cuda.is_available() else cpu(0)

    try:
        # First, read metadata on CPU to determine size without loading to GPU
        vr_cpu = VideoReader(str(video_path), ctx=cpu(0))
        h, w, _ = vr_cpu[0].shape
        orig_fps_probe = float(vr_cpu.get_avg_fps())
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
    orig_fps = float(vr.get_avg_fps()) if hasattr(vr, 'get_avg_fps') else orig_fps_probe
    eff_fps = min(float(fps), float(orig_fps))
    if eff_fps <= 0:
        eff_fps = max(1.0, float(fps))

    # Calculate indices to sample
    step = max(1, int(orig_fps / eff_fps))
    end_index_exclusive = len(vr) - max(0, int(skip_tail))
    if end_index_exclusive < 0:
        end_index_exclusive = 0
    indices = list(range(0, end_index_exclusive, step))

    if not indices:
        return np.array([]), np.array([])

    motions = []
    times = []

    # Helper: safe batched get with retries & chunking
    try:
        from decord._ffi.base import DECORDError as _DecordError
    except Exception:
        _DecordError = Exception  # Fallback

    def safe_get_batch(reader: VideoReader, idxs: List[int]) -> Tuple[Optional[torch.Tensor], List[int]]:
        if not idxs:
            return None, []
        try:
            frames = reader.get_batch(idxs)
            return frames, idxs
        except _DecordError as e:
            # Split and try smaller chunks
            if len(idxs) == 1:
                # Drop this problematic index
                logging.warning(
                    "\nDecord EOF retry exceeded for last indices; skipping tail frame index %s. Hint: increase DECORD_EOF_RETRY_MAX or set DECORD_SKIP_TAIL_FRAMES.",
                    idxs[0],
                )
                return None, []
            mid = len(idxs) // 2
            left_frames, left_kept = safe_get_batch(reader, idxs[:mid])
            right_frames, right_kept = safe_get_batch(reader, idxs[mid:])
            kept = left_kept + right_kept
            if not kept:
                return None, []
            if left_frames is None:
                return right_frames, right_kept
            if right_frames is None:
                return left_frames, left_kept
            try:
                return torch.cat([left_frames, right_frames], dim=0), kept
            except Exception:
                # As a last resort, fetch one-by-one preserving order
                tensors = []
                kept_final = []
                for ii in idxs:
                    try:
                        t = reader.get_batch([ii])
                        tensors.append(t)
                        kept_final.append(ii)
                    except Exception:
                        logging.warning("Skipping unreadable frame index %s at EOF.", ii)
                if not tensors:
                    return None, []
                return torch.cat(tensors, dim=0), kept_final
        except Exception:
            # Unknown error type; try to continue with singles
            tensors = []
            kept_final = []
            for ii in idxs:
                try:
                    t = reader.get_batch([ii])
                    tensors.append(t)
                    kept_final.append(ii)
                except Exception:
                    logging.warning("Skipping unreadable frame index %s due to unknown error.", ii)
            if not tensors:
                return None, []
            return torch.cat(tensors, dim=0), kept_final

    # Process in batches - small size to save memory
    batch_size = 2
    prev_batch_last = None

    total_batches = (len(indices) + batch_size - 1) // batch_size
    pbar = tqdm(total=total_batches, desc=f"Video action", unit="batch")

    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]
        frames, kept_idx = safe_get_batch(vr, batch_indices)
        if frames is None or len(kept_idx) == 0:
            pbar.update(1)
            continue

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

        # Timestamps; align to kept indices actually read
        batch_times = torch.tensor(kept_idx, device=gray.device).float() / orig_fps
        times.append(batch_times)

        prev_batch_last = gray[-1]

        del frames, frames_small, gray, diffs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pbar.update(1)

    pbar.close()

    if len(motions) == 0:
        return np.array([]), np.array([])

    motions = torch.cat(motions)
    times = torch.cat(times)

    # Normalize and smooth (similar to audio)
    if motions.numel() == 0:
        return np.array([]), np.array([])
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


def get_render_params(
    video_path: Path,
    start_point: float,
    final_clip_length: float,
    config: ProcessingConfig,
) -> RenderParams:
    """Calculate all parameters needed for rendering the final clip."""

    # Use decord CPU to get dimensions quickly
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
        h, w, _ = vr[0].shape
    except Exception:
        # Fallback to GPU context if CPU fails (unlikely)
        vr = VideoReader(str(video_path))
        h, w, _ = vr[0].shape

    # Calculate crop parameters (same logic as before: crop to target ratio)
    current_ratio = w / h
    target_ratio = config.target_ratio_w / config.target_ratio_h

    if current_ratio > target_ratio:
        # Too wide, crop width
        new_width = round(h * config.target_ratio_w / config.target_ratio_h)
        crop_w = new_width
        crop_h = h
        crop_x = int(w * config.x_center - crop_w / 2)
        crop_y = int(h * config.y_center - crop_h / 2)
    else:
        # Too tall, crop height
        new_height = round(w / config.target_ratio_w * config.target_ratio_h)
        crop_w = w
        crop_h = new_height
        crop_x = int(w * config.x_center - crop_w / 2)
        crop_y = int(h * config.y_center - crop_h / 2)

    # Clamp crop coordinates
    crop_x = max(0, min(w - crop_w, crop_x))
    crop_y = max(0, min(h - crop_h, crop_y))

    # Calculate background/output resolution
    bg_w, bg_h = select_background_resolution(crop_w)

    # Logic from get_final_clip to determine layout
    is_vertical_bg = False

    if crop_w >= crop_h:
        # Landscape/Squareish
        # "background_clip = background_clip.resized(width=720, height=720)"
        # "result_clip = result_clip.resized(width=bg_w)" -> final output is bg_w x bg_w
        # This implies we want a square output if the main clip is landscape/square
        bg_h = bg_w # Force square output
        is_vertical_bg = False
    elif crop_w / 9 < crop_h / 16:
         # Very tall portrait
         is_vertical_bg = True
    else:
        # Default fallback
        pass

    return RenderParams(
        source_path=video_path,
        start_time=start_point,
        duration=final_clip_length,
        output_width=bg_w,
        output_height=bg_h,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_w=crop_w,
        crop_h=crop_h,
        bg_width=bg_w,
        bg_height=bg_h,
        is_vertical_bg=is_vertical_bg
    )


def render_video_gpu(
        params: RenderParams,
        output_path: Path,
        max_error_depth: int = 3,
) -> None:
    """Render the clip using GPU compositing and FFMPEG NVENC (Optimized)."""

    logging.info(f"Rendering GPU: {output_path.name}")

    # 1. Extract audio
    temp_audio = output_path.with_suffix(".aac")
    cmd_audio = [
        "ffmpeg", "-y",
        "-ss", f"{params.start_time:.3f}",
        "-t", f"{params.duration:.3f}",
        "-i", str(params.source_path),
        "-vn", "-acodec", "copy",
        str(temp_audio)
    ]
    subprocess.run(cmd_audio, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    # 2. Setup FFMPEG process
    fps = 30.0
    try:
        vr_probe = VideoReader(str(params.source_path), ctx=cpu(0))
        src_fps = vr_probe.get_avg_fps()
        fps = min(src_fps, 60.0)
        del vr_probe  # Clean up immediately
    except:
        fps = 30.0

    cmd_ffmpeg = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{params.output_width}x{params.output_height}",
        "-pix_fmt", "rgb24",
        "-r", f"{fps}",
        "-i", "-",
        "-i", str(temp_audio),
        "-c:v", "hevc_nvenc",  # Use hardware encoder
        "-preset", "p7",
        "-tune", "hq",
        "-rc", "vbr",
        "-cq", "23",  # Slightly increased CQ to reduce bitrate spikes
        "-maxrate", "80M",  # Cap bitrate to prevent buffer bloat
        "-bufsize", "100M",
        "-pix_fmt", "yuv420p",
        "-g", f"{int(fps * 2)}",
        "-bf", "2",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_path)
    ]

    if not temp_audio.exists():
        cmd_ffmpeg = [x for x in cmd_ffmpeg if x not in ["-i", str(temp_audio), "-c:a", "aac"]]

    # redirect stderr to a file to prevent buffer deadlock
    log_path = output_path.with_suffix(".ffmpeg.log")
    ffmpeg_log = open(log_path, "w")

    try:
        process = subprocess.Popen(
            cmd_ffmpeg,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=ffmpeg_log
        )
    except Exception:
        ffmpeg_log.close()
        raise

    # 3. GPU Rendering Loop
    # Use torch.no_grad() to prevent graph building overhead
    with torch.no_grad():
        try:
            ctx = gpu(0) if torch.cuda.is_available() else cpu(0)

            # Calculation of dims (omitted full recalc for brevity, logic stays same)
            fg_w = params.output_width
            fg_h = int(params.crop_h * (params.output_width / params.crop_w))

            # ... [Background dims logic from original script stays here] ...
            # Re-implementing simplified BG logic for context:
            vr_temp = VideoReader(str(params.source_path), ctx=cpu(0))
            src_h, src_w, _ = vr_temp[0].shape
            del vr_temp  # Important: delete CPU reader immediately

            if params.is_vertical_bg:
                bg_ratio_w, bg_ratio_h = 9, 16
                if (src_w / src_h) > (bg_ratio_w / bg_ratio_h):
                    bg_crop_h, bg_crop_w = src_h, int(src_h * bg_ratio_w / bg_ratio_h)
                else:
                    bg_crop_w, bg_crop_h = src_w, int(src_w * bg_ratio_h / bg_ratio_w)
                bg_crop_x, bg_crop_y = int(src_w * 0.5 - bg_crop_w / 2), int(src_h * 0.5 - bg_crop_h / 2)
            else:
                bg_dim = min(src_w, src_h)
                bg_crop_w, bg_crop_h = bg_dim, bg_dim
                bg_crop_x, bg_crop_y = int(src_w * 0.5 - bg_crop_w / 2), int(src_h * 0.5 - bg_crop_h / 2)

            # Main Reader - Created ONCE. Do not recreate in loop.
            vr = VideoReader(str(params.source_path), ctx=ctx)

            total_frames = int(params.duration * fps)
            src_fps_val = vr.get_avg_fps()

            # Pre-calculate indices to avoid math in loop
            frame_indices = [int((params.start_time + (i / fps)) * src_fps_val) for i in range(total_frames)]

            BATCH_SIZE = 4
            total_batches = (len(frame_indices) + BATCH_SIZE - 1) // BATCH_SIZE

            log_memory_usage("Render Start")

            with tqdm(total=total_batches, desc="Video render", unit="batch") as pbar_render:
                for i in range(0, len(frame_indices), BATCH_SIZE):
                    if i % (BATCH_SIZE * 50) == 0:
                        logging.info(f"Rendering batch {i // BATCH_SIZE}/{total_batches}")

                    # REMOVED: The block causing the leak (del vr / new vr)

                    if process.poll() is not None:
                        logging.error("FFMPEG died")
                        break

                    batch_idxs = frame_indices[i: i + BATCH_SIZE]
                    # Clamp indices
                    max_idx = len(vr) - 1
                    batch_idxs = [min(x, max_idx) for x in batch_idxs]

                    try:
                        frames = vr.get_batch(batch_idxs)  # (B, H, W, 3)
                    except Exception as e:
                        logging.warning(f"Batch read failed: {e}")
                        break

                    # 1. Background Processing
                    bg_frames = frames[:, bg_crop_y:bg_crop_y + bg_crop_h, bg_crop_x:bg_crop_x + bg_crop_w, :]
                    bg_frames = bg_frames.permute(0, 3, 1, 2).float()  # to NCHW

                    # Resize for blur (low res)
                    blur_w, blur_h = 720, (1280 if params.is_vertical_bg else 720)
                    bg_small = torch.nn.functional.interpolate(
                        bg_frames, size=(blur_h, blur_w), mode='bilinear', align_corners=False
                    )

                    # Blur via Cupy
                    bg_small = bg_small.permute(0, 2, 3, 1).contiguous()  # back to NHWC for cupy
                    cp_bg = cp.from_dlpack(torch.to_dlpack(bg_small))
                    f_bg = cp_bg.astype(float)
                    blurred_bg_cp = cupyx.scipy.ndimage.gaussian_filter(f_bg, sigma=(0, 16, 16, 0))
                    # Explicit cast back to match original dtype usually helps, but float is fine for interpolate
                    blurred_bg = torch.utils.dlpack.from_dlpack(blurred_bg_cp).float()

                    # Resize to Final Output
                    blurred_bg = blurred_bg.permute(0, 3, 1, 2)  # NCHW
                    final_bg = torch.nn.functional.interpolate(
                        blurred_bg, size=(params.output_height, params.output_width), mode='bilinear',
                        align_corners=False
                    )

                    # 2. Foreground Processing
                    fg_frames = frames[
                        :, params.crop_y:params.crop_y + params.crop_h, params.crop_x:params.crop_x + params.crop_w, :]
                    fg_frames = fg_frames.permute(0, 3, 1, 2).float()
                    final_fg = torch.nn.functional.interpolate(
                        fg_frames, size=(fg_h, fg_w), mode='bilinear', align_corners=False
                    )

                    # 3. Composite (Overlay)
                    y_off, x_off = (params.output_height - fg_h) // 2, (params.output_width - fg_w) // 2
                    y1, y2 = max(0, y_off), min(params.output_height, y_off + fg_h)
                    x1, x2 = max(0, x_off), min(params.output_width, x_off + fg_w)
                    sy1, sx1 = max(0, -y_off), max(0, -x_off)

                    # Direct tensor insertion
                    if (y2 > y1) and (x2 > x1):
                        final_bg[:, :, y1:y2, x1:x2] = final_fg[:, :, sy1:(sy1 + (y2 - y1)), sx1:(sx1 + (x2 - x1))]

                    # 4. Write to Pipe
                    # Convert to byte and move to CPU
                    # .detach() ensures no grad tracking (redundant with torch.no_grad but safe)
                    out_tensor = final_bg.permute(0, 2, 3, 1).contiguous().byte()
                    out_bytes = out_tensor.cpu().numpy().tobytes()

                    try:
                        process.stdin.write(out_bytes)
                    except BrokenPipeError:
                        break

                    # 5. Explicit Cleanup (Critical for Loop)
                    del frames, bg_frames, bg_small, blurred_bg, final_bg, fg_frames, final_fg, out_tensor, out_bytes, cp_bg, blurred_bg_cp

                    # Periodic GC - keep it, but less frequent is fine
                    if i > 0 and (i // BATCH_SIZE) % 100 == 0:
                        gc.collect()

                    pbar_render.update(1)

        except Exception as e:
            logging.error(f"Error during GPU render: {e}", exc_info=True)
        finally:
            # Clean up processes and memory
            if process:
                try:
                    process.stdin.close()
                except:
                    pass
                process.wait()

            if 'ffmpeg_log' in locals() and ffmpeg_log:
                ffmpeg_log.close()

            if temp_audio.exists():
                temp_audio.unlink()

            # Final memory sweep
            if 'vr' in locals(): del vr
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()


def render_video_gpu_isolated(*args, **kwargs) -> None:
    """Runs render_video_gpu in a separate process to ensure memory cleanup."""
    # Use 'spawn' to ensure a fresh process space
    ctx = multiprocessing.get_context('spawn')
    p = ctx.Process(
        target=render_video_gpu,
        args=args,
        kwargs=kwargs
    )
    p.start()
    p.join()

    if p.exitcode != 0:
        logging.error("Render process failed with exit code %s", p.exitcode)
        # If exit code is 137 (128+9) or -9, it was likely OOM killed.
        if p.exitcode == -9 or p.exitcode == 137:
             logging.error("Render process was likely OOM killed.")


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

    # We need to get video duration efficiently. Use VideoReader on CPU.
    try:
        vr_probe = VideoReader(str(video_file), ctx=cpu(0))
        video_duration = len(vr_probe) / vr_probe.get_avg_fps()
        del vr_probe
    except Exception:
        # Fallback to MoviePy if decord fails (legacy support)
        logging.warning("Decord probe failed, using MoviePy to check duration.")
        from moviepy import VideoFileClip
        video_clip = VideoFileClip(str(video_file))
        video_duration = video_clip.duration
        video_clip.close()

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

            render_file_name = f"{video_file.stem} scene-{i}{video_file.suffix}"
            render_path = output_dir / render_file_name

            # Prepare render params
            params = get_render_params(
                video_file,
                best_start,
                float(short_length),
                config
            )

            # Execute GPU render
            render_video_gpu_isolated(
                params,
                render_path,
                max_error_depth=config.max_error_depth,
            )
    else:
        # No scenes found, fallback to random clip
        short_length = random.randint(
            config.min_short_length, config.max_short_length
        )

        if video_duration < config.max_short_length:
            adapted_short_length = min(math.floor(video_duration), short_length)
        else:
            adapted_short_length = short_length

        min_start_point = min(10, math.floor(video_duration) - adapted_short_length)
        max_start_point = math.floor(video_duration - adapted_short_length)

        start_point = float(random.randint(int(min_start_point), int(max_start_point)))

        params = get_render_params(
            video_file,
            start_point,
            float(adapted_short_length),
            config
        )

        render_video_gpu_isolated(
            params,
            output_dir / video_file.name,
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
