"""Utility for automatically generating short video clips using GPU acceleration.

This module processes gameplay videos and creates resized clips
that fit common short-video aspect ratios. It leverages NVIDIA GPU
(CUDA) for scene detection, audio/video analysis, image filtering,
and video encoding to maximize performance.

Now enhanced with AI-agentic semantic analysis (Gemini/OpenAI) and
artistic subtitle generation (Whisper + PyCaps).
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import os

# Force NVRTC to use C++17 to avoid "libcu++ requires at least C++ 17" error
# CuPy uses NVRTC for JIT compilation, which needs these flags
os.environ["NVCC_APPEND_FLAGS"] = "-std=c++17 --allow-unsupported-compiler"
os.environ["CCCL_IGNORE_DEPRECATED_CPP_DIALECT"] = "1"  # Silence libcu++ C++11 warning
# CUPY_NVRTC environment variable to pass options to the JIT compiler
os.environ["CUPY_NVCC_FLAGS"] = "-std=c++17 --allow-unsupported-compiler"

import gc
import multiprocessing
import tempfile
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

# AI Provider and Subtitle modules
from ai_providers import (
    get_analyzer,
    ClipScore,
    extract_candidate_clip,
)
from subtitle_generator import generate_subtitles, is_subtitles_enabled

# Load environment variables from a .env file if present.
load_dotenv()

# Configure basic logging.
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Ensure decord uses the correct backend
decord.bridge.set_bridge('torch')


# =============================================================================
# DEBUG CHECKPOINT SYSTEM
# Set DEBUG_SKIP_ANALYSIS=1 to skip video/audio analysis and use cached results
# Set DEBUG_RENDERED_CLIPS="path1,path2" to skip rendering and use existing clips
# =============================================================================

import pickle

def _get_debug_checkpoint_path(video_file: Path) -> Path:
    """Get the checkpoint file path for a video."""
    return video_file.with_suffix(".debug_checkpoint.pkl")


def _save_debug_checkpoint(
    video_file: Path,
    analysis_result: "VideoAnalysisResult",
    audio_times: np.ndarray,
    audio_score: np.ndarray,
    processed_scenes: List,
    ai_ranked: Optional[List] = None,
    scene_categories: Optional[dict] = None,
    rendered_clips: Optional[List] = None,
) -> Path:
    """Save intermediate analysis results for debugging.
    
    This allows you to skip expensive video/audio analysis and AI ranking
    when debugging pycaps, TTS, and caption generation.
    """
    checkpoint_path = _get_debug_checkpoint_path(video_file)
    
    # Convert _SecondsTime objects to serializable format
    scenes_data = [
        (s[0].get_seconds(), s[1].get_seconds()) 
        for s in analysis_result.scenes
    ]
    
    processed_scenes_data = [
        (s[0].get_seconds(), s[1].get_seconds()) 
        for s in processed_scenes
    ]
    
    # Convert AI ranked results
    ai_ranked_data = None
    if ai_ranked:
        ai_ranked_data = [
            ((s[0].get_seconds(), s[1].get_seconds()), score, category)
            for s, score, category in ai_ranked
        ]
    
    # Convert rendered clips
    rendered_clips_data = None
    if rendered_clips:
        rendered_clips_data = [(str(p), cat) for p, cat in rendered_clips]
    
    checkpoint = {
        "video_file": str(video_file),
        "scenes": scenes_data,
        "video_times": analysis_result.video_times,
        "video_scores": analysis_result.video_scores,
        "audio_times": audio_times,
        "audio_score": audio_score,
        "processed_scenes": processed_scenes_data,
        "ai_ranked": ai_ranked_data,
        "scene_categories": scene_categories,
        "rendered_clips": rendered_clips_data,
    }
    
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)
    
    logging.info(f"[DEBUG] Saved checkpoint to: {checkpoint_path}")
    return checkpoint_path


def _load_debug_checkpoint(video_file: Path) -> Optional[dict]:
    """Load saved checkpoint data if it exists."""
    checkpoint_path = _get_debug_checkpoint_path(video_file)
    
    if not checkpoint_path.exists():
        logging.warning(f"[DEBUG] No checkpoint found: {checkpoint_path}")
        return None
    
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
    
    # Reconstruct _SecondsTime objects
    checkpoint["scenes"] = [
        (_SecondsTime(s), _SecondsTime(e)) 
        for s, e in checkpoint["scenes"]
    ]
    
    checkpoint["processed_scenes"] = [
        (_SecondsTime(s), _SecondsTime(e)) 
        for s, e in checkpoint["processed_scenes"]
    ]
    
    if checkpoint.get("ai_ranked"):
        checkpoint["ai_ranked"] = [
            ((_SecondsTime(s), _SecondsTime(e)), score, category)
            for (s, e), score, category in checkpoint["ai_ranked"]
        ]
    
    if checkpoint.get("rendered_clips"):
        checkpoint["rendered_clips"] = [
            (Path(p), cat) for p, cat in checkpoint["rendered_clips"]
        ]
    
    logging.info(f"[DEBUG] Loaded checkpoint from: {checkpoint_path}")
    return checkpoint


def _is_debug_skip_analysis() -> bool:
    """Check if we should skip analysis and use cached checkpoint."""
    return os.getenv("DEBUG_SKIP_ANALYSIS", "").lower() in ("1", "true", "yes")


def _is_debug_skip_render() -> bool:
    """Check if we should skip rendering and use existing clips."""
    return os.getenv("DEBUG_SKIP_RENDER", "").lower() in ("1", "true", "yes")


def _get_debug_rendered_clips() -> Optional[List[Tuple[Path, str]]]:
    """Get manually specified rendered clips for debugging."""
    clips_str = os.getenv("DEBUG_RENDERED_CLIPS", "")
    if not clips_str:
        return None
    
    clips = []
    for item in clips_str.split(","):
        item = item.strip()
        if not item:
            continue
        # Format: "path:category" or just "path" (default category: action)
        if ":" in item:
            path_str, category = item.rsplit(":", 1)
        else:
            path_str, category = item, "action"
        
        path = Path(path_str)
        if path.exists():
            clips.append((path, category))
        else:
            logging.warning(f"[DEBUG] Clip not found: {path}")
    
    return clips if clips else None


# =============================================================================


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



@dataclass
class VideoAnalysisResult:
    scenes: List[Tuple[_SecondsTime, _SecondsTime]]
    video_times: np.ndarray
    video_scores: np.ndarray


def analyze_video_content(
    video_path: Path, 
    scene_threshold: float = 27.0,
    action_fps: int = 6
) -> VideoAnalysisResult:
    """Unified pass for Scene Detection and Action Profiling.
    
    Reads video once using GPU acceleration to compute:
    1. Scene cuts (using HSV diff + FlashFilter logic)
    2. Visual Action Score (using pixel diffs on subsampled frames)
    
    Args:
        video_path: Path to video file.
        scene_threshold: Threshold for scene cuts.
        action_fps: Target FPS for action score sampling.
        
    Returns:
        VideoAnalysisResult containing scenes and action scores.
    """
    import cv2
    
    # 1) Probe video for resolution and FPS (CPU lightweight probe)
    try:
        vr_probe = VideoReader(str(video_path), ctx=cpu(0))
        h0, w0, _ = vr_probe[0].shape
        fps = float(vr_probe.get_avg_fps())
        del vr_probe
    except Exception:
        # Fallback
        vr_probe = VideoReader(str(video_path))
        h0, w0, _ = vr_probe[0].shape
        fps = float(vr_probe.get_avg_fps())
        del vr_probe
        
    if fps <= 0:
        fps = 30.0

    # 2) Calculate common resolution (SceneDetect standard: ~256px wide)
    # This is small enough for fast action diffing too.
    TARGET_MIN_WIDTH = 256
    if w0 < TARGET_MIN_WIDTH:
        downscale = 1.0
    else:
        downscale = w0 / float(TARGET_MIN_WIDTH)

    w_eff = max(1, int(w0 / downscale))
    h_eff = max(1, int(h0 / downscale))

    # 3) Setup VideoReader on GPU
    ctx = gpu(0) if torch.cuda.is_available() else cpu(0)
    try:
        # Batch size 96 is a good balance between throughput and responsiveness
        # num_threads=4 for background decoding
        vr = VideoReader(str(video_path), ctx=ctx, width=w_eff, height=h_eff, num_threads=4)
        using_gpu = True
    except Exception as e:
        logging.warning(f"VideoReader GPU open failed: {e}. Falling back to CPU.")
        vr = VideoReader(str(video_path), ctx=cpu(0), width=w_eff, height=h_eff, num_threads=8)
        using_gpu = False

    frame_count = len(vr)
    if frame_count == 0:
        return VideoAnalysisResult([], np.array([]), np.array([]))

    # --- Setup Scene Detection State ---
    class _FlashFilterMerge:
        def __init__(self, length: int):
            self._filter_length = int(length)
            self._last_above: Optional[int] = None
            self._merge_enabled: bool = False
            self._merge_triggered: bool = False
            self._merge_start: Optional[int] = None

        def filter(self, frame_num: int, above_threshold: bool) -> List[int]:
            if not (self._filter_length > 0):
                return [frame_num] if above_threshold else []
            if self._last_above is None:
                self._last_above = frame_num
            # MERGE path logic locally inlined or method called
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

    min_scene_len = 15
    flash_filter = _FlashFilterMerge(length=min_scene_len)
    
    # We will track last_hsv TENSOR on GPU
    last_hsv_tensor: Optional[torch.Tensor] = None
    
    # Accumulate scores on GPU/CPU list to avoid sync
    all_scene_scores = []
    
    # --- Setup Action Profiling State ---
    action_step = max(1, int(fps / action_fps))
    all_action_motions = [] 
    all_action_times = []
    prev_action_frame_tensor = None # Keep on GPU (or CPU if context is CPU)

    # --- Processing Loop ---
    # With Async GPU logic, we can max out VRAM batch size
    batch_size = 128
    total_batches = (frame_count + batch_size - 1) // batch_size
    
    pbar = tqdm(total=total_batches, desc="Analyzing video", unit="batch")

    for i in range(0, frame_count, batch_size):
        end_idx = min(i + batch_size, frame_count)
        
        # Read batch (GPU if avail)
        try:
            frames_t = vr.get_batch(range(i, end_idx))
            # frames_t: (B, H, W, 3) (uint8)
        except Exception:
            logging.error(f"Failed to read batch {i}-{end_idx}")
            continue

        # --- Scene Detection Part (PURE GPU in FP32) ---
        # 1. Convert to Float/HSV on GPU
        # frames_t is (B, H, W, 3) RGB
        
        # Cast to FLOAT (FP32) for precision in scene detection
        frames_float = frames_t.float()
        
        # Convert to HSV (vectorized)
        # Helper expects (B, 3, H, W) input usually? 
        # Let's adapt frames_float first: (B, H, W, 3) -> (B, 3, H, W)
        frames_permuted = frames_float.permute(0, 3, 1, 2)
        hsv_batch = rgb_to_hsv_torch(frames_permuted) # returns (B, 3, H, W) in FP16
        
        # We need differences between ADJACENT frames.
        # Current batch: [F0, F1, ..., F_Last]
        # We need [Prev, F0], [F0, F1], ...
        
        # Prepare comparison tensor
        # If we have last_hsv from previous batch, prepend it.
        if last_hsv_tensor is not None:
            # last_hsv_tensor is (3, H, W)
            # Unsqueeze to (1, 3, H, W) and cat
            combined_hsv = torch.cat([last_hsv_tensor.unsqueeze(0), hsv_batch], dim=0)
        else:
            # First batch. First frame diff is 0 by definition.
            combined_hsv = torch.cat([hsv_batch[0:1], hsv_batch], dim=0)

        # Diff: (B, 3, H, W)
        # diff[k] = abs(combined[k+1] - combined[k])
        diffs = torch.abs(combined_hsv[1:] - combined_hsv[:-1])
        
        # Compute mean per channel per frame
        # Mean over H (dim 2) and W (dim 3)
        # shape: (B, 3)
        means = diffs.mean(dim=(2, 3))
        
        # Sum compoents and divide by 3 (Frame Score)
        # shape: (B,)
        batch_scores = means.sum(dim=1) / 3.0
        
        # Accumulate scores (keep on GPU? Or move to CPU list async?)
        # Moving to CPU list is fine if we don't sync.
        # But actually appending tensor to list and cat() later on GPU is faster?
        # Let's keep on GPU for now to avoid ANY implicit sync.
        all_scene_scores.append(batch_scores)

        # Save last frame of this batch for next iteration
        last_hsv_tensor = hsv_batch[-1].clone()

        # --- Action Profiling Part (Keep on Tensor/GPU) ---
        # We only need every 'action_step' frame.
        # We can select them directly from 'frames_t'
        
        # Indices in this batch that match the step cadence
        # Global idx: K = i + j
        # We want K % action_step == 0
        
        # First index in batch is 'i'.
        # offset to first matching frame: (action_step - (i % action_step)) % action_step
        start_offset = (action_step - (i % action_step)) % action_step
        
        # Slice the batch
        # frames_t[start_offset::action_step]
        if start_offset < batch_size:
            action_frames = frames_t[start_offset::action_step].half() # USE FP16
        else:
            action_frames = torch.tensor([], device=frames_t.device, dtype=torch.float16)
        
        if len(action_frames) > 0:
            # Grayscale: 0.299R + 0.587G + 0.114B
            # frames_t is RGB (B, H, W, 3)
            gray = (action_frames[..., 0] * 0.299 +
                    action_frames[..., 1] * 0.587 +
                    action_frames[..., 2] * 0.114)
            
            # Diff against previous
            if prev_action_frame_tensor is not None:
                combined = torch.cat([prev_action_frame_tensor.unsqueeze(0), gray])
                diffs = torch.abs(combined[1:] - combined[:-1])
            else:
                # First batch ever
                combined = torch.cat([gray[0:1], gray])
                diffs = torch.abs(combined[1:] - combined[:-1])
                diffs[0] = 0.0
            
            # Mean diff per frame
            batch_motions = diffs.mean(dim=(1, 2)) # (B_subset,)
            # Keep on GPU
            all_action_motions.append(batch_motions)
            
            # Calculate times
            # Local indices inside batch
            local_indices = torch.arange(start_offset, frames_t.shape[0], action_step)
            global_indices = i + local_indices
            batch_times = global_indices.float() / fps
            all_action_times.append(batch_times)
            
            prev_action_frame_tensor = gray[-1]

        # Cleanup
        del frames_t, frames_float, frames_permuted, hsv_batch, combined_hsv, diffs, means, batch_scores
        pbar.update(1)

    pbar.close()

    # --- Post-Process Scenes (CPU One-Shot) ---
    cut_indices: List[int] = []
    if all_scene_scores:
        # 1. Cat all scores (GPU)
        full_scores_gpu = torch.cat(all_scene_scores)
        # 2. Move to CPU once (and cast to float32 for numpy)
        full_scores_cpu = full_scores_gpu.float().cpu().numpy()
        
        # 3. Iterate fast on CPU
        for frame_num, score in enumerate(full_scores_cpu):
             above = score >= scene_threshold
             if frame_num == 0: above = False
             
             emitted = flash_filter.filter(frame_num, above)
             if emitted:
                 cut_indices.extend(emitted)
    
    # --- Finalize Scenes ---
    if not cut_indices:
        scenes = []
    else:
        cut_indices = sorted(set(cut_indices))
        scenes = []
        last_cut = 0
        for cut in cut_indices:
            scenes.append((_SecondsTime(last_cut / fps), _SecondsTime(cut / fps)))
            last_cut = cut
        scenes.append((_SecondsTime(last_cut / fps), _SecondsTime(frame_count / fps)))

    # --- Finalize Action Scores ---
    if not all_action_motions:
        video_scores = np.array([])
        video_times = np.array([])
    else:
        motions = torch.cat(all_action_motions).float().cpu() # Move to CPU at end
        video_times = torch.cat(all_action_times).float().cpu().numpy()
        
        # Normalize
        if motions.numel() == 0:
            video_scores = np.array([])
        else:
            std_val = motions.std()
            if std_val == 0:
                motions_norm = motions
            else:
                motions_norm = (motions - motions.mean()) / (std_val + 1e-8)
            
            # Smooth
             # cpu-side smoothing is fine for 1D array
            motions_np = motions_norm.numpy()
            win = int(fps)
            if win > 1:
                window = np.ones(win) / win
                video_scores = np.convolve(motions_np, window, mode='same')
            else:
                video_scores = motions_np

    return VideoAnalysisResult(scenes, video_times, video_scores)


def rgb_to_hsv_torch(image: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """Convert a batch of RGB images to HSV using PyTorch.

    Args:
        image: RGB tensor of shape (B, 3, H, W). Values can be 0-255 or 0-1.
               It is assumed that if max > 1.0, it is 0-255 data.

    Returns:
        HSV tensor of shape (B, 3, H, W).
        H: [0, 180] (to match OpenCV's Hue range for 8-bit images, though we use float here).
           Actually, let's strictly MATCH OpenCV semantic for `ContentDetector`:
           OpenCV BGR2HSV:
             H: 0-179 (Hue/2)
             S: 0-255
             V: 0-255
           
           Our input is Float. We should normalize to 0-1 first for calculation, 
           then scale back to 0-255 / 0-180.
    """
    if image.max() > 1.0:
        image = image / 255.0

    r, g, b = image[:, 0], image[:, 1], image[:, 2]
    
    max_val, _ = image.max(dim=1)
    min_val, _ = image.min(dim=1)
    diff = max_val - min_val
    
    # Value (0.0 - 1.0)
    v = max_val
    
    # Saturation (0.0 - 1.0)
    # s = diff / v
    # handle v=0
    s = torch.where(v > epsilon, diff / v, torch.zeros_like(diff))
    
    # Hue
    # if r==max: (g-b)/diff
    # if g==max: 2 + (b-r)/diff
    # if b==max: 4 + (r-g)/diff
    
    # To avoid div by zero in diff:
    diff_safe = torch.where(diff < epsilon, torch.ones_like(diff), diff)
    
    h = torch.zeros_like(v)
    
    mask_r = (r == max_val)
    mask_g = (g == max_val) & (~mask_r)
    mask_b = (b == max_val) & (~mask_r) & (~mask_g)
    
    h[mask_r] = (g[mask_r] - b[mask_r]) / diff_safe[mask_r]
    h[mask_g] = 2.0 + (b[mask_g] - r[mask_g]) / diff_safe[mask_g]
    h[mask_b] = 4.0 + (r[mask_b] - g[mask_b]) / diff_safe[mask_b]
    
    h = (h / 6.0) % 1.0
    
    # Scale to OpenCV ranges
    # H: 0-179 -> 180 * h
    # S: 0-255 -> 255 * s
    # V: 0-255 -> 255 * v
    
    out_h = h * 179.0 # Use 179 to be safe or 180? OpenCV uses 0-179 inclusive.
    out_s = s * 255.0
    out_v = v * 255.0
    
    return torch.stack([out_h, out_s, out_v], dim=1)



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
    
    # CHUNK_SIZE = 48000 * 60  # ~1 minute chunks (Unused)
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


def blur_gpu(image_tensor: torch.Tensor, sigma: float = 8.0) -> torch.Tensor:
    """Return a blurred version of ``image_tensor`` using PyTorch.
    
    Args:
        image_tensor: (H, W, 3) torch tensor on GPU (uint8 or float).
        sigma: Blur sigma.

    Returns:
        Blurred torch tensor (H, W, 3).
    """
    return gaussian_blur_torch(image_tensor, sigma)


def gaussian_blur_torch(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply Gaussian blur using pure PyTorch (no CuPy).
    
    Works on (H, W, C) or (N, H, W, C) tensors.
    Uses separable 1D convolutions for efficiency.
    """
    if sigma <= 0:
        return image
    
    original_dtype = image.dtype
    device = image.device
    
    # Work in NCHW format for conv2d
    if image.ndim == 3:
        # (H, W, C) -> (1, C, H, W)
        img = image.float().permute(2, 0, 1).unsqueeze(0)
        squeeze_output = True
    elif image.ndim == 4:
        # (N, H, W, C) -> (N, C, H, W)
        img = image.float().permute(0, 3, 1, 2)
        squeeze_output = False
    else:
        return image
    
    # Create 1D Gaussian kernel
    kernel_size = int(6 * sigma + 1) | 1  # Ensure odd
    x = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Reshape for grouped conv2d (each channel independently)
    C = img.shape[1]
    kernel_h = kernel_1d.view(1, 1, -1, 1).expand(C, 1, -1, 1)  # (C, 1, K, 1)
    kernel_w = kernel_1d.view(1, 1, 1, -1).expand(C, 1, 1, -1)  # (C, 1, 1, K)
    
    padding_h = kernel_size // 2
    padding_w = kernel_size // 2
    
    # Apply separable blur
    blurred = torch.nn.functional.conv2d(img, kernel_h, padding=(padding_h, 0), groups=C)
    blurred = torch.nn.functional.conv2d(blurred, kernel_w, padding=(0, padding_w), groups=C)
    
    # Convert back to original format
    if squeeze_output:
        # (1, C, H, W) -> (H, W, C)
        blurred = blurred.squeeze(0).permute(1, 2, 0)
    else:
        # (N, C, H, W) -> (N, H, W, C)
        blurred = blurred.permute(0, 2, 3, 1)
    
    # Restore dtype if needed
    if original_dtype == torch.uint8:
        blurred = blurred.clamp(0, 255).to(torch.uint8)
    
    return blurred


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
    except Exception:
        fps = 30.0

    cmd_ffmpeg = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{params.output_width}x{params.output_height}",
        "-pix_fmt", "rgb24",
        "-r", f"{fps}",
        "-i", "-",
    ]

    has_valid_audio = temp_audio.exists() and temp_audio.stat().st_size > 0

    if has_valid_audio:
        cmd_ffmpeg.extend(["-i", str(temp_audio)])

    cmd_ffmpeg.extend([
        "-c:v", "hevc_nvenc",  # Use hardware encoder
        "-preset", "slow",
        "-tune", "hq",
        "-rc", "vbr",
        "-cq", "23",  # Slightly increased CQ to reduce bitrate spikes
        "-maxrate", "80M",  # Cap bitrate to prevent buffer bloat
        "-bufsize", "100M",
        "-pix_fmt", "yuv420p",
        "-g", f"{int(fps * 2)}",
        "-bf", "2",
    ])

    if has_valid_audio:
        cmd_ffmpeg.extend(["-c:a", "aac", "-b:a", "192k"])

    cmd_ffmpeg.extend(["-shortest", str(output_path)])

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

            BATCH_SIZE = 16
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

                    # Blur via PyTorch (no CuPy needed)
                    bg_small = bg_small.permute(0, 2, 3, 1).contiguous()  # to NHWC for blur helper
                    blurred_bg = gaussian_blur_torch(bg_small, sigma=16.0)

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
                    del frames, bg_frames, bg_small, blurred_bg, final_bg, fg_frames, final_fg, out_tensor, out_bytes

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
                except Exception:
                    pass
                process.wait()

            if 'ffmpeg_log' in locals() and ffmpeg_log:
                ffmpeg_log.close()

            if temp_audio.exists():
                temp_audio.unlink()

            # Final memory sweep
            if 'vr' in locals():
                del vr
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


def _get_ai_config() -> Tuple[bool, int, int, str]:
    """Get AI analysis configuration from environment.
    
    Returns:
        Tuple of (enabled, candidate_count, clip_duration, goal)
    """
    enabled = os.getenv("AI_ANALYSIS_ENABLED", "true").lower() in ("true", "1", "yes")
    candidate_count = int(os.getenv("CANDIDATE_CLIP_COUNT", "30"))
    clip_duration = int(os.getenv("CANDIDATE_CLIP_DURATION", "120"))
    goal = os.getenv("SEMANTIC_GOAL", "action")
    return enabled, candidate_count, clip_duration, goal


def rank_scenes_with_ai(
    video_file: Path,
    scenes: List[List],
    audio_times: np.ndarray,
    audio_score: np.ndarray,
    video_times: np.ndarray,
    video_score: np.ndarray,
    candidate_count: int = 30,
    clip_duration: int = 120,
    goal: str = "action",
) -> List[Tuple[List, float, str]]:
    """Rank scenes using AI semantic analysis.
    
    This function:
    1. Pre-filters scenes by heuristic score to get top candidates
    2. Extracts short clips for each candidate
    3. Sends clips to AI for semantic analysis
    4. Returns scenes ranked by combined score
    
    Args:
        video_file: Source video path
        scenes: List of scene time ranges
        audio_times, audio_score: Audio action profile
        video_times, video_score: Video action profile
        candidate_count: Number of top candidates to send to AI
        clip_duration: Duration of each candidate clip (seconds)
        goal: Semantic analysis goal
        
    Returns:
        List of (scene, combined_score, reason) tuples, sorted by score
    """
    # Step 1: Compute heuristic scores and pre-filter
    scene_scores = []
    for scene in scenes:
        score = scene_action_score(
            scene, audio_times, audio_score, video_times, video_score
        )
        scene_scores.append((scene, score))
    
    # Sort by heuristic score and take top candidates
    scene_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Diversity Strategy: 70% Top Action, 30% Random Exploration
    # This allows lower-action-score clips (like funny dialogue) to have a chance.
    top_tier_count = int(candidate_count * 0.7)
    random_tier_count = candidate_count - top_tier_count

    top_candidates = scene_scores[:top_tier_count]
    remaining_pool = scene_scores[top_tier_count:]
    
    if remaining_pool and random_tier_count > 0:
        # Pick random scenes from the rest
        random_candidates = random.sample(remaining_pool, min(len(remaining_pool), random_tier_count))
        top_candidates.extend(random_candidates)
        logging.info(f"Scene candidates: {len(top_candidates) - len(random_candidates)} high-action + {len(random_candidates)} random exploration.")
    else:
        # Fallback if not enough scenes
        top_candidates = scene_scores[:candidate_count]
    
    if not top_candidates:
        return []
    
    # Normalize heuristic scores to 0-1 range
    max_score = max(s[1] for s in top_candidates) if top_candidates else 1.0
    min_score = min(s[1] for s in top_candidates) if top_candidates else 0.0
    score_range = max_score - min_score if max_score > min_score else 1.0
    
    # Step 2: Extract candidate clips to temp directory
    logging.info(f"Extracting {len(top_candidates)} candidate clips for AI analysis...")
    
    clip_infos: List[ClipScore] = []
    temp_dir = Path(tempfile.mkdtemp(prefix="shorts_ai_"))
    
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Helper function for parallel execution
        def _process_candidate(idx, scene_tuple):
            scene, heuristic_score = scene_tuple
            start_sec = scene[0].get_seconds()
            end_sec = scene[1].get_seconds()
            scene_duration = end_sec - start_sec
            
            # Use shorter clip duration if scene is shorter
            actual_duration = min(clip_duration, scene_duration)
            
            clip_path = temp_dir / f"clip_{idx:03d}.mp4"
            
            success = extract_candidate_clip(
                video_file,
                start_sec,
                actual_duration,
                clip_path
            )
            
            if success:
                # Normalize heuristic score to 0-1
                normalized_score = (heuristic_score - min_score) / score_range
                return ClipScore(
                    clip_path=clip_path,
                    original_start=start_sec,
                    original_end=end_sec,
                    heuristic_score=normalized_score,
                )
            return None

        # Run extraction in parallel (limit threads to avoid disk I/O thrashing)
        max_workers = min(8, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_candidate, i, candidate): i 
                for i, candidate in enumerate(top_candidates)
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                result = future.result()
                if result:
                    clip_infos.append(result)
                else:
                    i = futures[future]
                    logging.warning(f"Failed to extract clip {i}")
        
        # Sort back to original order (optional, but good for consistency)
        clip_infos.sort(key=lambda x: x.clip_path.name)
        
        if not clip_infos:
            logging.warning("No clips extracted for AI analysis. Using heuristic ranking.")
            return [(s, score, "Heuristic only") for s, score in scene_scores]
        
        # Step 3: Run AI analysis
        logging.info(f"Running AI semantic analysis ({goal})...")
        analyzer = get_analyzer()
        result = analyzer.analyze_clips(clip_infos, goal=goal)
        
        logging.info(f"AI analysis complete (provider: {result.provider})")
        
        # Step 4: Map AI scores back to scenes with categories
        ranked_scenes = []
        
        # Create a mapping from clip start time to scene
        start_to_scene = {scene[0].get_seconds(): scene for scene, _ in top_candidates}
        
        for clip_score in result.clips:
            scene = start_to_scene.get(clip_score.original_start)
            if scene:
                # Return tuple: (scene, combined_score, detected_category)
                ranked_scenes.append((
                    scene,
                    clip_score.combined_score,
                    clip_score.detected_category or "action"
                ))
        
        # Sort by combined score
        ranked_scenes.sort(key=lambda x: x[1], reverse=True)
        
        # Log AI rankings with category
        logging.info("AI-ranked scenes:")
        for i, (scene, score, category) in enumerate(ranked_scenes[:10], 1):
            logging.info(
                f"    #{i}: Score {score:.3f} [{category}] | {scene[0].get_timecode()}-{scene[1].get_timecode()}"
            )
        
        return ranked_scenes
        
    finally:
        # Cleanup temp clips
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


def process_video(video_file: Path, config: ProcessingConfig, output_dir: Path) -> None:
    """Process a single video file and generate short clips."""

    logging.info("\nProcess: %s", video_file.name)

    # ==========================================================================
    # DEBUG MODE: Check if we should skip analysis and use cached checkpoint
    # ==========================================================================
    if _is_debug_skip_analysis():
        logging.info("[DEBUG] DEBUG_SKIP_ANALYSIS=1 - Loading from checkpoint...")
        checkpoint = _load_debug_checkpoint(video_file)
        
        if checkpoint:
            # Restore from checkpoint
            scene_list = checkpoint["scenes"]
            video_times = checkpoint["video_times"]
            video_score = checkpoint["video_scores"]
            audio_times = checkpoint["audio_times"]
            audio_score = checkpoint["audio_score"]
            processed_scene_list = checkpoint["processed_scenes"]
            ai_ranked = checkpoint.get("ai_ranked")
            scene_categories = checkpoint.get("scene_categories", {})
            
            logging.info(f"[DEBUG] Loaded {len(scene_list)} scenes, {len(processed_scene_list)} processed scenes")
            
            # Jump directly to rendering or subtitle stage
            if _is_debug_skip_render():
                # Use pre-rendered clips from checkpoint or env var
                rendered_clips = checkpoint.get("rendered_clips")
                if not rendered_clips:
                    rendered_clips = _get_debug_rendered_clips()
                
                if rendered_clips:
                    logging.info(f"[DEBUG] DEBUG_SKIP_RENDER=1 - Using {len(rendered_clips)} cached clips")
                    # Go directly to subtitle generation
                    if is_subtitles_enabled() and rendered_clips:
                        logging.info(f"[DEBUG] Generating subtitles for {len(rendered_clips)} clips...")
                        
                        for clip_path, detected_category in rendered_clips:
                            try:
                                subtitled_path = clip_path.with_stem(clip_path.stem + "_sub")
                                result = generate_subtitles(clip_path, subtitled_path, detected_category)
                                
                                if result and result.exists():
                                    # Replace original with subtitled version
                                    clip_path.unlink()
                                    result.rename(clip_path)
                                    logging.info(f"Subtitles added: {clip_path.name} ({detected_category} style)")
                                else:
                                    logging.warning(f"Subtitle generation failed for: {clip_path.name}")
                            except Exception as e:
                                logging.error(f"Error adding subtitles to {clip_path.name}: {e}")
                    
                    logging.info(f"[DEBUG] Processing complete: {len(rendered_clips)} clips")
                    return
                else:
                    logging.warning("[DEBUG] No rendered clips found. Will render from checkpoint data.")
            
            # Fall through to rendering with checkpoint data
            # Build analysis_result for compatibility
            analysis_result = VideoAnalysisResult(
                scenes=scene_list,
                video_times=video_times,
                video_scores=video_score
            )
            
            # Skip to final scene selection
            goto_rendering = True
        else:
            logging.warning("[DEBUG] No checkpoint found. Running full analysis...")
            goto_rendering = False
    else:
        goto_rendering = False
    
    # ==========================================================================
    # NORMAL MODE: Full analysis pipeline
    # ==========================================================================
    
    if not goto_rendering:
        logging.info("\nProcess: %s", video_file.name)

        # Unified Video Analysis (Single Pass)
        logging.info("Analyzing video content (Scenes & Action) [GPU]...")
        analysis_result = analyze_video_content(video_file)
        
        scene_list = analysis_result.scenes
        video_times = analysis_result.video_times
        video_score = analysis_result.video_scores

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

        # Note: process_video now has video_times/video_score from the unified pass
        
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

        logging.info("Sorted scenes list (by heuristic action score):")
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

        # --- AI-ASSISTED RANKING ---
        ai_enabled, candidate_count, clip_duration, goal = _get_ai_config()
        
        # Track detected categories for each scene (for subtitle style matching)
        scene_categories: dict = {}  # scene_start -> detected_category
        ai_ranked = None
        
        if ai_enabled and len(processed_scene_list) > 0:
            logging.info("Running AI-assisted scene ranking...")
            ai_ranked = rank_scenes_with_ai(
                video_file,
                processed_scene_list,
                audio_times,
                audio_score,
                video_times,
                video_score,
                candidate_count=candidate_count,
                clip_duration=clip_duration,
                goal=goal,
            )
    else:
        # Using checkpoint data - need to reconstruct some variables
        sorted_processed_scene_list = sorted(
            processed_scene_list,
            key=lambda s: scene_action_score(s, audio_times, audio_score, video_times, video_score),
            reverse=True,
        )
        ai_enabled, candidate_count, clip_duration, goal = _get_ai_config()

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

    # --- FINAL SCENE SELECTION ---
    # Extract scenes and their categories from AI ranking
    if ai_ranked:
        final_scene_list = []
        selected_indices = set()
        limit = config.scene_limit
        
        # Strategy: Ensure diversity by picking at least one of each main category if available
        # 1. Pick best Highlight
        for i, (scene, score, category) in enumerate(ai_ranked):
            if category == "highlight" and i not in selected_indices:
                final_scene_list.append(scene)
                scene_categories[scene[0].get_seconds()] = category
                selected_indices.add(i)
                break
        
        # 2. Pick best Action (if we have budget)
        if len(final_scene_list) < limit:
            for i, (scene, score, category) in enumerate(ai_ranked):
                if category == "action" and i not in selected_indices:
                    final_scene_list.append(scene)
                    scene_categories[scene[0].get_seconds()] = category
                    selected_indices.add(i)
                    break

        # 3. Pick best Funny (if we have budget)
        if len(final_scene_list) < limit:
            for i, (scene, score, category) in enumerate(ai_ranked):
                if category == "funny" and i not in selected_indices:
                    final_scene_list.append(scene)
                    scene_categories[scene[0].get_seconds()] = category
                    selected_indices.add(i)
                    break

        # 4. Fill the rest with top scorers
        for i, (scene, score, category) in enumerate(ai_ranked):
            if len(final_scene_list) >= limit:
                break
            if i not in selected_indices:
                final_scene_list.append(scene)
                scene_categories[scene[0].get_seconds()] = category
                selected_indices.add(i)

        logging.info(f"Using AI-ranked top {len(final_scene_list)} scenes (Balanced Selection with Funny)")
    else:
        if ai_enabled:
            logging.warning("AI ranking returned no results. Using heuristic ranking.")
        else:
            logging.info("AI analysis disabled or no scenes. Using heuristic ranking.")
        final_scene_list = sorted_processed_scene_list[:config.scene_limit]

    # Track rendered clips with their detected categories for subtitle processing
    rendered_clips: List[Tuple[Path, str]] = []  # (clip_path, detected_category)

    if final_scene_list:
        for i, scene in enumerate(final_scene_list):
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
            
            # Track for subtitle processing with detected category
            if render_path.exists():
                scene_key = scene[0].get_seconds()
                category = scene_categories.get(scene_key, "action")
                rendered_clips.append((render_path, category))
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

        render_path = output_dir / video_file.name
        render_video_gpu_isolated(
            params,
            render_path,
            max_error_depth=config.max_error_depth,
        )
        
        if render_path.exists():
            rendered_clips.append((render_path, "action"))

    # ==========================================================================
    # DEBUG: Save checkpoint before subtitle generation
    # This allows rerunning just the subtitle/TTS part for debugging
    # ==========================================================================
    if rendered_clips and not _is_debug_skip_analysis():
        try:
            _save_debug_checkpoint(
                video_file=video_file,
                analysis_result=analysis_result,
                audio_times=audio_times,
                audio_score=audio_score,
                processed_scenes=processed_scene_list,
                ai_ranked=ai_ranked,
                scene_categories=scene_categories,
                rendered_clips=rendered_clips,
            )
        except Exception as e:
            logging.warning(f"[DEBUG] Failed to save checkpoint: {e}")
    # ==========================================================================

    # --- SUBTITLE GENERATION (NEW) ---
    if is_subtitles_enabled() and rendered_clips:
        logging.info(f"Generating subtitles for {len(rendered_clips)} clips...")
        
        for clip_path, detected_category in rendered_clips:
            try:
                subtitled_path = clip_path.with_stem(clip_path.stem + "_sub")
                result = generate_subtitles(clip_path, subtitled_path, detected_category)
                
                if result and result.exists():
                    # Replace original with subtitled version
                    clip_path.unlink()
                    result.rename(clip_path)
                    logging.info(f"Subtitles added: {clip_path.name} ({detected_category} style)")
                else:
                    logging.warning(f"Subtitle generation failed for: {clip_path.name}")
            except Exception as e:
                logging.error(f"Error adding subtitles to {clip_path.name}: {e}")
    
    logging.info(f"Processing complete: {len(rendered_clips)} clips generated")



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
    # args = parse_args()
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
