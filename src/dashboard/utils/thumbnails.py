from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from moviepy import VideoFileClip


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    duration: float
    size_mb: float
    resolution: str
    thumbnail: Optional[Path]


THUMB_DIR = Path("generated/.thumbnails")


def _ensure_thumb_dir() -> None:
    THUMB_DIR.mkdir(parents=True, exist_ok=True)


def build_thumbnail(video_path: Path) -> Optional[Path]:
    _ensure_thumb_dir()
    thumb_path = THUMB_DIR / f"{video_path.stem}.png"
    if thumb_path.exists():
        return thumb_path
    try:
        with VideoFileClip(str(video_path)) as clip:
            frame_time = min(1.0, clip.duration / 2.0) if clip.duration else 0.0
            frame = clip.get_frame(frame_time)
            clip.save_frame(str(thumb_path), t=frame_time)
        return thumb_path
    except Exception:
        return None


def get_video_info(video_path: Path) -> VideoInfo:
    try:
        with VideoFileClip(str(video_path)) as clip:
            duration = clip.duration or 0.0
            size_mb = video_path.stat().st_size / (1024 * 1024)
            resolution = f"{clip.w}x{clip.h}"
    except Exception:
        duration = 0.0
        size_mb = video_path.stat().st_size / (1024 * 1024)
        resolution = "unknown"
    thumbnail = build_thumbnail(video_path)
    return VideoInfo(
        path=video_path,
        duration=duration,
        size_mb=size_mb,
        resolution=resolution,
        thumbnail=thumbnail,
    )


def list_videos(folder: Path) -> List[VideoInfo]:
    if not folder.exists():
        return []
    video_paths = []
    for path in folder.iterdir():
        # Check if it's a regular file or a symlink pointing to a file
        if (path.is_file() or path.is_symlink()) and path.suffix.lower() in {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}:
            # Verify symlink points to existing file
            if path.is_symlink():
                try:
                    if not path.resolve().exists():
                        continue  # Skip broken symlinks
                except OSError:
                    continue  # Skip inaccessible symlinks
            video_paths.append(path)
    return [get_video_info(path) for path in sorted(video_paths)]
