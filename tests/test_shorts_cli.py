"""CPU-only tests for the CLI path-selection helpers."""
from __future__ import annotations

import logging
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Mock heavy dependencies before importing shorts.
for _mod in (
    "torch",
    "torchaudio",
    "cupy",
    "cupyx",
    "cupyx.scipy",
    "cupyx.scipy.ndimage",
):
    sys.modules.setdefault(_mod, MagicMock())

_decord = MagicMock()
_decord.cpu = lambda x: f"cpu({x})"
_decord.gpu = lambda x: f"gpu({x})"
sys.modules.setdefault("decord", _decord)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import shorts  # noqa: E402


def test_parse_args_defaults():
    args = shorts.parse_args([])

    assert args.input == Path("gameplay")
    assert args.output == Path("generated")


def test_parse_args_explicit_paths():
    args = shorts.parse_args(["--input", "footage", "--output", "clips"])

    assert args.input == Path("footage")
    assert args.output == Path("clips")


def test_discover_videos_accepts_single_file(tmp_path):
    video = tmp_path / "clip.MP4"
    video.write_bytes(b"")

    assert shorts.discover_videos(video) == [video]


def test_discover_videos_lists_supported_files_in_order(tmp_path):
    for name in ("b.mp4", "a.mkv", "c.mov", "ignore.txt"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "nested").mkdir()

    assert [path.name for path in shorts.discover_videos(tmp_path)] == [
        "a.mkv",
        "b.mp4",
        "c.mov",
    ]


def test_discover_videos_rejects_unsupported_file(tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("not a video", encoding="utf-8")

    assert shorts.discover_videos(notes) == []


def test_discover_videos_returns_empty_for_missing_path(tmp_path):
    assert shorts.discover_videos(tmp_path / "missing") == []


def test_main_uses_default_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gameplay_dir = tmp_path / "gameplay"
    gameplay_dir.mkdir()
    video = gameplay_dir / "clip.mp4"
    video.write_bytes(b"")
    calls = []

    monkeypatch.setattr(
        shorts,
        "parse_args",
        lambda: Namespace(input=Path("gameplay"), output=Path("generated")),
    )
    monkeypatch.setattr(shorts, "config_from_env", lambda: object())
    monkeypatch.setattr(
        shorts,
        "process_video",
        lambda video_file, config, out: calls.append((video_file, out)),
    )

    shorts.main()

    output_dir = tmp_path / "generated"
    assert output_dir.is_dir()
    assert calls == [(Path("gameplay") / "clip.mp4", Path("generated"))]


def test_main_uses_explicit_input_and_output(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    video = input_dir / "clip.mp4"
    video.write_bytes(b"")
    output_dir = tmp_path / "output"
    calls = []

    monkeypatch.setattr(
        shorts,
        "parse_args",
        lambda: Namespace(input=input_dir, output=output_dir),
    )
    monkeypatch.setattr(shorts, "config_from_env", lambda: object())
    monkeypatch.setattr(
        shorts,
        "process_video",
        lambda video_file, config, out: calls.append((video_file, out)),
    )

    shorts.main()

    assert output_dir.is_dir()
    assert calls == [(video, output_dir)]


def test_main_warns_for_missing_input(tmp_path, monkeypatch, caplog):
    missing = tmp_path / "missing"
    output_dir = tmp_path / "output"

    monkeypatch.setattr(
        shorts,
        "parse_args",
        lambda: Namespace(input=missing, output=output_dir),
    )
    monkeypatch.setattr(shorts, "config_from_env", lambda: object())
    monkeypatch.setattr(
        shorts,
        "process_video",
        lambda *args: pytest.fail("process_video should not be called"),
    )

    with caplog.at_level(logging.WARNING):
        shorts.main()

    assert "does not exist" in caplog.text
    assert not output_dir.exists()


def test_main_warns_for_unsupported_input(tmp_path, monkeypatch, caplog):
    notes = tmp_path / "notes.txt"
    notes.write_text("not a video", encoding="utf-8")
    output_dir = tmp_path / "output"

    monkeypatch.setattr(
        shorts,
        "parse_args",
        lambda: Namespace(input=notes, output=output_dir),
    )
    monkeypatch.setattr(shorts, "config_from_env", lambda: object())
    monkeypatch.setattr(
        shorts,
        "process_video",
        lambda *args: pytest.fail("process_video should not be called"),
    )

    with caplog.at_level(logging.WARNING):
        shorts.main()

    assert "No supported video files" in caplog.text
    assert not output_dir.exists()


def test_main_rejects_output_file(tmp_path, monkeypatch, caplog):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "clip.mp4").write_bytes(b"")
    output_file = tmp_path / "output"
    output_file.write_text("not a directory", encoding="utf-8")

    monkeypatch.setattr(
        shorts,
        "parse_args",
        lambda: Namespace(input=input_dir, output=output_file),
    )
    monkeypatch.setattr(shorts, "config_from_env", lambda: object())
    monkeypatch.setattr(
        shorts,
        "process_video",
        lambda *args: pytest.fail("process_video should not be called"),
    )

    with caplog.at_level(logging.ERROR):
        shorts.main()

    assert "is not a directory" in caplog.text


def test_render_video_ffmpeg_codec_selection(tmp_path, monkeypatch):
    import subprocess
    captured_commands = []

    def fake_popen(cmd, *args, **kwargs):
        captured_commands.append(cmd)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.close = MagicMock()
        mock_proc.wait.return_value = 0
        return mock_proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: MagicMock(returncode=0))

    mock_vr = MagicMock()
    mock_vr.get_avg_fps.return_value = 30.0
    monkeypatch.setattr(shorts, "VideoReader", lambda *args, **kwargs: mock_vr)

    mock_torch = MagicMock()
    monkeypatch.setattr(shorts, "torch", mock_torch)

    params = shorts.RenderParams(
        source_path=tmp_path / "src.mp4",
        start_time=0.0,
        duration=1.0,
        output_width=1080,
        output_height=1920,
        crop_x=0,
        crop_y=0,
        crop_w=1080,
        crop_h=1920,
        bg_width=1080,
        bg_height=1920,
        is_vertical_bg=True,
    )

    output_path = tmp_path / "out.mp4"

    # 1. When CUDA is not available -> libx264
    mock_torch.cuda.is_available.return_value = False
    try:
        shorts.render_video_gpu(params, output_path)
    except Exception:
        pass

    assert len(captured_commands) >= 1
    cmd = captured_commands[0]
    assert "-c:v" in cmd
    idx = cmd.index("-c:v")
    assert cmd[idx + 1] == "libx264"
    assert "hevc_nvenc" not in cmd

    # 2. When CUDA is available -> hevc_nvenc
    captured_commands.clear()
    mock_torch.cuda.is_available.return_value = True
    try:
        shorts.render_video_gpu(params, output_path)
    except Exception:
        pass

    assert len(captured_commands) >= 1
    cmd = captured_commands[0]
    assert "-c:v" in cmd
    idx = cmd.index("-c:v")
    assert cmd[idx + 1] == "hevc_nvenc"
    assert "libx264" not in cmd
