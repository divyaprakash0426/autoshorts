# Shorts Maker (GPU Optimized)

Shorts Maker generates vertical video clips from longer gameplay footage. The script detects scenes, computes audio and video action profiles (sound intensity + visual motion), and combines them to rank scenes by overall intensity. It then crops to the desired aspect ratio and renders ready‑to‑upload shorts.

**This version has been heavily optimized for NVIDIA GPUs using CUDA.**

## Features

- **GPU-Accelerated Processing**:
  - **Scene Detection**: Custom implementation using `decord` and PyTorch on GPU.
  - **Audio Analysis**: Uses `torchaudio` on GPU for fast RMS and spectral flux calculation.
  - **Video Analysis**: Direct frame access on GPU via `decord` for motion estimation.
  - **Image Processing**: `cupy` (CUDA-accelerated NumPy) used for heavy operations like blurring backgrounds.
  - **Rendering**: Hardware encoding via `h264_nvenc` (NVIDIA NVENC).
- Audio + video action scoring:
  - Combined ranking with tunable weights (defaults: audio 0.6, video 0.4).
- Scenes ranked by combined action score rather than duration.
- Smart cropping with optional blurred background for non‑vertical footage.
- Retry logic during rendering to avoid spurious failures.
- Configuration via `.env` environment variables.

## Requirements

- **NVIDIA GPU** with CUDA support.
- **NVIDIA Drivers** (compatible with CUDA 12.1+ recommended).
- Python 3.10+
- FFmpeg (required by `moviepy`).
- System libraries: `libgl1`, `libglib2.0-0` (often needed for vision libraries).

Python dependencies (see `requirements.txt`):
- `torch`, `torchaudio` (with CUDA support)
- `cupy-cuda12x`
- `decord`
- `moviepy`

## Installation

### Manual Setup (Linux with CUDA)

Ensure you have the NVIDIA drivers and CUDA toolkit installed.

```bash
git clone https://github.com/artryazanov/shorts-maker-gpu.git
cd shorts-maker
python3 -m venv venv
source venv/bin/activate

# Install dependencies (ensure pip picks up the CUDA versions for torch/cupy)
pip install -r requirements.txt
```

If you encounter issues with PyTorch or CuPy not finding the GPU, refer to their respective installation guides for your specific CUDA version.

## Usage

1. Place source videos inside the `gameplay/` directory.
2. Run the script:

```bash
python shorts.py
```

3. Generated clips are written to the `generated/` directory.

During processing, the log shows an action score for each combined scene and the final list sorted by that score. The top scenes (by action intensity) are rendered first using NVENC.

## Docker (Recommended)

The easiest way to run this application is using Docker with the NVIDIA Container Toolkit.

**Prerequisite**: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) must be installed on the host.

Build and run:

```bash
docker build -t shorts-maker .

# Run with GPU access
docker run --rm \
    --gpus all \
    -v $(pwd)/gameplay:/app/gameplay \
    -v $(pwd)/generated:/app/generated \
    --env-file .env \
    shorts-maker
```

Note the `--gpus all` flag, which is essential for the application to access hardware acceleration.

## Configuration

Copy `.env.example` to `.env` and adjust values as needed.

Supported variables (defaults shown):
- `TARGET_RATIO_W=1` — Width part of the target aspect ratio (e.g., 9 for 9:16).
- `TARGET_RATIO_H=1` — Height part of the target aspect ratio (e.g., 16 for 9:16).
- `SCENE_LIMIT=6` — Maximum number of top scenes rendered per source video.
- `X_CENTER=0.5` — Horizontal crop center in range [0.0, 1.0].
- `Y_CENTER=0.5` — Vertical crop center in range [0.0, 1.0].
- `MAX_ERROR_DEPTH=3` — Maximum retry depth if rendering fails.
- `MIN_SHORT_LENGTH=15` — Minimum short length in seconds.
- `MAX_SHORT_LENGTH=179` — Maximum short length in seconds.
- `MAX_COMBINED_SCENE_LENGTH=300` — Maximum combined length (in seconds).
- `DECORD_EOF_RETRY_MAX=65536` — Decord EOF retry attempts; increase for very long 4K files where the last frames are slow to retrieve.
- `DECORD_SKIP_TAIL_FRAMES=0` — Optionally skip the last N frames when sampling video action profile to avoid problematic EOF reads (leave 0 unless you consistently hit EOF on specific files).

## Running Tests

Unit tests live in the `tests/` folder. Run them with:

```bash
pytest -q
```

Note: The tests are designed to mock GPU availability if it is missing, so they can run in standard CI environments.

## Troubleshooting

- **"Torch not installed" / "CUDA not available"**: Ensure you are running inside the Docker container with `--gpus all` or have the correct CUDA toolkit installed locally.
- **NVENC Error**: If `h264_nvenc` fails, the script attempts to fall back to software encoding (`libx264`). Check if your GPU supports NVENC and if the drivers are up to date.

## Acknowledgments

Thank the Binary-Bytes for the original code and idea: https://github.com/Binary-Bytes/Auto-YouTube-Shorts-Maker

## License

This project is released under the [Unlicense](LICENSE).
