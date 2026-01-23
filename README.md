# AutoShorts

> Automatically generate viral-ready vertical short clips from long-form gameplay footage using AI-powered scene analysis, GPU-accelerated rendering, and optional AI voiceovers.

AutoShorts analyzes your gameplay videos to identify the most engaging moments—action sequences, funny fails, or highlight achievements—then automatically crops, renders, and adds subtitles or AI voiceovers to create ready-to-upload short-form content.

![Python](https://img.shields.io/badge/python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.x-green)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎬 Example Output

Here are some shorts automatically generated from gameplay footage:

| Sample 1 | Sample 2 | Sample 3 | Sample 4 |
| :---: | :---: | :---: | :---: |
| ![Sample 1](generated/showcase/indianajones_pt1_scene-0.gif) | ![Sample 2](generated/showcase/indianajones_pt1_scene-1.gif) | ![Sample 3](generated/showcase/indianajones_pt1_scene-2.gif) | ![Sample 4](generated/showcase/indianajones_pt1_scene-3.gif) |

---

## ✨ Features

### 🎯 AI-Powered Scene Analysis

- **Multi-Provider Support**: Choose between **OpenAI** (GPT-5-mini, GPT-4o) or **Google Gemini** for scene analysis
- **Semantic Analysis Modes**:
  - `action` — Focus on intense combat/action moments
  - `funny` — Detect fail compilations and humorous moments  
  - `highlight` — Find memorable achievements and clutch plays
  - `mixed` — Auto-detect the best category for each clip (recommended)

### 🎙️ Subtitle Generation

- **Speech Mode**: Uses OpenAI Whisper to transcribe voice/commentary
- **AI Captions Mode**: AI-generated contextual captions for gameplay without voice
- **Caption Styles**: `gaming`, `dramatic`, `funny`, `minimal`, or `auto`
- **PyCaps Integration**: Multiple visual templates including `hype`, `retro-gaming`, `neo-minimal`
- **AI Enhancement**: Semantic tagging and emoji suggestions (e.g., "HEADSHOT! 💀🔥")

### 🔊 AI Voiceover (ChatterBox TTS)

- **Local TTS Generation**: No cloud API needed for voice synthesis
- **Emotion Control**: Adjustable emotion/exaggeration levels for English
- **Multilingual Support**: 20+ languages including Japanese, Korean, Chinese, Spanish, French, and more
- **Voice Cloning**: Optional reference audio for custom voice styles
- **Smart Mixing**: Automatic ducking of game audio when voiceover plays

### ⚡ GPU-Accelerated Pipeline

- **Scene Detection**: Custom implementation using `decord` + PyTorch on GPU
- **Audio Analysis**: `torchaudio` on GPU for fast RMS and spectral flux calculation
- **Video Analysis**: GPU streaming via `decord` for stable motion estimation
- **Image Processing**: `cupy` (CUDA-accelerated NumPy) for blur and transforms
- **Rendering**: PyTorch + **NVENC** hardware encoder for ultra-fast rendering

### 📐 Smart Video Processing

- Scenes ranked by combined action score (audio 0.6 + video 0.4 weights)
- Configurable aspect ratio (default 9:16 for TikTok/Shorts/Reels)
- Smart cropping with optional blurred background for non-vertical footage
- Retry logic during rendering to avoid spurious failures

### 🛡️ Robust Fallback System

AutoShorts is designed to work even when optimal components fail:

| Component | Primary | Fallback |
|-----------|---------|----------|
| **Video Encoding** | NVENC (GPU) | libx264 (CPU) |
| **Subtitle Rendering** | PyCaps (styled) | FFmpeg burn-in (basic) |
| **AI Analysis** | OpenAI/Gemini API | Heuristic scoring (local) |
| **TTS Device** | CUDA (GPU) | CPU inference |

---

## 📋 Requirements

### Hardware

- **NVIDIA GPU** with CUDA support (RTX series recommended for NVENC + TTS)
- **NVIDIA Drivers** compatible with CUDA 12.x

### Software

- Python 3.10
- FFmpeg 4.4.2 (for Decord compatibility)
- CUDA Toolkit with `nvcc` (for building Decord from source)
- System libraries: `libgl1`, `libglib2.0-0`

---

## 🚀 Installation

### Option 1: Makefile Installation (Recommended)

The Makefile handles everything automatically—environment creation, dependency installation, and building Decord with CUDA support.

```bash
git clone https://github.com/divyaprakash0426/autoshorts.git
cd autoshorts

# Run the installer (uses conda/micromamba automatically)
make install

# Activate the environment
overlay use .venv/bin/activate.nu    # For Nushell
# OR
source .venv/bin/activate            # For Bash/Zsh
```

The Makefile will:

1. Download micromamba if conda/mamba is not found
2. Create a Python 3.10 environment with FFmpeg 4.4.2
3. Install NV Codec Headers for NVENC support
4. Build Decord from source with CUDA enabled
5. Install all pip requirements

### Option 2: Docker (GPU Required)

**Prerequisite**: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) must be installed.

```bash
# Build the image
docker build -t autoshorts .

# Run with GPU access
docker run --rm \
    --gpus all \
    -v $(pwd)/gameplay:/app/gameplay \
    -v $(pwd)/generated:/app/generated \
    --env-file .env \
    autoshorts
```

> **Note**: The `--gpus all` flag is essential for NVENC and CUDA acceleration.

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

### Key Configuration Options

| Category | Variable | Description |
|----------|----------|-------------|
| **AI Provider** | `AI_PROVIDER` | `openai`, `gemini`, or `local` |
| | `AI_ANALYSIS_ENABLED` | Enable/disable AI scene analysis |
| | `OPENAI_MODEL` | Model for analysis (e.g., `gpt-5-mini`) |
| | `AI_SCORE_WEIGHT` | How much to weight AI vs heuristic (0.0-1.0) |
| **Semantic Analysis** | `SEMANTIC_GOAL` | `action`, `funny`, `highlight`, or `mixed` |
| | `CANDIDATE_CLIP_COUNT` | Number of clips to analyze |
| **Subtitles** | `ENABLE_SUBTITLES` | Enable subtitle generation |
| | `SUBTITLE_MODE` | `speech` (Whisper), `ai_captions`, or `none` |
| | `CAPTION_STYLE` | `gaming`, `dramatic`, `funny`, `minimal`, `auto` |
| | `PYCAPS_TEMPLATE` | Visual template for captions |
| **TTS Voiceover** | `ENABLE_TTS` | Enable ChatterBox voiceover |
| | `TTS_LANGUAGE` | Language code (e.g., `en`, `ja`, `es`) |
| | `TTS_EMOTION_LEVEL` | Emotion intensity or `auto` |
| **Video Output** | `TARGET_RATIO_W/H` | Aspect ratio (default 9:16) |
| | `SCENE_LIMIT` | Max clips per source video |
| | `MIN/MAX_SHORT_LENGTH` | Clip duration bounds (seconds) |

See `.env.example` for the complete list with detailed descriptions.

---

## 📖 Usage

1. **Place source videos** in the `gameplay/` directory
2. **Run the script**:

   ```bash
   python run.py
   ```

3. **Generated clips** are saved to `generated/`

### Output Structure

```
generated/
├── video_name scene-0.mp4          # Rendered short clip
├── video_name scene-0_sub.json     # Subtitle data
├── video_name scene-0.ffmpeg.log   # Render log
├── video_name scene-1.mp4
└── ...
```

### Example Output

After processing, you'll find clips like:

| File | Description |
|------|-------------|
| `scene-0.mp4` | Highest-ranked action scene (24MB, ~60s) |
| `scene-1.mp4` | Second-ranked scene (10MB, ~30s) |
| `scene-2.mp4` | Third-ranked scene with AI captions |
| `scene-3.mp4` | Fourth-ranked scene with TTS voiceover |

---

## 🧪 Development

### Linting

```bash
pip install ruff
ruff check .
```

### Running Tests

```bash
pytest -q
```

> Tests mock GPU availability and can run in standard CI environments.

### Debug Variables

For faster iteration during development, you can skip expensive steps using these environment variables in your `.env`:

| Variable | Description |
|----------|-------------|
| `DEBUG_SKIP_ANALYSIS=1` | Skip AI scene analysis (uses cached/heuristic scores) |
| `DEBUG_SKIP_RENDER=1` | Skip video rendering (useful for testing analysis only) |
| `DEBUG_RENDERED_CLIPS="path1:category,path2"` | Test with specific pre-rendered clips |

Example workflow for testing subtitles only:

```bash
# In .env
DEBUG_SKIP_ANALYSIS=1
DEBUG_SKIP_RENDER=1
DEBUG_RENDERED_CLIPS="generated/test_clip.mp4:action"
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| **"CUDA not available"** | Ensure `--gpus all` (Docker) or CUDA toolkit is installed |
| **NVENC Error** | Falls back to `libx264` automatically; check GPU driver |
| **PyCaps fails** | Falls back to FFmpeg burn-in subtitles automatically |
| **Decord EOF hang** | Increase `DECORD_EOF_RETRY_MAX` or set `DECORD_SKIP_TAIL_FRAMES=300` |
| **API rate limits** | Switch to `gpt-5-mini` (10M free tokens/day) or use `local` provider |

---

## 🙏 Acknowledgments

This project builds upon the excellent work of:

- **[artryazanov/shorts-maker-gpu](https://github.com/artryazanov/shorts-maker-gpu)** — Original GPU-optimized shorts maker
- **[Binary-Bytes/Auto-YouTube-Shorts-Maker](https://github.com/Binary-Bytes/Auto-YouTube-Shorts-Maker)** — Original concept and inspiration

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
