# AutoShorts

Turn long-form videos into short clips with AI-assisted moment selection,
GPU rendering, styled captions, and optional voiceovers.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

## Showcase

Four clips generated from gameplay footage:

| Clip 1 | Clip 2 | Clip 3 | Clip 4 |
| :---: | :---: | :---: | :---: |
| ![Gameplay clip 1](generated/showcase/indianajones_pt1_scene-0.gif) | ![Gameplay clip 2](generated/showcase/indianajones_pt1_scene-1.gif) | ![Gameplay clip 3](generated/showcase/indianajones_pt1_scene-2.gif) | ![Gameplay clip 4](generated/showcase/indianajones_pt1_scene-3.gif) |

## What it does

- **Content-aware clipping:** 15 presets with their own moment categories, analysis
  prompts, caption styles, and voice presets. Gaming/esports use action heuristics
  plus AI ranking; other types use AI-first selection.
- **Flexible analysis:** Gemini, including full-video deep analysis; OpenAI vision;
  or local heuristic selection without API analysis.
- **Captions and narration:** Whisper transcription, AI-written captions, PyCaps
  styling with FFmpeg fallback, and cross-clip story narration.
- **Optional voiceovers:** Local Qwen3-TTS with multilingual voice design,
  adjustable source/voice volumes, and subtitle timing matched to speech.
- **GPU rendering:** CUDA decoding and analysis, cropping, blurred backgrounds,
  configurable aspect ratios, and NVENC encoding.
- **Dashboard:** Choose content and caption styles, manage the video queue,
  monitor jobs, and browse generated clips.

**Supported content:** gaming, esports, podcasts, interviews, news/commentary,
entertainment, TV shows, comedy, music, vlogs, cooking/DIY, fitness, education,
documentaries, and sports.

**Examples:** [English roast](https://www.youtube.com/shorts/tTUipTAdBlk) /
[Japanese narration](https://www.youtube.com/shorts/DYNEr1CzTpY) /
[Chinese news style](https://www.youtube.com/shorts/kGRrpu66fpk).

## Get started

You need an NVIDIA GPU and driver, a CUDA toolkit with `nvcc`, FFmpeg with
NVENC support, and Python 3.10+. For voiceovers, 6 GB+ VRAM and 16 GB system RAM
are recommended.

```nu
git clone https://github.com/divyaprakash0426/autoshorts.git
cd autoshorts
make install
cp .env.example .env
```

The [Makefile](Makefile) creates a single `.venv` with the Python dependencies and
CUDA-enabled Decord. Decord's build requires FFmpeg 4.x libraries; see
[environment.yml](environment.yml). PyTorch 2.10 pairs with TorchCodec 0.10.0.
An interactive [installer](install.sh) and [Docker setup](Dockerfile) are also
provided; Docker requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

Edit `.env` before running: choose a content type and AI provider, add the
corresponding API key, and set `DEBUG_SKIP_ANALYSIS=false` and
`DEBUG_SKIP_RENDER=false` for normal processing.

### Dashboard or CLI

```nu
# Dashboard: configure settings, add videos, and start processing
.venv/bin/python -m streamlit run src/dashboard/About.py

# CLI: read videos from gameplay/ and write clips to generated/
.venv/bin/python run.py

# Or choose a specific source file/directory and output directory
.venv/bin/python run.py --input /path/to/videos --output /path/to/clips
```

Use the dashboard's file picker or place source videos in `gameplay/`.
Output clips are saved in `generated/`. `--input` accepts either a single
supported video file or a directory; `--output` is created when needed.

## Configuration

Settings live in `.env` and can also be changed through the dashboard.

| Setting | Purpose |
| --- | --- |
| `AI_PROVIDER` | `gemini`, `openai`, or `local` |
| `VIDEO_TYPE` | Content preset, such as `gaming`, `podcasts`, or `educational` |
| `ENABLE_SUBTITLES` / `SUBTITLE_MODE` | Toggle captions; choose `speech`, `ai_captions`, or `none` |
| `CAPTION_STYLE` | `auto` matches the detected moment; explicit styles are content-specific |
| `ENABLE_TTS` / `TTS_LANGUAGE` | Toggle voiceovers and select their language |
| `SCENE_LIMIT` | Maximum clips per source video |
| `MIN_SHORT_LENGTH` / `MAX_SHORT_LENGTH` | Clip duration bounds; `0` selects automatic lengths |
| `TARGET_RATIO_W` / `TARGET_RATIO_H` | Output aspect ratio, typically `9` / `16` |

See [.env.example](.env.example) for all settings and
[voice design](docs/TTS_VOICE_DESIGN.md) for advanced TTS options.

**Privacy and cost:** Cloud providers receive video content. Gemini deep analysis
uploads the full source video; API usage may incur charges. `local` uses heuristic
selection instead, but GPU rendering still requires NVIDIA hardware.

## Development

```nu
# Lightweight suite; real GPU/model checks are skipped
.venv/bin/python -m pytest tests/ -q

# Include actual CUDA decoding, rendering, and local model inference
.venv/bin/python -m pytest tests/ -q --run-gpu --run-models
```

See [Contributing](CONTRIBUTING.md) for test prerequisites and bounded footage
checks, [pipeline flow](docs/flow_diagram.md) for architecture, and
[Roadmap](ROADMAP.md) for planned features.

## License and credits

[Apache 2.0](LICENSE); see [NOTICE](NOTICE). Dependencies and model weights retain
their own licenses.

Built on work from [shorts-maker-gpu](https://github.com/artryazanov/shorts-maker-gpu)
and [Auto-YouTube-Shorts-Maker](https://github.com/Binary-Bytes/Auto-YouTube-Shorts-Maker).

[Support the project](https://www.buymeacoffee.com/divyaprakash0426) -- donations go to charity.
