# Copilot instructions for AutoShorts

Purpose: provide concise guidance for future Copilot CLI / Code Assistant sessions working on this repository.

## 1) Build, test, and lint commands

- Quick install (preferred):
  - `./install.sh`
  - or: `make install`
- Virtualenv created at `.venv/`. Activate with `source .venv/bin/activate` (Bash/Zsh).
- Run single test file with pytest:
  - `pytest tests/test_some_specific.py`
  - or run a single test function: `pytest tests/test_file.py::test_function_name`
- Full test suite:
  - `pytest tests/`
- Verify GPU PyTorch install:
  - `python test_pytorch.py`
- Useful Make targets:
  - `make install_torch`
  - `make install_pip_requirements`
  - `make install_decord`
  - `make download_models`
  - `make verify`

## 2) Project structure

The codebase uses a flattened layout under `src/` (the older `modules/` split was reverted; functionality is colocated in top-level modules):

```
src/
├── shorts.py                # Main pipeline orchestration (analysis, selection, rendering)
├── ai_providers.py          # AI provider abstraction (Gemini/OpenAI/Local), ClipScore, classification prompts
├── subtitle_generator.py    # Whisper transcription, SRT generation and PyCaps subtitle application
├── tts_generator.py         # Qwen3-TTS wrapper, voice presets, generate_voice_description()
├── story_narrator.py        # Cross-clip story generation (narrative text)
├── dashboard/               # Streamlit UI pages and utilities (optional)
├── __init__.py              # package marker
```

### Key Entry Points

| File | Purpose |
|------|---------|
| `run.py` | Top-level CLI entry point and signal handling |
| `src/shorts.py` | Main pipeline orchestration and helper classes (ProcessingConfig, RenderParams) |
| `src/ai_providers.py` | AI provider logic: ClipScore, SemanticAnalyzer implementations, prompts |
| `src/tts_generator.py` | TTS wrapper and voice description generator (Qwen3-TTS) |
| `src/subtitle_generator.py` | Whisper + PyCaps transcription and subtitle rendering |
| `src/story_narrator.py` | Cross-clip story/narration generation (Gemini/OpenAI integration) |

### Processing Pipeline Flow

1. Analysis - Scene detection, audio scoring and rough clip extraction (implemented in `src/shorts.py` and assisted by `ai_providers.py`)
2. Selection - Filtering and ranking clips (`shorts.py` + `ai_providers.py`)
3. Rendering - GPU compositing and encoding (`shorts.py` uses decord, cupy, torch, ffmpeg)
4. Content - Subtitles and TTS (`subtitle_generator.py` and `tts_generator.py`)

See `docs/flow_diagram.md` for detailed Mermaid diagrams.

## 3) Key conventions and repository-specific patterns

- **Config via environment**: Use `.env` (copy `.env.example`) or export variables. Key settings: `AI_PROVIDER`, `ENABLE_TTS`, `ENABLE_SUBTITLES`, `CAPTION_STYLE`, `TARGET_RATIO_W/H`.

- **AI Providers** (`AI_PROVIDER` env var):
  - `gemini` - Google Gemini API (cloud)
  - `openai` - OpenAI API (cloud)
  - `qwen3vl` - Qwen3-VL-8B local GPU model
  - `local` - Heuristics only, no AI

- **"Pregen" vs per-clip generation**:
  - Local providers (qwen3vl) pre-generate captions/story during analysis
  - Remote providers generate on-demand to avoid API costs

- **Render metadata tracking**:
  - Rendered clips are tuples: `(Path, category, render_meta)`
  - `render_meta` contains: scene, start, length, source file, scene_idx

- **GPU model lifecycle**:
  - Heavy models use singleton pattern: `get_instance()` / `clear_instance()`
  - Call `gc.collect()` + `torch.cuda.empty_cache()` between phases
  - `render_video_gpu_isolated()` uses multiprocessing.spawn for VRAM isolation

- **FFmpeg gotchas**:
  - Use `ffprobe -print_format json` for precise durations
  - amix uses `duration=longest` to preserve TTS audio
  - NVENC encoding via `-c:v h264_nvenc`

## 4) Relevant files for quick edits or debugging

| Component | File |
|-----------|------|
| Pipeline orchestration | `src/shorts.py` |
| AI providers | `src/ai_providers.py` |
| Subtitles / SRT handling | `src/subtitle_generator.py` |
| TTS / voice generation | `src/tts_generator.py` |
| Story / narration | `src/story_narrator.py` |
| Dashboard (Streamlit UI) | `src/dashboard/` |
| Installation | `Makefile`, `install.sh` |

## 5) AI-assistant / Copilot usage tips

- When changing ffmpeg commands, test with a short sample and capture stderr
- For timeline changes (subtitles/TTS), verify: voice duration vs clip duration, SRT alignment
- Call `clear_instance()` on models between tests to avoid VRAM buildup
- Use debug checkpoints (`DEBUG_SKIP_ANALYSIS=1`, `DEBUG_SKIP_RENDER=1`) to speed up iteration

---

## Nushell Environment

**IMPORTANT**: The repository owner uses **Nushell** as their primary shell. Nushell is NOT POSIX-compliant and has different syntax from Bash/Zsh.

### Key Differences from Bash

| Bash | Nushell | Notes |
|------|---------|-------|
| `cmd1 && cmd2` | `cmd1; cmd2` | Nushell has NO && operator for command chaining |
| `cmd1 \|\| cmd2` | Use if blocks | No \|\| for "run on failure" |
| `export FOO=bar` | `$env.FOO = "bar"` | Environment variables |
| `$VAR` | `$env.VAR` | Accessing env vars |
| `cat file` | `open file` | File reading |
| `echo "text" > file` | `"text" \| save file` | File writing |
| `source script.sh` | `source script.nu` | Sourcing files |

### Handling && (Logical AND) in Nushell

Nushell does **NOT** support Bash-style && for command chaining. Alternatives:

```nu
# Sequential execution (always runs both):
cmd1; cmd2

# Run second only if first succeeds:
if (do { cmd1 } | complete).exit_code == 0 { cmd2 }

# Or check LAST_EXIT_CODE after command:
cmd1
if $env.LAST_EXIT_CODE == 0 { cmd2 }
```

### Logical Operators in Conditions

```nu
# In conditions, use 'and' / 'or' keywords:
if $a > 0 and $b < 10 { ... }
if $x == "foo" or $y == "bar" { ... }

# NOT valid:
# if $a > 0 && $b < 10 { ... }  # WRONG - && not supported
```

### Common Nushell Commands

```nu
# List files with details
ls -l | where type == "dir"

# Environment variables
$env.PATH                          # Access
$env.FOO = "bar"                   # Set
with-env { FOO: "bar" } { cmd }    # Temporary

# File operations
open file.json                     # Parse JSON/TOML/YAML automatically
open file.txt --raw                # Raw text
"content" | save file.txt          # Write file

# Process management
ps | where name == "python"

# Running external commands (escape with ^)
^ls -la                            # Use system ls instead of nu's

# Pipelines and filtering
ls | where size > 1mb | select name size
open log.txt | lines | where ($it =~ "error")
```

### Nushell Equivalents Cheat Sheet

```nu
# Bash: cd dir && make install
# Nushell:
cd dir; make install

# Bash: test -f file && echo "exists"
# Nushell:
if ("file" | path exists) { print "exists" }

# Bash: export PATH="$PATH:/new/path"
# Nushell:
$env.PATH = ($env.PATH | append "/new/path")

# Bash: cat file | grep pattern
# Nushell:
open file --raw | lines | where ($it =~ "pattern")

# Bash: for f in *.mp4; do echo $f; done
# Nushell:
ls *.mp4 | each { |f| print $f.name }
```

When providing shell examples for this repository, **always use Nushell syntax**.
