# AutoShorts: Modular Architecture Flow Diagram

This file documents the end-to-end flow for the refactored modular AutoShorts codebase. The system processes videos through distinct phases: Analysis, Selection, Rendering, and Content Generation (Subtitles/TTS).

## Architecture Overview

```
src/
├── core/                    # Shared types, config, utilities
│   ├── config.py           # ProcessingConfig from environment
│   ├── types.py            # RenderParams, VideoAnalysisResult
│   └── utils.py            # Memory logging, helpers
├── modules/
│   ├── ai/                 # AI Provider abstraction
│   │   ├── base.py         # SemanticAnalyzer abstract class
│   │   ├── factory.py      # get_analyzer() factory
│   │   ├── generation.py   # generate_ai_captions()
│   │   └── providers/      # Gemini, OpenAI, Qwen3VL, Local
│   ├── analysis/           # Video/Audio analysis
│   │   ├── visual.py       # Scene detection, action scoring (GPU)
│   │   ├── audio.py        # Audio action profile (GPU)
│   │   └── scoring.py      # Heuristic scoring functions
│   ├── selection/          # Scene filtering & ranking
│   │   ├── filtering.py    # combine_scenes, split_overlong
│   │   └── ranking.py      # AI-assisted ranking, diversity
│   ├── rendering/          # GPU video rendering
│   │   ├── render.py       # render_video_gpu, get_render_params
│   │   └── filters.py      # Blur filters (PyTorch/CuPy)
│   ├── content/            # Subtitles & TTS
│   │   ├── subtitles.py    # generate_subtitles, Whisper, PyCaps
│   │   └── tts.py          # QwenTTS, mix_audio_with_video
│   └── narrative/          # Story generation
│       └── generator.py    # generate_unified_story()
├── pipeline/
│   └── runner.py           # Main orchestration logic
└── dashboard/              # Streamlit UI (optional)
```

## Main Processing Flow

```mermaid
flowchart TD
    subgraph Entry["Entry Point (run.py)"]
        Start([Start]) --> LoadEnv[Load .env config]
        LoadEnv --> ConfigEnv[config_from_env]
        ConfigEnv --> ScanVideos[Scan gameplay/ directory]
    end

    ScanVideos --> Pipeline

    subgraph Pipeline["Pipeline Runner (pipeline/runner.py)"]
        direction TB
        
        subgraph Analysis["Phase 1: Analysis"]
            AnalyzeVis[analyze_video_content\nmodules/analysis/visual.py\n- Scene detection\n- Video action scoring]
            AnalyzeAud[compute_audio_action_profile\nmodules/analysis/audio.py\n- Audio energy analysis]
            AnalyzeVis --> AnalyzeAud
        end

        subgraph Selection["Phase 2: Selection"]
            Filter[combine_scenes + split_overlong\nmodules/selection/filtering.py]
            HeuristicSort[Sort by heuristic score\nmodules/analysis/scoring.py]
            AIRank{AI_ANALYSIS_ENABLED?}
            RankAI[rank_scenes_with_ai\nmodules/selection/ranking.py]
            SelectFinal[Select diverse scenes\nby category + score]
            
            Filter --> HeuristicSort
            HeuristicSort --> AIRank
            AIRank -->|Yes| RankAI
            AIRank -->|No| SelectFinal
            RankAI --> SelectFinal
        end

        subgraph Rendering["Phase 3: Rendering"]
            GetParams[get_render_params\nmodules/rendering/render.py]
            RenderGPU[render_video_gpu_isolated\n- Decord GPU decode\n- CuPy blur\n- FFmpeg NVENC encode]
            GetParams --> RenderGPU
        end

        subgraph Content["Phase 4: Content Generation"]
            PregenCheck{Story Mode?}
            StoryGen[generate_unified_story\nmodules/narrative/generator.py]
            LocalPregen[Batch AI caption pregen\nmodules/ai/generation.py]
            
            SubCheck{ENABLE_SUBTITLES?}
            GenSubs[generate_subtitles\nmodules/content/subtitles.py]
            
            TTSCheck{ENABLE_TTS?}
            GenTTS[QwenTTS + mix_audio\nmodules/content/tts.py]
            
            PregenCheck -->|Yes| StoryGen
            PregenCheck -->|No + Local AI| LocalPregen
            StoryGen --> SubCheck
            LocalPregen --> SubCheck
            PregenCheck -->|No + Remote AI| SubCheck
            
            SubCheck -->|Yes| GenSubs
            GenSubs --> TTSCheck
            SubCheck -->|No| TTSCheck
            TTSCheck -->|Yes| GenTTS
        end

        Analysis --> Selection
        Selection --> Rendering
        Rendering --> Content
    end

    Content --> Final([Final Clips in generated/])

    classDef phase fill:#e1f5fe,stroke:#01579b
    classDef decision fill:#fff3e0,stroke:#e65100
    classDef module fill:#f3e5f5,stroke:#7b1fa2
    
    class Analysis,Selection,Rendering,Content phase
    class AIRank,PregenCheck,SubCheck,TTSCheck decision
```

## AI Provider Flow

```mermaid
flowchart LR
    subgraph Factory["modules/ai/factory.py"]
        GetAnalyzer[get_analyzer]
    end

    GetAnalyzer --> ProviderCheck{AI_PROVIDER env}
    
    ProviderCheck -->|gemini| Gemini[GeminiAnalyzer\nproviders/gemini.py]
    ProviderCheck -->|openai| OpenAI[OpenAIAnalyzer\nproviders/openai.py]
    ProviderCheck -->|qwen3vl| Qwen[Qwen3VLAnalyzer\nproviders/qwen.py\nLocal GPU model]
    ProviderCheck -->|local| Local[LocalAnalyzer\nproviders/local.py\nHeuristics only]

    Gemini --> Methods
    OpenAI --> Methods
    Qwen --> Methods
    Local --> Methods

    subgraph Methods["SemanticAnalyzer Interface"]
        Analyze[analyze_clip]
        Caption[generate_captions]
        Cleanup[cleanup]
    end

    classDef local fill:#c8e6c9,stroke:#2e7d32
    classDef remote fill:#bbdefb,stroke:#1565c0
    class Qwen,Local local
    class Gemini,OpenAI remote
```

## Subtitle & TTS Flow

```mermaid
flowchart TD
    ClipIn([Rendered Clip]) --> ModeCheck{SUBTITLE_MODE}
    
    ModeCheck -->|speech| Whisper[transcribe_audio\nWhisper model]
    ModeCheck -->|ai_captions| AICheck{Pregen available?}
    ModeCheck -->|story_*| StoryMode[TTS-first probing\nBuild SRT from TTS durations]
    
    AICheck -->|Yes| UsePregen[Use pregenerated captions]
    AICheck -->|No| GenOnDemand[generate_ai_captions\nOn-demand API call]
    
    Whisper --> ToSRT[Convert to SRT]
    UsePregen --> ToSRT
    GenOnDemand --> ToSRT
    StoryMode --> ToSRT
    
    ToSRT --> BurnSubs[apply_pycaps_subtitles\nBurn captions to video]
    
    BurnSubs --> TTSEnabled{ENABLE_TTS?}
    TTSEnabled -->|Yes| ParseSRT[Parse SRT timing]
    ParseSRT --> QwenTTS[QwenTTS.generate_for_captions]
    QwenTTS --> MixAudio[mix_audio_with_video\nFFmpeg amix]
    
    TTSEnabled -->|No| SubOnly([Subtitled Video])
    MixAudio --> Final([Final: Subtitled + Voiced])

    classDef tts fill:#fff9c4,stroke:#f9a825
    class QwenTTS,MixAudio tts
```

## GPU Memory Management

```mermaid
flowchart LR
    subgraph MemoryStrategy["Memory Strategy"]
        direction TB
        A[Load Model] --> B[Process Batch]
        B --> C[gc.collect + empty_cache]
        C --> D{More work?}
        D -->|Yes| B
        D -->|No| E[model.cleanup / clear_instance]
    end
    
    subgraph Singletons["Singleton Models"]
        QwenVL[Qwen3-VL Analyzer\n~8GB VRAM]
        QwenTTS[Qwen3-TTS\n~4GB VRAM]
    end
    
    Note["Key: Heavy models are singletons.\nCleanup between phases to avoid OOM.\nrender_video_gpu_isolated uses\nmultiprocessing.spawn for isolation."]
```

## Key Configuration (.env)

| Variable | Module | Description |
|----------|--------|-------------|
| `TARGET_RATIO_W/H` | rendering/render.py | Output aspect ratio (default: 9:16) |
| `AI_PROVIDER` | ai/factory.py | gemini, openai, qwen3vl, local |
| `AI_ANALYSIS_ENABLED` | selection/ranking.py | Enable AI scene ranking |
| `ENABLE_SUBTITLES` | content/subtitles.py | Generate/burn subtitles |
| `ENABLE_TTS` | content/tts.py | Generate voiceover |
| `SUBTITLE_MODE` | content/subtitles.py | speech, ai_captions, story_* |
| `CAPTION_STYLE` | content/subtitles.py | gaming, cinematic, story_epic, etc. |
| `SCENE_LIMIT` | pipeline/runner.py | Max clips per video |

## File References

| Component | File Path |
|-----------|-----------|
| Entry point | `run.py` |
| Pipeline orchestration | `src/pipeline/runner.py` |
| Configuration | `src/core/config.py` |
| Scene detection | `src/modules/analysis/visual.py` |
| Audio analysis | `src/modules/analysis/audio.py` |
| Scene scoring | `src/modules/analysis/scoring.py` |
| Scene filtering | `src/modules/selection/filtering.py` |
| AI ranking | `src/modules/selection/ranking.py` |
| GPU rendering | `src/modules/rendering/render.py` |
| Blur filters | `src/modules/rendering/filters.py` |
| AI providers | `src/modules/ai/providers/*.py` |
| Subtitles | `src/modules/content/subtitles.py` |
| TTS | `src/modules/content/tts.py` |
| Story generation | `src/modules/narrative/generator.py` |
| Dashboard UI | `src/dashboard/` |

---

*File: docs/flow_diagram.md*
*Updated for modular architecture refactor*