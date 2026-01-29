# 🗺️ AutoShorts Roadmap

This document outlines the planned evolution of AutoShorts. We prioritize features that reduce manual labor for creators while maintaining high technical performance.

## 🚀 Phase 1: Pipeline Refinement

*Focus: Stabilizing the core GPU pipeline and improving output quality.*

- [x] **Project Reorganization**: Move to `src/` structure and clean up imports.
- [x] **Local TTS Integration**: Basic ChatterBox support for offline voiceovers.

## 🗣️ Phase 2: Audio, Voice & UI

*Focus: Pushing the boundaries of open-source TTS and Making the tool accessible.*

**Powered by GitHub Copilot CLI**:
We will extensively use Copilot CLI to accelerate the development of these features.

- [x] **Voice Engine Upgrades**: Integrate **Qwen3-TTS 1.7B-VoiceDesign** for natural language-driven voice design with fine-grained emotion control (funny, angry, hyped) adaptive to gameplay moments.
- [x] **Local Web Dashboard**: A lightweight **Streamlit** (Python) UI running on `localhost` to:
  - Preview generated clips.
  - Manually override AI scores (Action/Funny).
  - Trigger processing jobs.
  - **No Context Switching**: Keeps the entire stack in Python.
- [x] **GenZ Mode**: GenZ slang-heavy caption style (`CAPTION_STYLE=genz`) with modern internet culture vocabulary ("bruh 💀", "no cap", "finna") for higher engagement with younger audiences.
- [x] **Story Mode / Narrative Overlay**: Multiple narrative caption styles for cohesive storytelling:
  - `story_news`: Professional esports broadcaster commentary
  - `story_roast`: Sarcastic, playful roasting narration
  - `story_creepypasta`: Horror-style tension building narrative
  - `story_dramatic`: Epic cinematic movie-trailer narration
  - [ ] **Multi-Speaker / Podcast Mode**: NotebookLM-style narration with distinct character voices via text prompts (e.g., "Sarcastic GenZ commentator", "Hyped esports caster") for dialogue-driven content.

## 🧠 Phase 3: Visual Intelligence

*Focus: Moving beyond center-cropping to active scene understanding.*

- [ ] **Intelligent Auto-Zoom**: Integrate **YOLO / RT-DETR** to identify the "subject" (player, car, crosshair) and dynamically follow it in the 9:16 crop.
- [ ] **Scene Transition Styles**: AI-generated transitions between merged highlights to make the flow more cinematic.
- [ ] **Content Farm Mode**:
  - "Watch-Folder" support for fully automated background processing.
  - **Parallel Batch Processing**: Optimize the scoring engine to handle multiple clips simultaneously on multi-GPU setups.
- [ ] **Facecam / Reaction Support**: Detect if the source video has a facecam/webcam and preserve it in a separate layout layer during the vertical crop.
- [ ] **Cloud-Native Deployment**: Templates for deploying the pipeline as a serverless worker on RunPod, Lambda Labs, or AWS (g5 instances).
- [ ] **Direct Social Upload**: Integration with YouTube/TikTok APIs for one-click publishing.

---

**Want to see a feature?** Open an issue or submit a PR for anything in the roadmap! 🚀
