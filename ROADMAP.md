# 🗺️ AutoShorts Roadmap

This document outlines the planned evolution of AutoShorts. We prioritize features that reduce manual labor for creators while maintaining high technical performance.

## 🚀 Phase 1: Pipeline Refinement (Current)

*Focus: Stabilizing the core GPU pipeline and improving output quality.*

- [x] **Project Reorganization**: Move to `src/` structure and clean up imports.
- [x] **Local TTS Integration**: Basic ChatterBox support for offline voiceovers.
- [ ] **Adaptive Audio Mixing**: Improve the ducking logic to ensure game audio doesn't drown out the AI voiceover.
- [ ] **Parallel Batch Processing**: Optimize the scoring engine to handle multiple clips simultaneously on multi-GPU setups.

## 🧠 Phase 2: Visual Intelligence

*Focus: Moving beyond center-cropping to active scene understanding.*

- [ ] **Intelligent Auto-Zoom**: Integrate **YOLO / RT-DETR** to identify the "subject" (player, car, crosshair) and dynamically follow it in the 9:16 crop.
- [ ] **Scene Transition Styles**: AI-generated transitions between merged highlights to make the flow more cinematic.
- [ ] **Motion Blur Smoothing**: Add optional GPU-calculated motion blur for fast-moving gameplay segments.

## 🗣️ Phase 3: Audio & Voice Upgrades

*Focus: Pushing the boundaries of open-source TTS.*

- [ ] **Voice Engine Upgrades**: Support for **ChatterBoxTurbo**, **Qwen-TTS**, and **NVIDIA's latest TTS models**.
- [ ] **Custom Voice Cloning**: Simplified CLI interface to clone a creator's voice from a short 10-second reference file.
- [ ] **Multi-Speaker Support**: Allowing different voices for "Narrator" vs. "Game Character" roles in the shorts.

## 🌐 Phase 4: Platform & UI

*Focus: Making the tool accessible to non-CLI users.*

- [ ] **Watch-Folder Support**: Automatically process any new video dropped into a specific directory.
- [ ] **Web-based Dashboard**: A lightweight Next.js/Vite UI to preview generated clips and manually override AI scores before final render.
- [ ] **Cloud-Native Deployment**: Templates for deploying the pipeline as a serverless worker on RunPod, Lambda Labs, or AWS (g5 instances).

---

**Want to see a feature?** Open an issue or submit a PR for anything in the roadmap! 🚀
