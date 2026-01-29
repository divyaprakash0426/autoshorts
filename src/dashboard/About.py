from __future__ import annotations

import sys
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if parent.name == "src":
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

import streamlit as st

from dashboard.utils.config import load_env_values


def main() -> None:
    st.set_page_config(page_title="AutoShorts", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")
    
    # Hero section
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="font-size: 3rem; margin-bottom: 0;">🎬 AutoShorts</h1>
            <p style="font-size: 1.3rem; color: #888; margin-top: 0.5rem;">
                Turn hours of gameplay into viral shorts in minutes
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Tagline
    st.markdown(
        """
        <div style="text-align: center; padding: 0 2rem 2rem 2rem;">
            <p style="font-size: 1.1rem;">
                AI-powered scene analysis • GPU-accelerated rendering • Auto voiceovers & captions
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Showcase GIFs
    st.subheader("✨ Example Outputs")
    showcase_dir = Path("generated/showcase")
    gifs = sorted(showcase_dir.glob("*.gif")) if showcase_dir.exists() else []
    
    if gifs:
        cols = st.columns(len(gifs))
        for idx, gif in enumerate(gifs[:4]):
            with cols[idx]:
                st.image(str(gif), width="stretch")
                st.caption(f"Sample {idx + 1}")
    else:
        st.info("Run your first generation to see sample outputs here!")
    
    st.divider()
    
    # Features grid
    st.subheader("🚀 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            #### 🎯 AI Scene Analysis
            - **OpenAI & Gemini** support for smart scene ranking
            - **7 semantic types** detected automatically:
              - `action` `funny` `clutch` `wtf` `epic_fail` `hype` `skill`
            - AI picks the most engaging moments from hours of footage
            """
        )
        
        st.markdown(
            """
            #### 🎙️ Smart Captions
            - **Speech mode**: Transcribe voice with Whisper
            - **AI Captions**: Generated commentary for silent gameplay
            - **10+ styles**: Gaming, GenZ, Story modes (News, Roast, Creepypasta)
            - **Auto-style**: Matches caption tone to detected content
            """
        )
    
    with col2:
        st.markdown(
            """
            #### 🔊 AI Voiceover
            - **Qwen3-TTS** voice synthesis engine
            - **Style-adaptive voices** per caption type:
              - GenZ → Energetic casual voice
              - News → Professional broadcaster
              - Roast → Sarcastic narrator
              - Creepypasta → Deep ominous tone
            - **10 languages** supported
            """
        )
        
        st.markdown(
            """
            #### ⚡ GPU-Accelerated
            - **NVENC** hardware encoding (10x faster)
            - **CUDA** scene detection & audio analysis
            - **Smart fallbacks** to CPU if GPU unavailable
            - Process a 1-hour video in ~10 minutes
            """
        )
    
    st.divider()
    
    # Quick stats
    st.subheader("📊 Current Configuration")
    values, _ = load_env_values()
    
    cols = st.columns(4)
    cols[0].metric("🎬 Scene Limit", values.get("SCENE_LIMIT", 6))
    cols[1].metric("💬 Caption Style", values.get("CAPTION_STYLE", "auto"))
    cols[2].metric("🤖 AI Provider", values.get("AI_PROVIDER", "openai").upper())
    cols[3].metric("🔊 TTS", "ON" if str(values.get("ENABLE_TTS", "true")).lower() in ("true", "1") else "OFF")
    
    st.divider()
    
    # Getting started
    st.subheader("🏁 Getting Started")
    st.markdown(
        """
        1. **Add gameplay videos** to the `gameplay/` folder
        2. **Configure settings** in the ⚙️ Settings page or use defaults
        3. **Go to Generate** and click **Start Processing**
        4. **Browse results** in the 📁 Browse page
        
        ---
        
        <div style="text-align: center; padding: 1rem; color: #888;">
            <p>Built with ❤️ using PyTorch, Streamlit & FFmpeg</p>
            <p style="font-size: 0.9rem;">
                <a href="https://github.com/divyaprakash0426/autoshorts" target="_blank">GitHub</a> •
                <a href="https://github.com/divyaprakash0426/autoshorts/blob/main/LICENSE" target="_blank">MIT License</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
