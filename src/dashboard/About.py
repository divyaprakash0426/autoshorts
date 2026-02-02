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


LOGO_PATH = Path("assets/logo.png")


def main() -> None:
    st.set_page_config(page_title="AutoShorts", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")
    
    # Sidebar logo
    if LOGO_PATH.exists():
        st.logo(str(LOGO_PATH))
    
    # Import shared theme
    from dashboard.utils.theme import get_shared_css
    
    # Apply shared theme
    st.markdown(get_shared_css(), unsafe_allow_html=True)
    
    # Hero section - place title inside the blue page-header box
    st.markdown("""
        <div class="page-header" style="text-align: center;">
            <h1 class="hero-title" style="margin:0;">AutoShorts</h1>
        </div>
    """, unsafe_allow_html=True)

    # Logo below header and descriptive text
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=180)
    st.markdown('<p class="hero-subtitle" style="text-align:center; margin-top: 0.5rem; color: #a0a0a0; font-size: 1.1rem;">AI-powered scene analysis • GPU-accelerated rendering • Auto voiceovers & captions</p>', unsafe_allow_html=True)
    
    # Showcase GIFs
    showcase_dir = Path("generated/showcase")
    gifs = sorted(showcase_dir.glob("*.gif")) if showcase_dir.exists() else []
    
    if gifs:
        st.markdown("### ✨ Example Outputs")
        cols = st.columns(min(len(gifs), 4))
        for idx, gif in enumerate(gifs[:4]):
            with cols[idx]:
                st.image(str(gif), use_container_width=True)
                st.caption(f"Sample {idx + 1}")
        st.divider()
    
    # Features grid
    st.markdown("### 🚀 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <h4>🎯 AI Scene Analysis</h4>
                <ul>
                    <li><strong>OpenAI & Gemini</strong> support for smart scene ranking</li>
                    <li><strong>7 semantic types</strong> detected automatically</li>
                    <li>AI picks the most engaging moments from hours of footage</li>
                </ul>
                <p style="font-size: 0.85rem; opacity: 0.7;">
                    Types: action • funny • clutch • wtf • epic_fail • hype • skill
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
            <div class="feature-card">
                <h4>🎙️ Smart Captions</h4>
                <ul>
                    <li><strong>Speech mode:</strong> Transcribe voice with Whisper</li>
                    <li><strong>AI Captions:</strong> Generated commentary for silent gameplay</li>
                    <li><strong>10+ styles:</strong> Gaming, GenZ, Story modes</li>
                    <li><strong>Auto-style:</strong> Matches tone to detected content</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card">
                <h4>🔊 AI Voiceover</h4>
                <ul>
                    <li><strong>Qwen3-TTS</strong> voice synthesis engine</li>
                    <li><strong>Style-adaptive voices</strong> per caption type</li>
                    <li><strong>10 languages</strong> supported</li>
                </ul>
                <p style="font-size: 0.85rem; opacity: 0.7;">
                    GenZ → Energetic • News → Professional • Roast → Sarcastic
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
            <div class="feature-card">
                <h4>⚡ GPU-Accelerated</h4>
                <ul>
                    <li><strong>NVENC</strong> hardware encoding (10x faster)</li>
                    <li><strong>CUDA</strong> scene detection & audio analysis</li>
                    <li><strong>Smart fallbacks</strong> to CPU if GPU unavailable</li>
                    <li>Process a 1-hour video in ~10 minutes</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Current config stats
    st.markdown("### 📊 Current Configuration")
    values, _ = load_env_values()
    
    cols = st.columns(4)
    
    with cols[0]:
        st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value">{values.get("SCENE_LIMIT", 6)}</div>
                <div class="stats-label">🎬 Scene Limit</div>
            </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        caption_style = values.get("CAPTION_STYLE", "auto")
        st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value" style="font-size: 1.2rem;">{caption_style}</div>
                <div class="stats-label">💬 Caption Style</div>
            </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        provider = values.get("AI_PROVIDER", "openai").upper()
        st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value" style="font-size: 1.2rem;">{provider}</div>
                <div class="stats-label">🤖 AI Provider</div>
            </div>
        """, unsafe_allow_html=True)
    
    with cols[3]:
        tts_status = "ON" if str(values.get("ENABLE_TTS", "true")).lower() in ("true", "1") else "OFF"
        tts_color = "#00d26a" if tts_status == "ON" else "#ff6b6b"
        st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value" style="color: {tts_color};">{tts_status}</div>
                <div class="stats-label">🔊 TTS Voiceover</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Getting started
    st.markdown("### 🏁 Getting Started")
    
    steps = st.columns(4)
    
    with steps[0]:
        st.markdown("""
            **1️⃣ Add Videos**
            
            Put your gameplay videos in the `gameplay/` folder
        """)
    
    with steps[1]:
        st.markdown("""
            **2️⃣ Configure**
            
            Adjust settings in ⚙️ Settings or use defaults
        """)
    
    with steps[2]:
        st.markdown("""
            **3️⃣ Generate**
            
            Click **Start Processing** on the Generate page
        """)
    
    with steps[3]:
        st.markdown("""
            **4️⃣ Browse**
            
            View and manage clips in 📁 Browse
        """)
    
    st.divider()
    
    # Footer
    st.markdown("""
        <div style="text-align: center; padding: 1.5rem; opacity: 0.7;">
            <p>Built with ❤️ using PyTorch, Streamlit & FFmpeg</p>
            <p style="font-size: 0.9rem;">
                <a href="https://github.com/divyaprakash0426/autoshorts" target="_blank">GitHub</a> •
                <a href="https://github.com/divyaprakash0426/autoshorts/blob/main/LICENSE" target="_blank">MIT License</a>
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
