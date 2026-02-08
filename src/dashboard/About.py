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
    

    # 2. Logo (Centered)
    if LOGO_PATH.exists():
        # Use columns to center the image
        lc, mc, rc = st.columns([1, 0.8, 1])
        with mc:
            st.image(str(LOGO_PATH), width=180)
            
    # 3. Summary Text (Centered)
    # Wrapped in a div to ensure text alignment since it's outside the hero-container
    st.markdown("""
        <div style="text-align: center; margin-left: 13rem; margin-bottom: 1rem;">
            <p class="hero-summary">
                Turn hours of raw gameplay into viral short-form content in minutes. 
                AutoShorts intelligently analyzes your footage to detect high-impact moments, 
                generates professional voiceovers, and applies dynamic captions—optimized 
                for TikTok, YouTube Shorts, and Instagram Reels.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Examples Divider
    st.divider()

    # Showcase GIFs if available
    showcase_dir = Path("generated/showcase")
    gifs = sorted(showcase_dir.glob("*.gif")) if showcase_dir.exists() else []
    
    if gifs:
        st.markdown("### ✨ Example Outputs")
        cols = st.columns(min(len(gifs), 4))
        for idx, gif in enumerate(gifs[:4]):
            with cols[idx]:
                st.image(str(gif), width="stretch")
                st.caption(f"Sample {idx + 1}")
        st.divider()
    
    # Features Grid with Pills
    st.markdown("### 🚀 Power Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # AI Analysis Card
        st.markdown("""
            <div class="feature-card">
                <h4>🎯 <span>AI Scene Analysis</span></h4>
                <div style="margin-bottom: 1rem;">
                    <span class="feature-pill">🤖 OpenAI & Gemini</span>
                    <span class="feature-pill">📊 Smart Ranking</span>
                    <span class="feature-pill">⏱️ Auto-Cuts</span>
                </div>
                <p style="font-size: 0.9rem; color: #aaa; margin-bottom: 0.5rem;">Automatically detects 7 semantic highlight types:</p>
                <div>
                    <span class="feature-pill" style="border-color: #ff4b4b; background: rgba(255, 75, 75, 0.1);">🔥 Action</span>
                    <span class="feature-pill" style="border-color: #ffd700; background: rgba(255, 215, 0, 0.1);">😂 Funny</span>
                    <span class="feature-pill" style="border-color: #00d26a; background: rgba(0, 210, 106, 0.1);">🏆 Clutch</span>
                    <span class="feature-pill">🤨 WTF</span>
                    <span class="feature-pill">💀 Fail</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Captions Card
        st.markdown("""
            <div class="feature-card">
                <h4>🎙️ <span>Smart Captions</span></h4>
                <div style="margin-bottom: 1rem;">
                    <span class="feature-pill">🎤 Whisper Transcription</span>
                    <span class="feature-pill">📝 AI Commentary</span>
                    <span class="feature-pill">✨ Auto-Style Match</span>
                </div>
                <p style="font-size: 0.9rem; color: #aaa; margin-bottom: 0.5rem;">Dynamic styles for every mood:</p>
                <div>
                    <span class="feature-pill">🎮 Gaming</span>
                    <span class="feature-pill">📢 News</span>
                    <span class="feature-pill">📖 Story</span>
                    <span class="feature-pill">🔥 Roast</span>
                    <span class="feature-pill">👻 Horror</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Voiceover Card
        st.markdown("""
            <div class="feature-card">
                <h4>🔊 <span>AI Voiceover</span></h4>
                <div style="margin-bottom: 1rem;">
                    <span class="feature-pill">🗣️ Qwen3-TTS Engine</span>
                    <span class="feature-pill">🎭 Style-Adaptive</span>
                    <span class="feature-pill">🌍 10+ Languages</span>
                </div>
                <p style="font-size: 0.9rem; color: #aaa; margin-bottom: 0.5rem;">Voices that match the vibe:</p>
                <div>
                    <span class="feature-pill">⚡ Energetic (GenZ)</span>
                    <span class="feature-pill">👔 Professional</span>
                    <span class="feature-pill">😏 Sarcastic</span>
                    <span class="feature-pill">😨 Tense</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # GPU Performance Card
        st.markdown("""
            <div class="feature-card">
                <h4>⚡ <span>GPU Accelerated</span></h4>
                <div style="margin-bottom: 1rem;">
                    <span class="feature-pill">🚀 NVENC Encoding</span>
                    <span class="feature-pill">👁️ CUDA Vision</span>
                    <span class="feature-pill">🔊 GPU Audio Analysis</span>
                </div>
                <div>
                    <span class="feature-pill" style="background: rgba(118, 185, 0, 0.15); border-color: #76b900;">✅ NVIDIA Optimized</span>
                    <span class="feature-pill">10x Faster than CPU</span>
                </div>
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
