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
    

    # 2. Logo + Summary (Centered together)
    if LOGO_PATH.exists():
        lc, mc, rc = st.columns([1, 2, 1])
        with mc:
            st.image(str(LOGO_PATH), width=180)
            st.markdown("""
                <p class="hero-summary">
                    Turn hours of raw footage into viral short-form content in minutes. 
                    AutoShorts intelligently analyzes your videos to detect high-impact moments, 
                    generates professional voiceovers, and applies dynamic captions—optimized 
                    for TikTok, YouTube Shorts, and Instagram Reels.
                </p>
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
                <p style="font-size: 0.9rem; color: #94A3B8; margin-bottom: 0.5rem;">Automatically detects 7 semantic highlight types:</p>
                <div>
                    <span class="feature-pill" style="border-color: #F87171; background: rgba(248, 113, 113, 0.1);">🔥 Action</span>
                    <span class="feature-pill" style="border-color: #FBBF24; background: rgba(251, 191, 36, 0.1);">😂 Funny</span>
                    <span class="feature-pill" style="border-color: #34D399; background: rgba(52, 211, 153, 0.1);">🏆 Clutch</span>
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
                <p style="font-size: 0.9rem; color: #94A3B8; margin-bottom: 0.5rem;">Dynamic styles for every mood:</p>
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
                <p style="font-size: 0.9rem; color: #94A3B8; margin-bottom: 0.5rem;">Voices that match the vibe:</p>
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
                    <span class="feature-pill" style="background: rgba(52, 211, 153, 0.1); border-color: #34D399;">✅ NVIDIA Optimized</span>
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
        tts_color = "#34D399" if tts_status == "ON" else "#F87171"
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
            <div class="step-card">
                <div class="step-number">1</div>
                <h4>Add Videos</h4>
                <p>Put your videos in the <code>gameplay/</code> folder or use the file picker</p>
            </div>
        """, unsafe_allow_html=True)
    
    with steps[1]:
        st.markdown("""
            <div class="step-card">
                <div class="step-number">2</div>
                <h4>Configure</h4>
                <p>Adjust settings in ⚙️ Settings or use the smart defaults</p>
            </div>
        """, unsafe_allow_html=True)
    
    with steps[2]:
        st.markdown("""
            <div class="step-card">
                <div class="step-number">3</div>
                <h4>Generate</h4>
                <p>Click <strong>Start Processing</strong> on the Generate page</p>
            </div>
        """, unsafe_allow_html=True)
    
    with steps[3]:
        st.markdown("""
            <div class="step-card">
                <div class="step-number">4</div>
                <h4>Browse</h4>
                <p>View and manage your clips in 📁 Browse</p>
            </div>
        """, unsafe_allow_html=True)
    
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
