"""Coming Soon - Roadmap features."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Coming Soon - AutoShorts", page_icon="🚀", layout="wide")

LOGO_PATH = Path("assets/logo.png")


def render() -> None:
    # Sidebar logo
    if LOGO_PATH.exists():
        st.logo(str(LOGO_PATH))
    
    # Import shared theme
    from dashboard.utils.theme import get_shared_css
    
    # Apply shared theme
    st.markdown(get_shared_css(), unsafe_allow_html=True)
    
    # Additional roadmap-specific styling
    st.markdown("""
        <style>
        .feature-item {
            background: linear-gradient(145deg, rgba(0, 255, 136, 0.03) 0%, rgba(0, 210, 106, 0.05) 100%);
            border-left: 3px solid #00d26a;
            padding: 1rem 1.25rem;
            margin: 0.75rem 0;
            border-radius: 0 10px 10px 0;
            transition: all 0.2s;
        }
        .feature-item:hover {
            background: linear-gradient(145deg, rgba(0, 255, 136, 0.08) 0%, rgba(0, 210, 106, 0.10) 100%);
            border-left-width: 4px;
            transform: translateX(2px);
        }
        .feature-item h4 {
            margin: 0 0 0.5rem 0;
            font-size: 1.1rem;
            color: #00ff88;
        }
        .feature-item p {
            margin: 0;
            color: #a0a0a0;
            line-height: 1.5;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
        <div class="hero-section">
            <h1>🚀 Coming Soon</h1>
            <p style="color: #a0a0a0; font-size: 1.2rem; margin-top: 1rem;">
                Features we're building to make AutoShorts even more powerful
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Phase 2 - Audio & Voice
    st.markdown("### 🗣️ Phase 2: Audio, Voice & UI")
    st.caption("Focus: Pushing the boundaries of open-source TTS and accessibility")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown("""
                <div class="feature-item">
                    <h4>🎙️ Multi-Speaker / Podcast Mode</h4>
                    <p>NotebookLM-style narration with distinct character voices via text prompts
                    (e.g., "Sarcastic GenZ commentator", "Hyped esports caster")</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div class="feature-item">
                    <h4>🎨 Enhanced Dashboard UI</h4>
                    <p>Modern, responsive interface with real-time progress tracking and 
                    better visual feedback</p>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown("""
                <div class="feature-item">
                    <h4>🌍 Multi-Language Expansion</h4>
                    <p>Extended language support with regional voice presets and 
                    automatic subtitle translation</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div class="feature-item">
                    <h4>🎵 Background Music Integration</h4>
                    <p>Auto-ducking, mood-matched royalty-free music selection 
                    based on clip category</p>
                </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Phase 3 - Visual Intelligence
    st.markdown("### 🧠 Phase 3: Visual Intelligence")
    st.caption("Focus: Moving beyond center-cropping to active scene understanding")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="feature-item">
                <h4>🎯 Intelligent Auto-Zoom</h4>
                <p>YOLO / RT-DETR integration to identify and dynamically follow 
                the subject (player, car, crosshair) in 9:16 crop</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="feature-item">
                <h4>🎬 Scene Transition Styles</h4>
                <p>AI-generated transitions between merged highlights 
                for more cinematic flow</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="feature-item">
                <h4>📹 Facecam / Reaction Support</h4>
                <p>Detect webcam overlays and preserve them in a 
                separate layout layer during vertical crop</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-item">
                <h4>🏭 Content Farm Mode</h4>
                <p>Watch-folder support for fully automated background processing 
                with parallel batch processing on multi-GPU setups</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="feature-item">
                <h4>☁️ Cloud-Native Deployment</h4>
                <p>Templates for RunPod, Lambda Labs, or AWS (g5 instances) 
                serverless worker deployment</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="feature-item">
                <h4>📤 Direct Social Upload</h4>
                <p>YouTube/TikTok API integration for one-click publishing
                with auto-generated titles and hashtags</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Call to action
    st.info("💡 **Want to see a feature prioritized?** Open an issue or submit a PR on GitHub!")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.link_button("🐙 GitHub Issues", "https://github.com/nicholascpark/shorts-maker/issues", type="primary")
    with col2:
        st.link_button("📋 Full Roadmap", "https://github.com/nicholascpark/shorts-maker/blob/main/ROADMAP.md")


render()

