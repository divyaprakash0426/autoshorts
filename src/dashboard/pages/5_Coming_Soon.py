"""Coming Soon - Roadmap features."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Coming Soon - AutoShorts", page_icon="🚀", layout="wide")


def render() -> None:
    st.title("🚀 Coming Soon")
    st.markdown("Features we're working on to make AutoShorts even better!")
    
    st.divider()
    
    # Phase 2 - In Progress
    st.subheader("🗣️ Phase 2: Audio, Voice & UI")
    st.caption("Focus: Pushing the boundaries of open-source TTS and accessibility")
    
    st.markdown("""
    - [ ] **Multi-Speaker / Podcast Mode**  
      NotebookLM-style narration with distinct character voices via text prompts 
      (e.g., "Sarcastic GenZ commentator", "Hyped esports caster")
    """)
    
    st.divider()
    
    # Phase 3 - Visual Intelligence
    st.subheader("🧠 Phase 3: Visual Intelligence")
    st.caption("Focus: Moving beyond center-cropping to active scene understanding")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - [ ] **Intelligent Auto-Zoom**  
          YOLO / RT-DETR integration to identify and dynamically follow 
          the subject (player, car, crosshair) in 9:16 crop
          
        - [ ] **Scene Transition Styles**  
          AI-generated transitions between merged highlights 
          for more cinematic flow
          
        - [ ] **Facecam / Reaction Support**  
          Detect webcam overlays and preserve them in a 
          separate layout layer during vertical crop
        """)
    
    with col2:
        st.markdown("""
        - [ ] **Content Farm Mode**  
          Watch-folder support for fully automated background processing 
          with parallel batch processing on multi-GPU setups
          
        - [ ] **Cloud-Native Deployment**  
          Templates for RunPod, Lambda Labs, or AWS (g5 instances) 
          serverless worker deployment
          
        - [ ] **Direct Social Upload**  
          YouTube/TikTok API integration for one-click publishing
        """)
    
    st.divider()
    
    # Call to action
    st.info("💡 **Want to see a feature?** Open an issue or submit a PR on GitHub!")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.link_button("🐙 GitHub Issues", "https://github.com/nicholascpark/shorts-maker/issues", type="primary")
    with col2:
        st.link_button("📋 Full Roadmap", "https://github.com/nicholascpark/shorts-maker/blob/main/ROADMAP.md")


render()

