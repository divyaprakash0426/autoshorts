from __future__ import annotations

import sys
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if parent.name == "src":
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

import streamlit as st

from dashboard.components.video_player import render_video
from dashboard.utils.thumbnails import list_videos


def _render_video_grid(videos, cols_count=3, key_prefix="vid"):
    """Render videos as thumbnail grid with click-to-play."""
    if not videos:
        st.info("No videos found.")
        return
    
    # Initialize selected video in session state
    selected_key = f"{key_prefix}_selected"
    if selected_key not in st.session_state:
        st.session_state[selected_key] = None
    
    # Video player at TOP (visible immediately)
    if st.session_state[selected_key]:
        selected_path = st.session_state[selected_key]
        if selected_path.exists():
            st.subheader(f"▶️ {selected_path.name}")
            st.video(str(selected_path))
            if st.button("✕ Close", key=f"{key_prefix}_close"):
                st.session_state[selected_key] = None
                st.rerun()
            st.divider()
        else:
            st.warning(f"Video not found: {selected_path}")
    
    # Thumbnail grid
    rows = [videos[i:i + cols_count] for i in range(0, len(videos), cols_count)]
    for row in rows:
        cols = st.columns(cols_count)
        for idx, info in enumerate(row):
            with cols[idx]:
                # Thumbnail as button
                if info.thumbnail and info.thumbnail.exists():
                    st.image(str(info.thumbnail), width="stretch")
                else:
                    st.markdown("🎬")
                
                # Video name and info
                st.caption(f"**{info.path.name}**")
                st.caption(f"{info.duration:.0f}s | {info.size_mb:.1f}MB")
                
                if st.button("▶️ Play", key=f"{key_prefix}_{info.path.stem}"):
                    st.session_state[selected_key] = info.path
                    st.rerun()


def render() -> None:
    st.title("📁 Browse Files")

    tab1, tab2 = st.tabs(["🎬 Generated Clips", "🎮 Gameplay Videos"])
    
    with tab1:
        generated_videos = list_videos(Path("generated"))
        # Sort by modification time, newest first
        generated_videos = sorted(generated_videos, key=lambda v: v.path.stat().st_mtime, reverse=True)
        _render_video_grid(generated_videos, cols_count=3, key_prefix="gen")
    
    with tab2:
        gameplay_videos = list_videos(Path("gameplay"))
        _render_video_grid(gameplay_videos, cols_count=3, key_prefix="gameplay")


if __name__ == "__main__":
    render()
