from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if parent.name == "src":
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

import streamlit as st

from dashboard.components.log_viewer import render_logs
from dashboard.state import get_process_manager, init_state, push_job_history
from dashboard.utils.config import load_env_values, save_env_values
from dashboard.utils.thumbnails import list_videos


def _build_env(overrides: dict) -> dict:
    env = os.environ.copy()
    env.update({k: str(v) for k, v in overrides.items()})
    return env


def render() -> None:
    st.title("🎮 Generate Shorts")
    init_state()

    values, extras = load_env_values()
    gameplay_videos = list_videos(Path("gameplay"))

    st.subheader("Gameplay selection")
    if gameplay_videos:
        options = {info.path.name: info for info in gameplay_videos}
        selection = st.selectbox("Pick a gameplay video", options=list(options.keys()))
        selected = options[selection]
        st.session_state.selected_gameplay = selected.path
        if selected.thumbnail:
            st.image(str(selected.thumbnail), caption=f"{selected.path.name} ({selected.resolution})")
        st.caption(f"Duration: {selected.duration:.1f}s | Size: {selected.size_mb:.1f} MB")
    else:
        st.info("No gameplay videos found in ./gameplay")
        st.caption("Processing uses all files in ./gameplay (selection is for preview only).")

    st.subheader("Quick settings")
    cols = st.columns(2)
    values["SCENE_LIMIT"] = cols[0].number_input(
        "Scene limit",
        value=int(values.get("SCENE_LIMIT", 6)),
        min_value=1,
        max_value=20,
        key="quick_scene_limit",
    )
    values["ENABLE_TTS"] = cols[1].checkbox(
        "Enable TTS",
        value=str(values.get("ENABLE_TTS", "true")).lower() in ("true", "1", "yes"),
        key="quick_enable_tts",
    )

    # Caption style with grouped UX
    st.subheader("Caption Style")
    current_caption = str(values.get("CAPTION_STYLE", "auto"))
    
    GAMING_STYLES = ["gaming", "dramatic", "funny", "minimal"]
    STORY_STYLES = ["story_news", "story_roast", "story_creepypasta", "story_dramatic"]
    
    # Determine current mode from value
    if current_caption == "auto":
        default_mode_idx = 0
    elif current_caption in GAMING_STYLES:
        default_mode_idx = 1
    elif current_caption == "genz":
        default_mode_idx = 2
    elif current_caption in STORY_STYLES:
        default_mode_idx = 3
    else:
        default_mode_idx = 0
    
    mode_options = ["🎮 Gaming (Auto)", "🎮 Gaming (Pick style)", "🔥 GenZ", "📖 Story Mode"]
    mode = st.radio("Mode", mode_options, index=default_mode_idx, horizontal=True, key="caption_mode")
    
    def save_caption(new_style):
        if new_style != current_caption:
            values["CAPTION_STYLE"] = new_style
            save_env_values(values, extras)
    
    if mode == "🎮 Gaming (Auto)":
        st.caption("AI picks best style: gaming, dramatic, funny, or minimal")
        save_caption("auto")
            
    elif mode == "🎮 Gaming (Pick style)":
        style_labels = {"gaming": "🎮 Gaming", "dramatic": "🎭 Dramatic", "funny": "😂 Funny", "minimal": "✨ Minimal"}
        
        current_gaming = current_caption if current_caption in GAMING_STYLES else "gaming"
        selected = st.selectbox(
            "Style",
            options=GAMING_STYLES,
            index=GAMING_STYLES.index(current_gaming),
            format_func=lambda x: style_labels[x],
            key="gaming_style_select",
            on_change=lambda: None,  # Prevent auto-rerun
        )
        save_caption(selected)
            
    elif mode == "🔥 GenZ":
        st.caption("Trendy slang, abbreviations, and emojis")
        save_caption("genz")
            
    elif mode == "📖 Story Mode":
        story_labels = {
            "story_news": "📰 News",
            "story_roast": "🔥 Roast", 
            "story_creepypasta": "👻 Creepypasta",
            "story_dramatic": "🎭 Dramatic",
        }
        
        current_story = current_caption if current_caption in STORY_STYLES else "story_news"
        selected = st.selectbox(
            "Style",
            options=STORY_STYLES,
            index=STORY_STYLES.index(current_story),
            format_func=lambda x: story_labels[x],
            key="story_style_select",
            on_change=lambda: None,  # Prevent auto-rerun
        )
        save_caption(selected)

    manager = get_process_manager()

    st.subheader("Job control")
    control_cols = st.columns(2)
    if control_cols[0].button("Start processing", type="primary"):
        save_env_values(values, extras)
        manager.start(env=_build_env(values))
        push_job_history(
            {
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "video": str(st.session_state.get("selected_gameplay") or "All gameplay videos"),
                "status": "running",
            }
        )
        st.rerun()
    if control_cols[1].button("Stop"):
        manager.stop()
        st.rerun()

    status = manager.status()
    if status.running:
        st.success(f"Running (PID {status.pid})")
    elif status.exit_code is not None:
        st.info(f"Last run exited with code {status.exit_code}")
    else:
        st.warning("Idle")

    show_all = st.checkbox("Show all logs", value=False)
    render_logs(status.tail, show_all=show_all)

    # Auto-refresh while running
    if status.running:
        import time
        time.sleep(2)
        st.rerun()


if __name__ == "__main__":
    render()
