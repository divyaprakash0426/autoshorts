from __future__ import annotations

import sys
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if parent.name == "src":
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

import streamlit as st

from dashboard.components.config_panel import render_config_panel
from dashboard.state import init_state
from dashboard.utils.config import get_schema, load_env_values, save_env_values

# Common video format presets
FORMAT_PRESETS = {
    "📱 TikTok/Reels (9:16)": (9, 16),
    "📺 YouTube (16:9)": (16, 9),
    "📷 Instagram Square (1:1)": (1, 1),
    "🎬 Cinema (21:9)": (21, 9),
    "📱 Portrait (4:5)": (4, 5),
    "📺 Standard (4:3)": (4, 3),
}


def render() -> None:
    st.title("⚙️ Settings")
    init_state()
    
    # Format presets section
    st.subheader("Quick Format Presets")
    values, extras = load_env_values()
    
    current_w = int(values.get("TARGET_RATIO_W", 9))
    current_h = int(values.get("TARGET_RATIO_H", 16))
    st.caption(f"Current: {current_w}:{current_h}")
    
    cols = st.columns(3)
    for idx, (label, (w, h)) in enumerate(FORMAT_PRESETS.items()):
        col = cols[idx % 3]
        is_current = (w == current_w and h == current_h)
        button_type = "primary" if is_current else "secondary"
        if col.button(label, key=f"preset_{w}_{h}", type=button_type):
            values["TARGET_RATIO_W"] = w
            values["TARGET_RATIO_H"] = h
            save_env_values(values, extras)
            st.rerun()
    
    st.divider()
    
    # Full config panel
    sections = tuple(get_schema())
    render_config_panel(sections)


if __name__ == "__main__":
    render()
