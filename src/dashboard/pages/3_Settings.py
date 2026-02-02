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
    "📱 TikTok/Reels": {"ratio": (9, 16), "desc": "9:16 vertical"},
    "📺 YouTube": {"ratio": (16, 9), "desc": "16:9 landscape"},
    "📷 Instagram": {"ratio": (1, 1), "desc": "1:1 square"},
    "🎬 Cinema": {"ratio": (21, 9), "desc": "21:9 ultrawide"},
    "📱 Portrait": {"ratio": (4, 5), "desc": "4:5 tall"},
    "📺 Standard": {"ratio": (4, 3), "desc": "4:3 classic"},
}

LOGO_PATH = Path("assets/logo.png")


def render() -> None:
    st.set_page_config(page_title="Settings - AutoShorts", page_icon="⚙️", layout="wide")
    
    # Sidebar logo
    if LOGO_PATH.exists():
        st.logo(str(LOGO_PATH))
    
    # Import shared theme
    from dashboard.utils.theme import get_shared_css
    
    # Apply shared theme
    st.markdown(get_shared_css(), unsafe_allow_html=True)
    
    init_state()
    
    # Header
    st.markdown("""
        <div class="page-header">
            <h1>⚙️ Settings</h1>
            <p>Configure video format, AI providers, subtitles, and more</p>
        </div>
    """, unsafe_allow_html=True)
    
    values, extras = load_env_values()
    
    # TARGET_RATIO values may be in extras (not in schema)
    current_w = int(extras.get("TARGET_RATIO_W", values.get("TARGET_RATIO_W", "9")))
    current_h = int(extras.get("TARGET_RATIO_H", values.get("TARGET_RATIO_H", "16")))
    
    # Format presets section
    st.markdown('### 📐 Output Format')
    st.caption(f"Current aspect ratio: **{current_w}:{current_h}**")
    st.caption("")  # Add spacing
    
    cols = st.columns(6)
    for idx, (label, preset) in enumerate(FORMAT_PRESETS.items()):
        w, h = preset["ratio"]
        is_current = (w == current_w and h == current_h)
        
        with cols[idx]:
            # All presets are buttons - selected one is primary type
            btn_type = "primary" if is_current else "secondary"
            if st.button(
                f"{label}\n\n{preset['desc']}",
                key=f"preset_{w}_{h}",
                use_container_width=True,
                type=btn_type
            ):
                if not is_current:
                    # Update extras since these aren't in schema
                    extras["TARGET_RATIO_W"] = str(w)
                    extras["TARGET_RATIO_H"] = str(h)
                    save_env_values(values, extras)
                    st.rerun()
    
    st.divider()
    
    # Full config panel with auto-save
    sections = tuple(get_schema())
    render_config_panel(sections)


if __name__ == "__main__":
    render()
