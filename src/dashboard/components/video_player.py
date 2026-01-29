from __future__ import annotations

from pathlib import Path

import streamlit as st


def render_video(path: Path, caption: str = "") -> None:
    if not path or not path.exists():
        st.warning("Video not found.")
        return
    st.video(str(path))
    if caption:
        st.caption(caption)
