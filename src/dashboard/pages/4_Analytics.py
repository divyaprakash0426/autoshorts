from __future__ import annotations

import sys
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if parent.name == "src":
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

import plotly.express as px
import streamlit as st

from dashboard.utils.thumbnails import list_videos


def render() -> None:
    st.title("📊 Analytics")
    generated = list_videos(Path("generated"))
    if not generated:
        st.info("No generated clips to analyze.")
        return

    durations = [info.duration for info in generated]
    sizes = [info.size_mb for info in generated]
    names = [info.path.name for info in generated]

    st.metric("Total clips", len(generated))
    st.metric("Total duration (s)", f"{sum(durations):.1f}")
    st.metric("Average clip duration (s)", f"{(sum(durations) / len(durations)):.1f}")

    fig = px.histogram(x=durations, nbins=10, labels={"x": "Duration (s)"}, title="Clip Duration Distribution")
    st.plotly_chart(fig)

    fig2 = px.bar(x=names, y=sizes, labels={"x": "Clip", "y": "Size (MB)"}, title="Clip Size (MB)")
    st.plotly_chart(fig2)


if __name__ == "__main__":
    render()
