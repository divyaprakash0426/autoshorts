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


LOGO_PATH = Path("assets/logo.png")


def render() -> None:
    st.set_page_config(page_title="Analytics - AutoShorts", page_icon="📊", layout="wide")
    
    # Sidebar logo
    if LOGO_PATH.exists():
        st.logo(str(LOGO_PATH))
    
    # Import shared theme
    from dashboard.utils.theme import get_shared_css
    
    # Apply shared theme
    st.markdown(get_shared_css(), unsafe_allow_html=True)
    
    # Header
    st.markdown("""
        <div class="page-header">
            <h1>📊 Analytics</h1>
            <p>View statistics about your generated clips</p>
        </div>
    """, unsafe_allow_html=True)
    
    generated = list_videos(Path("generated"))
    if not generated:
        st.info("📁 No generated clips to analyze. Generate some clips first!")
        return

    durations = [info.duration for info in generated]
    sizes = [info.size_mb for info in generated]
    names = [info.path.name for info in generated]

    # Stats cards
    cols = st.columns(4)
    cols[0].metric("🎬 Total clips", len(generated))
    cols[1].metric("⏱️ Total duration", f"{sum(durations):.1f}s")
    cols[2].metric("📊 Avg duration", f"{(sum(durations) / len(durations)):.1f}s")
    cols[3].metric("💾 Total size", f"{sum(sizes):.1f} MB")
    
    st.divider()

    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            x=durations, nbins=10, 
            labels={"x": "Duration (s)"}, 
            title="📈 Clip Duration Distribution",
            color_discrete_sequence=["#6C63FF"]
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(
            x=names[-10:], y=sizes[-10:], 
            labels={"x": "Clip", "y": "Size (MB)"}, 
            title="💾 Recent Clip Sizes (last 10)",
            color_discrete_sequence=["#6C63FF"]
        )
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    render()
