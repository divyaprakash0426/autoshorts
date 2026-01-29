from __future__ import annotations

from typing import Dict, List

import streamlit as st


def render_score_editor(scores: List[Dict]) -> List[Dict]:
    if not scores:
        st.info("No scores to display yet.")
        return scores
    updated = []
    for item in scores:
        cols = st.columns([3, 1, 1, 1])
        cols[0].markdown(f"**{item['label']}**")
        cols[1].markdown(f"AI: {item.get('ai_score', 0):.3f}")
        manual = cols[2].number_input(
            "Manual",
            value=float(item.get("manual_score", item.get("ai_score", 0.0))),
            key=f"score_{item['id']}",
        )
        include = cols[3].checkbox("Include", value=item.get("include", True), key=f"include_{item['id']}")
        updated.append({**item, "manual_score": manual, "include": include})
    return updated
