from __future__ import annotations

import re
from typing import List

import streamlit as st

# Patterns for important log lines
IMPORTANT_PATTERNS = [
    re.compile(r"Scene\s+\d+:", re.IGNORECASE),
    re.compile(r"#\d+:.*Score", re.IGNORECASE),
    re.compile(r"Analyzing video", re.IGNORECASE),
    re.compile(r"AI.*(analysis|ranking)", re.IGNORECASE),
    re.compile(r"Selected.*scene", re.IGNORECASE),
    re.compile(r"Using.*scenes", re.IGNORECASE),
    re.compile(r"(error|exception|failed|traceback)", re.IGNORECASE),
    re.compile(r"(complete|finished|done|success)", re.IGNORECASE),
    re.compile(r"Generating.*clip", re.IGNORECASE),
    re.compile(r"TTS|caption", re.IGNORECASE),
    re.compile(r"Extracting.*clip", re.IGNORECASE),
    re.compile(r"^\d+%|100%"),
    re.compile(r"Scene candidates", re.IGNORECASE),
    re.compile(r"Sorted scenes", re.IGNORECASE),
    re.compile(r"Running", re.IGNORECASE),
    re.compile(r"Computing", re.IGNORECASE),
]


def filter_logs(lines: List[str]) -> List[str]:
    """Filter logs to show only important lines."""
    filtered = []
    for line in lines:
        if any(p.search(line) for p in IMPORTANT_PATTERNS):
            filtered.append(line)
    return filtered if filtered else lines[-20:]  # Fallback to last 20 if nothing matches


def render_logs(lines: List[str], show_all: bool = False) -> None:
    if not lines:
        st.info("No logs yet.")
        return
    
    display_lines = lines if show_all else filter_logs(lines)
    # Reverse order - newest first so users see latest without scrolling
    reversed_lines = list(reversed(display_lines))
    st.text_area(
        f"Processing logs ({len(display_lines)}/{len(lines)} lines) - newest first",
        value="\n".join(reversed_lines),
        height=300,
    )
