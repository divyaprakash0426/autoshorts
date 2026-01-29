from __future__ import annotations

from typing import Dict, List, Optional

import streamlit as st

from dashboard.utils.process import ProcessManager


def get_process_manager() -> ProcessManager:
    if "process_manager" not in st.session_state:
        st.session_state.process_manager = ProcessManager()
    return st.session_state.process_manager


def init_state() -> None:
    st.session_state.setdefault("selected_gameplay", None)
    st.session_state.setdefault("selected_generated", None)
    st.session_state.setdefault("config_dirty", False)
    st.session_state.setdefault("override_scores", {})
    st.session_state.setdefault("job_history", [])
    st.session_state.setdefault("queue", [])
    st.session_state.setdefault("score_overrides", [])


def push_job_history(entry: Dict) -> None:
    history: List[Dict] = st.session_state.get("job_history", [])
    history.append(entry)
    st.session_state["job_history"] = history[-50:]
