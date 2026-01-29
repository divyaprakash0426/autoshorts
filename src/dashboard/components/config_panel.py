from __future__ import annotations

from typing import Dict, Tuple

import streamlit as st

from dashboard.utils.config import EnvSection, coerce_value, load_env_values, save_env_values

TTS_LANGUAGE_LABELS = {
    "en": "English (en)",
    "zh": "Chinese (zh)",
    "ja": "Japanese (ja)",
    "ko": "Korean (ko)",
    "de": "German (de)",
    "fr": "French (fr)",
    "ru": "Russian (ru)",
    "pt": "Portuguese (pt)",
    "es": "Spanish (es)",
    "it": "Italian (it)",
}


def _render_field(field, current_value):
    key = f"cfg_{field.name}"
    if field.field_type == "bool":
        return st.checkbox(field.label, value=bool(current_value), key=key, help=field.help_text)
    if field.field_type == "int":
        return st.number_input(
            field.label,
            value=int(current_value),
            min_value=int(field.min_value) if field.min_value is not None else None,
            max_value=int(field.max_value) if field.max_value is not None else None,
            step=int(field.step) if field.step is not None else 1,
            key=key,
            help=field.help_text,
        )
    if field.field_type == "int_auto":
        # Special field type: checkbox for auto (0), number input for manual override
        is_auto = int(current_value) == 0
        col1, col2 = st.columns([1, 2])
        auto_checked = col1.checkbox("Auto", value=is_auto, key=f"{key}_auto", help="Let system decide automatically")
        if auto_checked:
            return 0
        else:
            manual_value = int(current_value) if int(current_value) > 0 else int(field.min_value or 30)
            return col2.number_input(
                field.label,
                value=manual_value,
                min_value=int(field.min_value) if field.min_value is not None else 1,
                max_value=int(field.max_value) if field.max_value is not None else None,
                step=int(field.step) if field.step is not None else 1,
                key=key,
                help=field.help_text,
            )
    if field.field_type == "float":
        return st.slider(
            field.label,
            min_value=float(field.min_value) if field.min_value is not None else 0.0,
            max_value=float(field.max_value) if field.max_value is not None else 1.0,
            value=float(current_value),
            step=float(field.step) if field.step is not None else 0.01,
            key=key,
            help=field.help_text,
        )
    if field.field_type == "select":
        options = list(field.options or [])
        current_str = str(current_value)
        index = options.index(current_str) if current_str in options else 0

        if field.name == "TTS_LANGUAGE":
            return st.selectbox(
                field.label,
                options=options,
                index=index,
                key=key,
                help=field.help_text,
                format_func=lambda v: TTS_LANGUAGE_LABELS.get(v, v),
            )

        return st.selectbox(
            field.label,
            options=options,
            index=index,
            key=key,
            help=field.help_text,
        )
    if field.field_type == "password":
        return st.text_input(field.label, value=str(current_value), type="password", key=key, help=field.help_text)
    if field.multiline:
        return st.text_area(field.label, value=str(current_value), key=key, help=field.help_text)
    return st.text_input(field.label, value=str(current_value), key=key, help=field.help_text)


def render_config_panel(sections: Tuple[EnvSection, ...]) -> Tuple[Dict[str, str], Dict[str, str]]:
    values, extras = load_env_values()
    updated_values: Dict[str, str] = dict(values)
    
    # Track AI provider for conditional rendering
    ai_provider = values.get("AI_PROVIDER", "openai")

    for section in sections:
        with st.expander(section.title, expanded=section.expanded):
            for field in section.fields:
                # Conditional rendering based on AI provider
                if field.name == "GEMINI_API_KEY" and ai_provider != "gemini":
                    continue  # Hide Gemini key when not using Gemini
                if field.name == "OPENAI_API_KEY" and ai_provider not in ("openai",):
                    continue  # Hide OpenAI key when not using OpenAI
                if field.name == "GEMINI_MODEL" and ai_provider != "gemini":
                    continue
                if field.name in ("OPENAI_MODEL", "OPENAI_TAGGING_MODEL") and ai_provider != "openai":
                    continue
                
                current_value = coerce_value(field, values.get(field.name))
                result = _render_field(field, current_value)
                updated_values[field.name] = result
                
                # Update AI provider tracking if it changed
                if field.name == "AI_PROVIDER":
                    ai_provider = result

    col1, col2 = st.columns([1, 1])
    saved = False
    if col1.button("Save configuration", type="primary"):
        save_env_values(updated_values, extras)
        saved = True
    if col2.button("Reload from .env"):
        st.rerun()
    if saved:
        st.success("Configuration saved to .env")
    return updated_values, extras
