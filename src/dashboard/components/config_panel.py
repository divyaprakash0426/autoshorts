from __future__ import annotations

from typing import Dict, Tuple

import streamlit as st

from dashboard.utils.config import EnvSection, coerce_value, load_env_values, save_env_values

TTS_LANGUAGE_LABELS = {
    "en": "🇺🇸 English",
    "zh": "🇨🇳 Chinese",
    "ja": "🇯🇵 Japanese",
    "ko": "🇰🇷 Korean",
    "de": "🇩🇪 German",
    "fr": "🇫🇷 French",
    "ru": "🇷🇺 Russian",
    "pt": "🇵🇹 Portuguese",
    "es": "🇪🇸 Spanish",
    "it": "🇮🇹 Italian",
}

# Section icons for visual enhancement
SECTION_ICONS = {
    "Core Settings": "🎯",
    "Clip Length Settings": "⏱️",
    "AI Providers": "🤖",
    "Semantic Analysis": "🔍",
    "Subtitles": "💬",
    "TTS Voiceover": "🔊",
    "Decord & Debug": "🔧",
    "Rendering Settings": "🎬",
}


def _render_field(field, current_value):
    key = f"cfg_{field.name}"
    if field.field_type == "bool":
        return st.toggle(field.label, value=bool(current_value), key=key, help=field.help_text)
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
        is_auto = int(current_value) == 0
        col1, col2 = st.columns([1, 2])
        auto_checked = col1.toggle("Auto", value=is_auto, key=f"{key}_auto", help="Let system decide automatically")
        if auto_checked:
            col2.caption(f"_{field.label}: Auto-calculated_")
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
                label_visibility="collapsed",
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
        return st.text_area(field.label, value=str(current_value), key=key, help=field.help_text, height=100)
    return st.text_input(field.label, value=str(current_value), key=key, help=field.help_text)


def render_config_panel(sections: Tuple[EnvSection, ...]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Render config panel with auto-save functionality."""
    
    # Custom styling
    st.markdown("""
        <style>
        .stExpander {
            border: 1px solid #2D3348;
            border-radius: 8px;
            margin-bottom: 0.5rem;
        }
        .stExpander > div:first-child {
            background: linear-gradient(90deg, #1A1D2E 0%, #1E3A5F 100%);
            border-radius: 8px 8px 0 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    values, extras = load_env_values()
    updated_values: Dict[str, str] = dict(values)
    
    # Track AI provider for conditional rendering
    ai_provider = values.get("AI_PROVIDER", "openai")
    
    # Track changes for auto-save
    changes_made = False

    for section in sections:
        icon = SECTION_ICONS.get(section.title, "📋")
        with st.expander(f"{icon} {section.title}", expanded=section.expanded):
            # Create columns for better layout in some sections
            for field in section.fields:
                # Conditional rendering based on AI provider
                if field.name == "GEMINI_API_KEY" and ai_provider != "gemini":
                    continue
                if field.name == "OPENAI_API_KEY" and ai_provider not in ("openai",):
                    continue
                if field.name == "GEMINI_MODEL" and ai_provider != "gemini":
                    continue
                if field.name == "GEMINI_DEEP_ANALYSIS" and ai_provider != "gemini":
                    continue
                if field.name in ("OPENAI_MODEL", "OPENAI_TAGGING_MODEL") and ai_provider != "openai":
                    continue
                
                current_value = coerce_value(field, values.get(field.name))
                result = _render_field(field, current_value)
                
                # Check if value changed
                if str(result) != str(current_value):
                    changes_made = True
                
                updated_values[field.name] = result
                
                # Update AI provider tracking if it changed
                if field.name == "AI_PROVIDER":
                    ai_provider = result

    # Auto-save when changes are detected
    if changes_made:
        save_env_values(updated_values, extras)
        st.toast("✅ Settings saved", icon="💾")
    
    # Manual save button as backup
    col1, col2 = st.columns([1, 3])
    if col1.button("💾 Save All", type="primary", width="stretch"):
        save_env_values(updated_values, extras)
        st.success("Configuration saved!")
    
    col2.caption("_Settings are auto-saved when you make changes_")
    
    return updated_values, extras
