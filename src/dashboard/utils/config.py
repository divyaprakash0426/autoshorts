from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import dotenv_values

ENV_PATH = Path(".env")


@dataclass(frozen=True)
class EnvField:
    name: str
    label: str
    field_type: str
    default: Any
    options: Optional[Sequence[str]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    help_text: Optional[str] = None
    multiline: bool = False


@dataclass(frozen=True)
class EnvSection:
    title: str
    fields: Sequence[EnvField]
    expanded: bool = False


def get_schema() -> List[EnvSection]:
    return [
        EnvSection(
            title="Core Settings",
            expanded=True,
            fields=[
                EnvField("X_CENTER", "Crop center X", "float", 0.5, min_value=0.0, max_value=1.0, step=0.01,
                         help_text="Where to center the crop horizontally. 0.5 = middle, 0 = left, 1 = right"),
                EnvField("Y_CENTER", "Crop center Y", "float", 0.5, min_value=0.0, max_value=1.0, step=0.01,
                         help_text="Where to center the crop vertically. 0.5 = middle, 0 = top, 1 = bottom"),
                EnvField("MAX_ERROR_DEPTH", "Max retry depth", "int", 3, min_value=1, max_value=10,
                         help_text="How many times to retry if rendering fails"),
            ],
        ),
        EnvSection(
            title="Clip Length Settings",
            expanded=True,
            fields=[
                EnvField("MIN_SHORT_LENGTH", "Min short length (s)", "int_auto", 0, min_value=5, max_value=120,
                         help_text="Shortest allowed clip. Auto mode picks based on video length"),
                EnvField("MAX_SHORT_LENGTH", "Max short length (s)", "int_auto", 0, min_value=15, max_value=300,
                         help_text="Longest allowed clip. Auto mode picks based on video length"),
                EnvField("MAX_COMBINED_SCENE_LENGTH", "Max combined scene (s)", "int_auto", 0, min_value=30, max_value=600,
                         help_text="Max length when merging multiple scenes. Auto adjusts based on scene limit"),
            ],
        ),
        EnvSection(
            title="AI Providers",
            expanded=True,
            fields=[
                EnvField("AI_PROVIDER", "AI provider", "select", "openai", options=["openai", "gemini", "local"],
                         help_text="Which AI service to use for scene analysis"),
                EnvField("AI_ANALYSIS_ENABLED", "AI analysis enabled", "bool", True,
                         help_text="Use AI to rank scenes by content quality. Disable for faster but less smart selection"),
                EnvField("GEMINI_API_KEY", "Gemini API key", "password", "",
                         help_text="Your Google Gemini API key (get from ai.google.dev)"),
                EnvField("OPENAI_API_KEY", "OpenAI API key", "password", "",
                         help_text="Your OpenAI API key (get from platform.openai.com)"),
                EnvField("GEMINI_MODEL", "Gemini model", "select", "gemini-2.0-flash",
                         options=["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-pro"],
                         help_text="Gemini model for video analysis"),
                EnvField("OPENAI_MODEL", "OpenAI model", "select", "gpt-5-mini",
                         options=["gpt-5-mini", "gpt-5", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
                         help_text="OpenAI model for video analysis"),
                EnvField("OPENAI_TAGGING_MODEL", "OpenAI tagging model", "select", "gpt-5-mini",
                         options=["gpt-5-mini", "gpt-5", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
                         help_text="Model for caption generation (separate from video analysis). Gemini uses same model for both"),
                EnvField("AI_SCORE_WEIGHT", "AI score weight", "float", 0.7, min_value=0.0, max_value=1.0, step=0.05,
                         help_text="How much AI ranking matters vs action detection. 1.0 = AI only, 0 = action only"),
            ],
        ),
        EnvSection(
            title="Semantic Analysis",
            fields=[
                EnvField("CANDIDATE_CLIP_COUNT", "Candidate clip count", "int_auto", 0, min_value=10, max_value=200,
                         help_text="How many clips to send to AI for ranking. More = better selection but higher cost"),
                EnvField("CANDIDATE_CLIP_DURATION", "Candidate clip duration (s)", "int", 120, min_value=10, max_value=600,
                         help_text="Gemini only: Length of video clips for analysis. OpenAI uses frames instead"),
            ],
        ),
        EnvSection(
            title="Subtitles",
            fields=[
                EnvField("ENABLE_SUBTITLES", "Enable subtitles", "bool", True,
                         help_text="Add text captions to generated clips"),
                EnvField(
                    "SUBTITLE_MODE",
                    "Subtitle mode",
                    "select",
                    "ai_captions",
                    options=["speech", "ai_captions", "none"],
                    help_text="speech = transcribe audio, ai_captions = AI-generated commentary, none = no subtitles",
                ),
                EnvField(
                    "WHISPER_MODEL",
                    "Whisper model",
                    "select",
                    "medium",
                    options=["tiny", "base", "small", "medium", "large"],
                    help_text="Speech recognition model. Larger = more accurate but slower",
                ),
                EnvField("MAX_CAPTIONS", "Max captions", "int", 0, min_value=0, max_value=50,
                         help_text="Maximum captions per clip. 0 = Auto (dynamic based on video duration)"),
                EnvField(
                    "PYCAPS_TEMPLATE",
                    "PyCaps template",
                    "select",
                    "hype",
                    options=[
                        "default",
                        "vibrant",
                        "model",
                        "word-focus",
                        "hype",
                        "line-focus",
                        "retro-gaming",
                        "neo-minimal",
                        "minimalist",
                        "classic",
                        "fast",
                        "explosive",
                    ],
                    help_text="Visual style for caption animations",
                ),
                EnvField("ENABLE_AI_CAPTION_ENHANCEMENT", "AI caption enhancement", "bool", True,
                         help_text="Use AI to make captions more engaging and contextual"),
                EnvField("ENABLE_CAPTION_EMOJIS", "Caption emojis", "bool", True,
                         help_text="Add emojis to captions for extra engagement"),
            ],
        ),
        EnvSection(
            title="TTS Voiceover",
            fields=[
                EnvField("ENABLE_TTS", "Enable TTS", "bool", True,
                         help_text="Add AI-generated voiceover narration to clips"),
                EnvField("TTS_MODEL", "TTS model", "select", "qwen", options=["qwen"],
                         help_text="Text-to-speech model (Qwen is fast and high quality)"),
                EnvField("TTS_DEVICE", "TTS device", "select", "cuda", options=["cuda", "cpu"],
                         help_text="cuda = GPU (fast), cpu = processor (slower but works everywhere)"),
                EnvField(
                    "TTS_LANGUAGE",
                    "TTS language",
                    "select",
                    "en",
                    options=["en", "zh", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
                    help_text="Language for voiceover narration",
                ),
                EnvField(
                    "TTS_VOICE_DESCRIPTION",
                    "Voice description override",
                    "text",
                    "",
                    help_text="Custom voice style (e.g. 'energetic male gamer'). Leave blank for auto",
                    multiline=True,
                ),
                EnvField("TTS_GAME_AUDIO_VOLUME", "Game audio volume", "float", 0.3, min_value=0.0, max_value=1.0, step=0.05,
                         help_text="How loud the original game audio is when voiceover plays"),
                EnvField("TTS_VOICEOVER_VOLUME", "Voiceover volume", "float", 1.0, min_value=0.0, max_value=2.0, step=0.05,
                         help_text="How loud the AI voiceover narration is"),
            ],
        ),
        EnvSection(
            title="Decord & Debug",
            fields=[
                EnvField("DECORD_EOF_RETRY_MAX", "Decord EOF retry max", "int", 65536, min_value=1, max_value=200000,
                         help_text="Max retries for video decoding errors (increase if videos fail to load)"),
                EnvField("DECORD_SKIP_TAIL_FRAMES", "Decord skip tail frames", "int", 0, min_value=0, max_value=1000,
                         help_text="Skip frames at end of video (helps with corrupted endings)"),
                EnvField("DEBUG_SKIP_ANALYSIS", "Debug: skip analysis", "bool", False,
                         help_text="Skip video analysis and use cached data (for testing)"),
                EnvField("DEBUG_SKIP_RENDER", "Debug: skip render", "bool", False,
                         help_text="Skip clip rendering (for testing subtitles only)"),
                EnvField("DEBUG_RENDERED_CLIPS", "Debug: rendered clips", "text", "", multiline=True,
                         help_text="Comma-separated list of pre-rendered clip paths (for testing)"),
            ],
        ),
    ]


def _field_map() -> Dict[str, EnvField]:
    return {field.name: field for field in iter_fields()}


def iter_fields() -> Iterable[EnvField]:
    for section in get_schema():
        for field in section.fields:
            yield field


def load_env_values() -> Tuple[Dict[str, str], Dict[str, str]]:
    defaults = {field.name: str(field.default) for field in iter_fields()}
    env_values = dotenv_values(str(ENV_PATH)) if ENV_PATH.exists() else {}
    values = dict(defaults)
    for key, value in env_values.items():
        if value is None:
            continue
        values[key] = value
    extras = {k: v for k, v in env_values.items() if k not in defaults and v is not None}
    return values, extras


def coerce_value(field: EnvField, raw_value: Optional[str]) -> Any:
    if raw_value is None or raw_value == "":
        return field.default
    if field.field_type == "bool":
        return str(raw_value).lower() in ("1", "true", "yes", "on")
    if field.field_type == "int":
        try:
            return int(raw_value)
        except Exception:
            return field.default
    if field.field_type == "float":
        try:
            return float(raw_value)
        except Exception:
            return field.default
    return str(raw_value)


def normalize_value(field: EnvField, value: Any) -> str:
    if field.field_type == "bool":
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def save_env_values(values: Dict[str, Any], extras: Optional[Dict[str, str]] = None) -> None:
    extras = extras or {}
    field_map = _field_map()
    lines: List[str] = []
    for field in iter_fields():
        raw_value = values.get(field.name, field.default)
        lines.append(f"{field.name}={normalize_value(field, raw_value)}")
    if extras:
        lines.append("")
        lines.append("# Extra settings")
        for key, value in sorted(extras.items()):
            if value is None:
                continue
            lines.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(lines) + "\n")
