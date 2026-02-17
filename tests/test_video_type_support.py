import os
import sys
from pathlib import Path
from unittest.mock import MagicMock


# Mock heavy dependencies before importing shorts
sys.modules["torch"] = MagicMock()
sys.modules["torchaudio"] = MagicMock()
sys.modules["cupy"] = MagicMock()
sys.modules["cupyx"] = MagicMock()
sys.modules["cupyx.scipy"] = MagicMock()
sys.modules["cupyx.scipy.ndimage"] = MagicMock()
decord_mock = MagicMock()
decord_mock.cpu = lambda x: f"cpu({x})"
decord_mock.gpu = lambda x: f"gpu({x})"
decord_mock.bridge.set_bridge = MagicMock()
sys.modules["decord"] = decord_mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shorts import get_video_type, should_bypass_heuristic_prefilter  # noqa: E402


def test_bypass_heuristics_for_non_gaming_gemini():
    old_provider = os.environ.get("AI_PROVIDER")
    old_type = os.environ.get("VIDEO_TYPE")
    os.environ["AI_PROVIDER"] = "gemini"
    os.environ["VIDEO_TYPE"] = "sports"
    try:
        assert should_bypass_heuristic_prefilter() is True
    finally:
        if old_provider is None:
            os.environ.pop("AI_PROVIDER", None)
        else:
            os.environ["AI_PROVIDER"] = old_provider
        if old_type is None:
            os.environ.pop("VIDEO_TYPE", None)
        else:
            os.environ["VIDEO_TYPE"] = old_type


def test_no_bypass_for_gaming():
    old_provider = os.environ.get("AI_PROVIDER")
    old_type = os.environ.get("VIDEO_TYPE")
    os.environ["AI_PROVIDER"] = "openai"
    os.environ["VIDEO_TYPE"] = "gaming"
    try:
        assert should_bypass_heuristic_prefilter() is False
    finally:
        if old_provider is None:
            os.environ.pop("AI_PROVIDER", None)
        else:
            os.environ["AI_PROVIDER"] = old_provider
        if old_type is None:
            os.environ.pop("VIDEO_TYPE", None)
        else:
            os.environ["VIDEO_TYPE"] = old_type


def test_video_type_normalization():
    old_type = os.environ.get("VIDEO_TYPE")
    os.environ["VIDEO_TYPE"] = "Podcasts"
    try:
        assert get_video_type() == "podcasts"
    finally:
        if old_type is None:
            os.environ.pop("VIDEO_TYPE", None)
        else:
            os.environ["VIDEO_TYPE"] = old_type
