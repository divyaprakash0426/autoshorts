#!/usr/bin/env python3
"""
AutoShorts - Entry point script.
Run this from the project root to generate shorts.
"""
import sys
from pathlib import Path
import signal
import logging
import gc

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from shorts import main


def _cleanup_models():
    """Attempt to clear model singletons and free GPU memory."""
    logging.info("Cleaning up models and GPU resources...")

    # Try clearing Qwen TTS singleton if present
    try:
        from tts_generator import QwenTTS
        QwenTTS.clear_instance()
        logging.info("Cleared QwenTTS instance")
    except Exception as e:
        logging.debug(f"No QwenTTS to clear or clear failed: {e}")

    # Try to free torch CUDA memory if torch is available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("Emptied torch.cuda cache")
    except Exception as e:
        logging.debug(f"torch cleanup failed: {e}")

    # Force garbage collection
    try:
        gc.collect()
        logging.info("Garbage collection complete")
    except Exception:
        pass


def _signal_handler(signum, frame):
    logging.info(f"Received signal {signum}, shutting down...")
    _cleanup_models()
    # Exit cleanly
    sys.exit(0)


# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received, cleaning up...")
        _cleanup_models()
        sys.exit(0)
    finally:
        # Ensure cleanup on normal exit as well
        _cleanup_models()
