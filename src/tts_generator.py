"""ChatterBox TTS integration for AI voiceover generation.

This module provides text-to-speech functionality using ChatterBox,
a high-quality open-source TTS model with emotion control.

Features:
- Local GPU inference (RTX 4080 optimal)
- Emotion control (monotone → dramatic)
- Paralinguistic tags ([laugh], [cough], etc.)
- Zero-shot voice cloning (optional)
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclass
class TTSConfig:
    """Configuration for TTS generation."""
    
    enabled: bool = True
    model: str = "chatterbox"  # Model variant
    language: str = "en"  # Language code (en = English with emotion, others = Multilingual)
    emotion_level: float = 0.5  # 0.0 (monotone) to 1.0 (dramatic) - only for English
    speed: float = 1.0  # Speech speed multiplier
    voice_reference: Optional[Path] = None  # For voice cloning
    device: str = "cuda"  # cuda, cpu
    
    # Supported languages for Multilingual model
    MULTILINGUAL_LANGUAGES = {
        "ar", "da", "de", "el", "es", "fi", "fr", "he", "hi", 
        "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", 
        "sv", "sw", "tr", "zh"
    }
    
    @property
    def use_multilingual(self) -> bool:
        """Check if we should use the multilingual model."""
        return self.language.lower() in self.MULTILINGUAL_LANGUAGES
    
    @property
    def supports_emotion(self) -> bool:
        """Check if current model supports emotion/exaggeration tuning."""
        # Only English model supports CFG & exaggeration
        return not self.use_multilingual
    
    @classmethod
    def from_env(cls) -> "TTSConfig":
        """Create config from environment variables."""
        voice_ref = os.getenv("TTS_VOICE_REFERENCE", "")
        return cls(
            enabled=os.getenv("ENABLE_TTS", "true").lower() in ("true", "1", "yes"),
            model=os.getenv("TTS_MODEL", "chatterbox"),
            language=os.getenv("TTS_LANGUAGE", "en"),
            emotion_level=float(os.getenv("TTS_EMOTION_LEVEL", "0.5")) if os.getenv("TTS_EMOTION_LEVEL", "0.5").replace('.', '', 1).isdigit() else 0.5,
            speed=float(os.getenv("TTS_SPEED", "1.0")),
            voice_reference=Path(voice_ref) if voice_ref else None,
            device=os.getenv("TTS_DEVICE", "cuda"),
        )

# Emotion presets for different styles (TTS exaggeration values)
EMOTION_PRESETS = {
    "monotone": 0.0,
    "calm": 0.2,
    "normal": 0.4,
    "expressive": 0.6,
    "dramatic": 0.8,
    "extreme": 1.0,
}

# Map detected video categories to emotion presets
CATEGORY_TO_PRESET = {
    "action": "dramatic",      # High energy, intense
    "funny": "expressive",     # Playful, animated
    "highlight": "expressive", # Excited but not over the top
}


def get_emotion_for_category(category: str) -> float:
    """Get the appropriate emotion level for a detected category.
    
    Uses the preset system: category -> preset name -> emotion value
    """
    preset_name = CATEGORY_TO_PRESET.get(category, "normal")
    return EMOTION_PRESETS.get(preset_name, 0.5)


class ChatterBoxTTS:
    """ChatterBox TTS wrapper for local inference.
    
    Uses a module-level cache to avoid reloading the model for each clip.
    """
    
    _instance: Optional["ChatterBoxTTS"] = None  # Singleton instance
    
    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig.from_env()
        self._model = None
        self._tokenizer = None
        self._initialized = False
    
    @classmethod
    def get_instance(cls, config: Optional[TTSConfig] = None) -> "ChatterBoxTTS":
        """Get or create the singleton TTS instance (model stays loaded)."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance
    
    @classmethod
    def clear_instance(cls):
        """Clear the cached instance (for cleanup or config changes)."""
        if cls._instance is not None:
            cls._instance._model = None
            cls._instance._initialized = False
            cls._instance = None
    
    def _ensure_initialized(self):
        """Lazy initialization of the model."""
        if self._initialized:
            return
        
        try:
            import torch
            
            if self.config.use_multilingual:
                # Load Multilingual model for non-English languages
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS as CBModel
                logging.info(f"Loading ChatterBox Multilingual TTS model (language: {self.config.language})...")
            else:
                # Load English model with emotion control
                from chatterbox.tts import ChatterboxTTS as CBModel
                logging.info("Loading ChatterBox TTS model (English with emotion control)...")
            
            logging.info(f"Device: {self.config.device}")
            
            # Load the model - it will download automatically on first use
            self._model = CBModel.from_pretrained(
                device=self.config.device
            )
            
            self._initialized = True
            model_type = "Multilingual" if self.config.use_multilingual else "English"
            logging.info(f"ChatterBox TTS model loaded successfully ({model_type})")
            
        except ImportError as e:
            logging.error(f"ChatterBox not installed: {e}")
            logging.error("Run: pip install chatterbox-tts")
            raise
        except Exception as e:
            logging.error(f"Failed to load ChatterBox model: {e}")
            raise
    
    def generate(
        self,
        text: str,
        output_path: Path,
        emotion_level: Optional[float] = None,
        voice_reference: Optional[Path] = None,
    ) -> bool:
        """Generate speech from text.
        
        Args:
            text: Text to speak
            output_path: Path to save audio file (WAV)
            emotion_level: Override emotion level (0.0-1.0)
            voice_reference: Optional audio file for voice cloning
            
        Returns:
            True if generation succeeded
        """
        if not self.config.enabled:
            logging.info("TTS disabled via ENABLE_TTS env var")
            return False
        
        self._ensure_initialized()
        
        emotion = emotion_level if emotion_level is not None else self.config.emotion_level
        ref_audio = voice_reference or self.config.voice_reference
        
        try:
            import torch
            import torchaudio
            
            if self.config.use_multilingual:
                # Multilingual model - no emotion control, uses language_id
                logging.info(f"Generating TTS: {text[:50]}... (lang: {self.config.language})")
                
                if ref_audio and ref_audio.exists():
                    wav = self._model.generate(
                        text=text,
                        audio_prompt_path=str(ref_audio),
                        language_id=self.config.language,
                    )
                else:
                    wav = self._model.generate(
                        text=text,
                        language_id=self.config.language,
                    )
            else:
                # English model - supports emotion/exaggeration
                logging.info(f"Generating TTS: {text[:50]}... (emotion: {emotion:.2f})")
                
                if ref_audio and ref_audio.exists():
                    wav = self._model.generate(
                        text=text,
                        audio_prompt_path=str(ref_audio),
                        exaggeration=emotion,
                    )
                else:
                    wav = self._model.generate(
                        text=text,
                        exaggeration=emotion,
                    )
            
            # Save to file
            torchaudio.save(
                str(output_path),
                wav.cpu(),
                self._model.sr
            )
            
            logging.info(f"TTS audio saved: {output_path.name}")
            return True
            
        except Exception as e:
            logging.error(f"TTS generation failed: {e}")
            return False
    
    def generate_for_captions(
        self,
        captions: List[dict],
        output_path: Path,
        detected_category: str = "action",
    ) -> Optional[Path]:
        """Generate TTS for a list of captions with proper timing alignment.
        
        Each caption's TTS audio is placed at the correct timestamp, with
        silence gaps between captions as needed.
        
        Args:
            captions: List of {"start": float, "end": float, "text": str}
            output_path: Output audio file path
            detected_category: Category for emotion mapping
            
        Returns:
            Path to generated audio, or None if failed
        """
        if not captions:
            logging.warning("No captions to generate TTS for")
            return None
        
        self._ensure_initialized()
        
        # Get emotion level from category
        emotion = get_emotion_for_category(detected_category)
        
        try:
            import torch
            import torchaudio
            
            sample_rate = self._model.sr
            audio_segments = []
            current_time = 0.0  # Track current position in seconds
            
            logging.info(f"Generating TTS for {len(captions)} captions ({detected_category} mode)")
            
            for i, caption in enumerate(captions):
                text = caption.get("text", "").strip()
                start_time = caption.get("start", 0.0)
                end_time = caption.get("end", start_time + 3.0)
                
                if not text:
                    continue
                
                # Calculate silence needed before this caption
                silence_duration = max(0, start_time - current_time)
                if silence_duration > 0:
                    silence_samples = int(silence_duration * sample_rate)
                    silence = torch.zeros(1, silence_samples)
                    audio_segments.append(silence)
                    current_time += silence_duration
                
                # Generate TTS for this caption
                try:
                    ref_audio = self.config.voice_reference
                    
                    if self.config.use_multilingual:
                        # Multilingual - no emotion, uses language_id
                        if ref_audio and ref_audio.exists():
                            wav = self._model.generate(
                                text=text,
                                audio_prompt_path=str(ref_audio),
                                language_id=self.config.language,
                            )
                        else:
                            wav = self._model.generate(
                                text=text,
                                language_id=self.config.language,
                            )
                    else:
                        # English - supports emotion/exaggeration and voice cloning
                        if ref_audio and ref_audio.exists():
                            wav = self._model.generate(
                                text=text,
                                audio_prompt_path=str(ref_audio),
                                exaggeration=emotion,
                            )
                        else:
                            wav = self._model.generate(
                                text=text,
                                exaggeration=emotion,
                            )
                    
                    # wav is (1, samples), get duration
                    tts_duration = wav.shape[-1] / sample_rate
                    caption_duration = end_time - start_time
                    
                    # If TTS is longer than caption duration, speed it up slightly
                    if tts_duration > caption_duration * 1.2:  # Allow 20% overshoot
                        # Resample to fit in caption window
                        target_samples = int(caption_duration * sample_rate)
                        if target_samples > 0 and wav.shape[-1] > 0:
                            wav = torchaudio.transforms.Resample(
                                orig_freq=int(wav.shape[-1]),
                                new_freq=target_samples
                            )(wav)
                            tts_duration = wav.shape[-1] / sample_rate
                    
                    audio_segments.append(wav.cpu())
                    current_time += tts_duration
                    
                except Exception as e:
                    logging.warning(f"TTS failed for caption {i}: {e}")
                    # Add small silence instead
                    silence_samples = int(0.5 * sample_rate)
                    audio_segments.append(torch.zeros(1, silence_samples))
                    current_time += 0.5
            
            if not audio_segments:
                return None
            
            # Concatenate all segments
            final_audio = torch.cat(audio_segments, dim=-1)
            
            torchaudio.save(str(output_path), final_audio, sample_rate)
            
            total_duration = final_audio.shape[-1] / sample_rate
            logging.info(f"TTS audio generated: {total_duration:.1f}s for {len(captions)} captions")
            
            return output_path
            
        except Exception as e:
            logging.error(f"Caption TTS generation failed: {e}")
            return None


def is_tts_enabled() -> bool:
    """Check if TTS is enabled via environment."""
    return os.getenv("ENABLE_TTS", "true").lower() in ("true", "1", "yes")


def get_tts_emotion_for_category(category: str) -> float:
    """Get appropriate emotion level for a detected category."""
    return get_emotion_for_category(category)


def generate_voiceover(
    text: str,
    output_path: Path,
    detected_category: str = "action",
    config: Optional[TTSConfig] = None,
) -> Optional[Path]:
    """Generate voiceover audio from text.
    
    Args:
        text: Text to speak
        output_path: Path to save WAV file
        detected_category: Category for emotion mapping
        config: Optional TTS config override
        
    Returns:
        Path to generated audio, or None if failed
    """
    if not is_tts_enabled():
        return None
    
    cfg = config or TTSConfig.from_env()
    emotion = get_emotion_for_category(detected_category)
    
    # Use singleton to keep model loaded across multiple calls
    tts = ChatterBoxTTS.get_instance(cfg)
    success = tts.generate(text, output_path, emotion_level=emotion)
    
    return output_path if success else None


def mix_audio_with_video(
    video_path: Path,
    voiceover_path: Path,
    output_path: Path,
    game_audio_volume: float = 0.3,
    voiceover_volume: float = 1.0,
) -> bool:
    """Mix voiceover audio with video's existing audio.
    
    Args:
        video_path: Input video with game audio
        voiceover_path: Voiceover WAV file
        output_path: Output video with mixed audio
        game_audio_volume: Volume multiplier for game audio (0.0-1.0)
        voiceover_volume: Volume multiplier for voiceover (0.0-1.0)
        
    Returns:
        True if successful
    """
    # FFmpeg command to mix audio tracks
    # - Lower the game audio volume
    # - Add voiceover on top
    # - Re-encode with NVENC
    
    filter_complex = (
        f"[0:a]volume={game_audio_volume}[game];"
        f"[1:a]volume={voiceover_volume}[voice];"
        "[game][voice]amix=inputs=2:duration=first:dropout_transition=2[aout]"
    )
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(voiceover_path),
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",  # Copy video stream (already encoded)
        "-c:a", "aac",
        "-b:a", "192k",
        str(output_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            logging.error(f"FFmpeg audio mix failed: {result.stderr[:200]}")
            return False
        
        return output_path.exists()
        
    except subprocess.TimeoutExpired:
        logging.error("FFmpeg audio mix timed out")
        return False
    except Exception as e:
        logging.error(f"Audio mixing failed: {e}")
        return False


def download_model():
    """Pre-download the ChatterBox model for offline use."""
    logging.info("Downloading ChatterBox TTS model...")
    
    try:
        from chatterbox.tts import ChatterboxTTS
        
        # This triggers model download
        model = ChatterboxTTS.from_pretrained(device="cuda")
        
        logging.info("Model downloaded successfully!")
        logging.info(f"Sample rate: {model.sr}")
        
        # Quick test
        test_wav = model.generate("Test successful.", exaggeration=0.5)
        logging.info(f"Test generation shape: {test_wav.shape}")
        
        return True
        
    except Exception as e:
        logging.error(f"Model download failed: {e}")
        return False


if __name__ == "__main__":
    # Run model download if executed directly
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        download_model()
    else:
        print("Usage: python tts_generator.py download")
        print("This will download the ChatterBox model for local use.")
