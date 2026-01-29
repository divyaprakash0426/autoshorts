"""Qwen3-TTS integration for AI voiceover generation.

This module provides text-to-speech functionality using Qwen3-TTS VoiceDesign,
enabling natural language voice creation via text prompts.

Features:
- Local GPU inference (RTX 4080 optimal)
- Dynamic AI-generated voice descriptions based on category + captions
- Fine-grained voice control via natural language prompts
- 10 language support (en, zh, ja, ko, de, fr, ru, pt, es, it)

Requirements:
- qwen-tts package: pip install qwen-tts
- sox system package: sudo pacman -S sox (Arch) / apt install sox (Debian)
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
    model: str = "qwen"  # Model variant (qwen)
    language: str = "en"  # Language code
    voice_description: Optional[str] = None  # Override voice description
    device: str = "cuda"  # cuda, cpu
    
    # Supported languages for Qwen3-TTS (must use full names)
    SUPPORTED_LANGUAGES = {"en", "zh", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"}
    
    # Map short codes to full language names for Qwen3-TTS API
    LANGUAGE_MAP = {
        "en": "English",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "de": "German",
        "fr": "French",
        "ru": "Russian",
        "pt": "Portuguese",
        "es": "Spanish",
        "it": "Italian",
    }
    
    def get_language_name(self) -> str:
        """Get full language name for Qwen3-TTS API."""
        return self.LANGUAGE_MAP.get(self.language, "English")
    
    @classmethod
    def from_env(cls) -> "TTSConfig":
        """Create config from environment variables."""
        voice_desc = os.getenv("TTS_VOICE_DESCRIPTION", "").strip()
        return cls(
            enabled=os.getenv("ENABLE_TTS", "true").lower() in ("true", "1", "yes"),
            model=os.getenv("TTS_MODEL", "qwen"),
            language=os.getenv("TTS_LANGUAGE", "en"),
            voice_description=voice_desc if voice_desc else None,
            device=os.getenv("TTS_DEVICE", "cuda"),
        )


def generate_voice_description(context: str) -> str:
    """Get preset voice description for Qwen3-TTS VoiceDesign.
    
    Uses 12-dimension Qwen3-TTS format presets.
    
    Args:
        context: Either a semantic type (action, funny, clutch, etc.) or 
                caption style (gaming, genz, story_news, etc.)
        
    Returns:
        Detailed voice description string for Qwen3-TTS VoiceDesign
    """
    # Import here to avoid circular imports
    from ai_providers import ClipScore
    
    # Get the preset
    preset = ClipScore.CAPTION_STYLE_VOICE_MAP.get(
        context, 
        ClipScore.VOICE_PRESET_MAP.get(context, ClipScore.VOICE_PRESET_MAP["action"])
    )
    
    logging.debug(f"Using preset voice description for '{context}': {preset.split(chr(10))[0]}...")
    return preset


class QwenTTS:
    """Qwen3-TTS VoiceDesign wrapper for local inference.
    
    Uses a module-level cache to avoid reloading the model for each clip.
    """
    
    _instance: Optional["QwenTTS"] = None  # Singleton instance
    
    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig.from_env()
        self._model = None
        self._initialized = False
        self._sample_rate = 24000  # Default, will be updated after model load
    
    @classmethod
    def get_instance(cls, config: Optional[TTSConfig] = None) -> "QwenTTS":
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
            from qwen_tts import Qwen3TTSModel
            
            logging.info("Loading Qwen3-TTS VoiceDesign model...")
            logging.info(f"Device: {self.config.device}")
            
            # Determine device and dtype
            device_map = f"{self.config.device}:0" if self.config.device == "cuda" else "cpu"
            dtype = torch.bfloat16 if self.config.device == "cuda" else torch.float32
            
            # Check if flash_attention_2 is available
            try:
                import flash_attn
                attn_impl = "flash_attention_2"
                logging.info("Using FlashAttention 2 for better GPU performance")
            except ImportError:
                attn_impl = "eager"
                logging.info("FlashAttention not available, using eager attention")
            
            # Load the model with proper device/dtype settings
            self._model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                device_map=device_map,
                dtype=dtype,
                attn_implementation=attn_impl,
            )
            
            self._initialized = True
            logging.info("Qwen3-TTS model loaded successfully")
            
        except ImportError as e:
            logging.error(f"qwen-tts not installed: {e}")
            logging.error("Run: pip install qwen-tts")
            raise
        except Exception as e:
            logging.error(f"Failed to load Qwen3-TTS model: {e}")
            raise
    
    @property
    def sample_rate(self) -> int:
        """Get the model's sample rate."""
        return self._sample_rate
    
    def generate(
        self,
        text: str,
        output_path: Path,
        voice_description: Optional[str] = None,
    ) -> bool:
        """Generate speech from text.
        
        Args:
            text: Text to speak
            output_path: Path to save audio file (WAV)
            voice_description: Voice persona description (instruct)
            
        Returns:
            True if generation succeeded
        """
        if not self.config.enabled:
            logging.info("TTS disabled via ENABLE_TTS env var")
            return False
        
        self._ensure_initialized()
        
        # Use provided voice_description, config override, or default
        voice_desc = voice_description or self.config.voice_description
        if not voice_desc:
            voice_desc = "A clear, engaging voice with natural intonation"
        
        try:
            import scipy.io.wavfile as wav
            
            logging.info(f"Generating TTS: {text[:50]}...")
            
            # Use generate_voice_design API
            # Returns: (wavs: List[np.ndarray], sample_rate: int)
            wavs, sr = self._model.generate_voice_design(
                text=text,
                instruct=voice_desc,
                language=self.config.get_language_name(),
            )
            
            self._sample_rate = sr
            
            if wavs and len(wavs) > 0:
                audio = wavs[0]  # Get first (only) audio output
                
                # Normalize and convert to int16
                audio = (audio * 32767).astype('int16')
                
                # Save to file
                wav.write(str(output_path), sr, audio)
                
                logging.info(f"TTS audio saved: {output_path.name}")
                return True
            else:
                logging.error("TTS generation returned empty audio")
                return False
            
        except Exception as e:
            logging.error(f"TTS generation failed: {e}")
            return False
    
    def generate_for_captions(
        self,
        captions: List[dict],
        output_path: Path,
        detected_category: str = "action",
        caption_style: Optional[str] = None,
    ) -> Optional[Path]:
        """Generate TTS for a list of captions with proper timing alignment.
        
        Each caption's TTS audio is placed at the correct timestamp, with
        silence gaps between captions as needed.
        
        Args:
            captions: List of {"start": float, "end": float, "text": str}
            output_path: Output audio file path
            detected_category: Category for voice description generation (semantic type)
            caption_style: Caption style for voice description (preferred over category)
            
        Returns:
            Path to generated audio, or None if failed
        """
        if not captions:
            logging.warning("No captions to generate TTS for")
            return None
        
        self._ensure_initialized()
        
        # Get voice description - either from config override or preset
        if self.config.voice_description:
            voice_desc = self.config.voice_description
            logging.info(f"Using configured voice description: {voice_desc[:60]}...")
        else:
            # Use preset based on context
            context = caption_style if caption_style else detected_category
            
            logging.info(f"Generating voice for context: {context}")
            voice_desc = generate_voice_description(context)
            
            # Log first line for confirmation
            first_line = voice_desc.split('\n')[0] if '\n' in voice_desc else voice_desc[:60]
            logging.info(f"Voice: {first_line}...")
        
        try:
            import numpy as np
            import scipy.io.wavfile as wav
            
            audio_segments = []
            current_time = 0.0  # Track current position in seconds
            sample_rate = self._sample_rate
            
            logging.info(f"Generating TTS for {len(captions)} captions ({detected_category} mode)")
            
            # Story modes: captions timed to TTS (no stretching needed)
            # Regular modes: captions timed to visual events (stretching may be needed)
            is_story_mode = caption_style and caption_style.startswith("story_")
            
            for i, caption in enumerate(captions):
                text = caption.get("text", "").strip()
                start_time = caption.get("start", 0.0)
                end_time = caption.get("end", start_time + 3.0)
                caption_duration = end_time - start_time
                
                if not text:
                    continue
                
                # Strip action words (e.g., *chuckles*, *laughs*) that TTS reads literally
                import re
                text = re.sub(r'\*[^*]+\*', '', text).strip()
                if not text:
                    continue
                
                # Calculate silence needed before this caption
                silence_duration = max(0, start_time - current_time)
                if silence_duration > 0:
                    silence_samples = int(silence_duration * sample_rate)
                    silence = np.zeros(silence_samples, dtype=np.float32)
                    audio_segments.append(silence)
                    current_time += silence_duration
                
                # Generate TTS for this caption
                try:
                    wavs, sr = self._model.generate_voice_design(
                        text=text,
                        instruct=voice_desc,
                        language=self.config.get_language_name(),
                    )
                    
                    sample_rate = sr  # Update sample rate
                    self._sample_rate = sr
                    
                    if wavs and len(wavs) > 0:
                        audio = wavs[0].astype(np.float32)
                        tts_duration = len(audio) / sample_rate
                        
                        # Time-stretch if TTS is longer than caption window
                        # SKIP stretching for story modes (caption duration = TTS duration by design)
                        # Use pitch-preserving stretching (NOT resampling which causes chipmunk)
                        if not is_story_mode and tts_duration > caption_duration and caption_duration > 0.5:
                            stretch_ratio = tts_duration / caption_duration
                            
                            # Only stretch if within reasonable bounds (up to 2x speedup)
                            # Beyond 2x speedup, audio becomes unintelligible
                            if stretch_ratio <= 2.0:
                                try:
                                    import librosa
                                    # librosa.effects.time_stretch: rate > 1 = faster
                                    audio = librosa.effects.time_stretch(audio, rate=stretch_ratio)
                                    new_duration = len(audio) / sample_rate
                                    logging.debug(f"Caption {i}: stretched {tts_duration:.2f}s → {new_duration:.2f}s (target: {caption_duration:.2f}s)")
                                except ImportError:
                                    logging.warning("librosa not available for pitch-preserving time stretch")
                            else:
                                logging.warning(f"Caption {i}: TTS too long ({tts_duration:.1f}s vs {caption_duration:.1f}s), skipping stretch to preserve quality")
                        
                        audio_segments.append(audio)
                        current_time += len(audio) / sample_rate
                    else:
                        # Add small silence if generation failed
                        silence_samples = int(0.5 * sample_rate)
                        audio_segments.append(np.zeros(silence_samples, dtype=np.float32))
                        current_time += 0.5
                    
                except Exception as e:
                    logging.warning(f"TTS failed for caption {i}: {e}")
                    # Add small silence instead
                    silence_samples = int(0.5 * sample_rate)
                    audio_segments.append(np.zeros(silence_samples, dtype=np.float32))
                    current_time += 0.5
            
            if not audio_segments:
                return None
            
            # Concatenate all segments
            final_audio = np.concatenate(audio_segments)
            
            # Normalize and convert to int16 for WAV file
            final_audio = (final_audio * 32767).astype(np.int16)
            
            wav.write(str(output_path), sample_rate, final_audio)
            
            total_duration = len(final_audio) / sample_rate
            logging.info(f"TTS audio generated: {total_duration:.1f}s for {len(captions)} captions")
            
            return output_path
            
        except Exception as e:
            logging.error(f"Caption TTS generation failed: {e}")
            return None


# Backward compatibility alias
ChatterBoxTTS = QwenTTS


def is_tts_enabled() -> bool:
    """Check if TTS is enabled via environment."""
    return os.getenv("ENABLE_TTS", "true").lower() in ("true", "1", "yes")


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
        detected_category: Category for voice description
        config: Optional TTS config override
        
    Returns:
        Path to generated audio, or None if failed
    """
    if not is_tts_enabled():
        return None
    
    cfg = config or TTSConfig.from_env()
    
    # Generate voice description if not set
    voice_desc = cfg.voice_description
    if not voice_desc:
        voice_desc = generate_voice_description(detected_category)
    
    # Use singleton to keep model loaded across multiple calls
    tts = QwenTTS.get_instance(cfg)
    success = tts.generate(text, output_path, voice_description=voice_desc)
    
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
    """Pre-download the Qwen3-TTS model for offline use."""
    logging.info("Downloading Qwen3-TTS VoiceDesign model...")
    
    try:
        import torch
        from qwen_tts import Qwen3TTSModel
        
        # Check for flash_attention
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            logging.info("FlashAttention 2 available")
        except ImportError:
            attn_impl = "eager"
            logging.info("FlashAttention not available")
        
        # This triggers model download
        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            attn_implementation=attn_impl,
        )
        
        logging.info("Model downloaded successfully!")
        
        # Quick test
        wavs, sr = model.generate_voice_design(
            text="Test successful.",
            instruct="A clear, friendly voice",
            language="English"
        )
        logging.info(f"Sample rate: {sr}")
        if wavs:
            logging.info(f"Test generation shape: {wavs[0].shape}")
        
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
        print("This will download the Qwen3-TTS model for local use.")
