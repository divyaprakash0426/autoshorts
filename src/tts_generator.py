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


# Slang/abbreviation expansions for TTS clarity
# TTS models often can't pronounce internet slang correctly
SLANG_EXPANSIONS = {
    # GenZ slang
    "fr fr": "for real for real",
    "fr,fr": "for real for real", 
    "fr, fr": "for real for real",
    "frfr": "for real for real",
    "fr": "for real",
    "rn": "right now",
    "ngl": "not gonna lie",
    "idk": "I don't know",
    "idc": "I don't care",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "tbh": "to be honest",
    "tho": "though",
    "thru": "through",
    "u": "you",
    "ur": "your",
    "r": "are",
    "w/": "with",
    "w/o": "without",
    "bc": "because",
    "b4": "before",
    "2day": "today",
    "2nite": "tonight",
    "2morrow": "tomorrow",
    "smh": "shaking my head",
    "omg": "oh my god",
    "lol": "laughing out loud",
    "lmao": "laughing my ass off",
    "rofl": "rolling on the floor laughing",
    "brb": "be right back",
    "btw": "by the way",
    "fyi": "for your information",
    "gg": "good game",
    "ez": "easy",
    "pog": "play of the game",
    "poggers": "play of the game",
    "goat": "greatest of all time",
    "goated": "greatest of all time",
    "bussin": "bussin'",
    "finna": "fixing to",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "kinda": "kind of",
    "sorta": "sort of",
    "prolly": "probably",
    "aight": "alright",
    "ight": "alright",
    "yall": "y'all",
    "ya'll": "y'all",
    # Gaming terms
    "1v1": "one v one",
    "2v2": "two v two", 
    "3v3": "three v three",
    "5v5": "five v five",
    "1hp": "one HP",
    "hp": "H P",
    "dps": "D P S",
    "aoe": "A O E",
    "fps": "F P S",
    "rpg": "R P G",
    "mmo": "M M O",
    "pvp": "P V P",
    "pve": "P V E",
    "npc": "N P C",
    "op": "O P",
    "nerf": "nerf",
    "buff": "buff",
    # Punctuation that TTS might read literally
    "...": ", ",
    "..": ", ",
    "--": ", ",
    "—": ", ",
    ",,": ",",
}


def preprocess_text_for_tts(text: str) -> str:
    """Expand slang and abbreviations for better TTS pronunciation.
    
    Args:
        text: Raw caption text
        
    Returns:
        Text with slang expanded for TTS clarity
    """
    import re
    
    result = text
    
    # First, handle punctuation replacements (no word boundaries)
    punctuation_replacements = {
        "...": ", ",
        "..": ", ",
        "--": ", ",
        "—": ", ",
        ",,": ",",
    }
    for punct, replacement in punctuation_replacements.items():
        result = result.replace(punct, replacement)
    
    # Sort slang by length (longest first) to avoid partial replacements
    # e.g., "fr fr" should be replaced before "fr"
    word_slang = {k: v for k, v in SLANG_EXPANSIONS.items() if k not in punctuation_replacements}
    sorted_slang = sorted(word_slang.keys(), key=len, reverse=True)
    
    for slang in sorted_slang:
        expansion = word_slang[slang]
        # Case-insensitive word boundary replacement
        # Use word boundaries to avoid replacing parts of words
        pattern = r'\b' + re.escape(slang) + r'\b'
        result = re.sub(pattern, expansion, result, flags=re.IGNORECASE)
    
    # Clean up multiple spaces
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result


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
    
    # Get the preset (exact style match first)
    preset = ClipScore.CAPTION_STYLE_VOICE_MAP.get(context)
    if not preset:
        style_fallbacks = {
            "podcast_": "minimal",
            "sports_": "story_news",
            "educational_": "dramatic",
            "news_": "story_news",
            "vlog_": "genz",
            "music_": "gaming",
            "interview_": "dramatic",
            "comedy_": "funny",
            "cooking_": "minimal",
            "fitness_": "gaming",
            "entertainment_": "dramatic",
            "tv_": "dramatic",
            "documentary_": "story_dramatic",
            "esports_": "story_news",
        }
        for prefix, fallback_style in style_fallbacks.items():
            if context.startswith(prefix):
                preset = ClipScore.CAPTION_STYLE_VOICE_MAP.get(fallback_style)
                break
    if not preset:
        preset = ClipScore.VOICE_PRESET_MAP.get(context, ClipScore.VOICE_PRESET_MAP["action"])
    
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
            # Use local_files_only=True to avoid network calls when model is cached
            try:
                self._model = Qwen3TTSModel.from_pretrained(
                    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                    device_map=device_map,
                    dtype=dtype,
                    attn_implementation=attn_impl,
                    local_files_only=True,  # Use cached model, no network
                )
            except Exception as e:
                # Fallback to network if local cache doesn't exist
                logging.warning(f"Local model not found, downloading: {e}")
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
            
            # Preprocess text for TTS (expand slang, abbreviations)
            processed_text = preprocess_text_for_tts(text)
            
            logging.info(f"Generating TTS: {processed_text[:50]}...")
            
            # Use generate_voice_design API
            # Returns: (wavs: List[np.ndarray], sample_rate: int)
            wavs, sr = self._model.generate_voice_design(
                text=processed_text,
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
                
                # Expand slang/abbreviations for better TTS pronunciation
                text = preprocess_text_for_tts(text)
                
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
                        
                        # NOTE: Voice stretching removed - it degrades audio quality
                        # Instead, we extend the video duration when TTS is longer
                        # (handled in mix_audio_with_video via tpad or amix=longest)
                        if tts_duration > caption_duration:
                            logging.debug(f"Caption {i}: TTS ({tts_duration:.1f}s) > caption window ({caption_duration:.1f}s), video will be extended")
                        
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


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _video_has_audio_stream(video_path: Path) -> bool:
    """Check if video file has an audio stream using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        # If output contains "audio", there's an audio stream
        return "audio" in result.stdout.strip().lower()
    except Exception:
        return False


def extend_video_with_tpad(
    video_path: Path,
    output_path: Path,
    target_duration: float,
) -> bool:
    """Extend video duration by freezing the last frame (tpad filter).
    
    Args:
        video_path: Input video
        output_path: Output path for extended video
        target_duration: Target duration in seconds
        
    Returns:
        True if successful
    """
    video_duration = get_video_duration(video_path)
    if video_duration <= 0:
        logging.error("Could not determine video duration for tpad")
        return False
    
    extend_by = target_duration - video_duration
    if extend_by <= 0:
        # No extension needed
        return False
    
    logging.info(f"Extending video by {extend_by:.1f}s using tpad (freeze last frame)")
    
    # tpad filter: stop=-1 means freeze last frame, stop_duration is how long
    filter_complex = f"[0:v]tpad=stop=-1:stop_duration={extend_by:.3f}[vout]"
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "0:a?",  # Copy audio if present
        "-c:v", "hevc_nvenc",
        "-preset", "fast",
        "-c:a", "copy",
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logging.error(f"tpad extension failed: {result.stderr[:200]}")
            return False
        return output_path.exists()
    except Exception as e:
        logging.error(f"tpad extension error: {e}")
        return False


@dataclass
class RenderMeta:
    """Metadata needed to re-render a clip at different duration."""
    source_path: Path
    start_time: float
    original_duration: float
    output_width: int
    output_height: int
    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int
    bg_width: int
    bg_height: int
    is_vertical_bg: bool


def rerender_video_longer(
    render_meta: RenderMeta,
    target_duration: float,
    output_path: Path,
) -> bool:
    """Re-render video from source with longer duration for TTS.
    
    Uses simple FFmpeg re-encoding (not full GPU pipeline) for speed.
    Extends beyond original scene boundaries if needed.
    
    Args:
        render_meta: Original render parameters
        target_duration: Target duration in seconds
        output_path: Output path for re-rendered video
        
    Returns:
        True if successful
    """
    logging.info(f"Re-rendering video: {render_meta.original_duration:.1f}s → {target_duration:.1f}s")
    
    # Calculate crop filter for the aspect ratio
    crop_filter = f"crop={render_meta.crop_w}:{render_meta.crop_h}:{render_meta.crop_x}:{render_meta.crop_y}"
    scale_filter = f"scale={render_meta.output_width}:{render_meta.output_height}"
    
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{render_meta.start_time:.3f}",
        "-t", f"{target_duration:.3f}",
        "-i", str(render_meta.source_path),
        "-vf", f"{crop_filter},{scale_filter}",
        "-c:v", "hevc_nvenc",
        "-preset", "fast",
        "-c:a", "aac",
        "-b:a", "192k",
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logging.error(f"Re-render failed: {result.stderr[:200]}")
            return False
        return output_path.exists()
    except Exception as e:
        logging.error(f"Re-render error: {e}")
        return False


def mix_audio_with_video(
    video_path: Path,
    voiceover_path: Path,
    output_path: Path,
    game_audio_volume: float = 0.3,
    voiceover_volume: float = 1.0,
    render_meta: Optional[RenderMeta] = None,
) -> bool:
    """Mix voiceover audio with video's existing audio.
    
    If TTS is longer than video:
    - Small overage (≤3s): Extend video using tpad (freeze last frame)
    - Large overage (>3s): Re-render from source if render_meta provided,
      otherwise use tpad as fallback
    
    Args:
        video_path: Input video with game audio
        voiceover_path: Voiceover WAV file
        output_path: Output video with mixed audio
        game_audio_volume: Volume multiplier for game audio (0.0-1.0)
        voiceover_volume: Volume multiplier for voiceover (0.0-1.0)
        render_meta: Optional render metadata for re-rendering longer clips
        
    Returns:
        True if successful
    """
    # Check durations to handle TTS longer than video
    video_duration = get_video_duration(video_path)
    audio_duration = get_audio_duration(voiceover_path)
    
    duration_diff = audio_duration - video_duration
    working_video = video_path
    extended_video_tmp = None
    rerendered_video_tmp = None
    
    if duration_diff > 0:
        logging.info(f"TTS ({audio_duration:.1f}s) longer than video ({video_duration:.1f}s) by {duration_diff:.1f}s")
        
        if duration_diff <= 3.0:
            # Small overage: extend video with tpad (freeze last frame)
            extended_video_tmp = video_path.with_stem(video_path.stem + "_extended_tmp")
            if extend_video_with_tpad(video_path, extended_video_tmp, audio_duration + 0.5):
                working_video = extended_video_tmp
                logging.info(f"Video extended to {audio_duration + 0.5:.1f}s with frozen last frame")
            else:
                logging.warning("tpad extension failed, TTS may be truncated")
        else:
            # Large overage: re-render from source if possible
            if render_meta is not None:
                rerendered_video_tmp = video_path.with_stem(video_path.stem + "_rerendered_tmp")
                target_duration = audio_duration + 1.0  # Add 1s buffer
                
                if rerender_video_longer(render_meta, target_duration, rerendered_video_tmp):
                    working_video = rerendered_video_tmp
                    logging.info(f"Video re-rendered to {target_duration:.1f}s from source")
                else:
                    # Fallback to tpad if re-render fails
                    logging.warning("Re-render failed, falling back to tpad")
                    extended_video_tmp = video_path.with_stem(video_path.stem + "_extended_tmp")
                    if extend_video_with_tpad(video_path, extended_video_tmp, audio_duration + 0.5):
                        working_video = extended_video_tmp
            else:
                # No render_meta, use tpad as fallback
                logging.info(f"Large TTS overage but no render_meta, using tpad fallback")
                extended_video_tmp = video_path.with_stem(video_path.stem + "_extended_tmp")
                if extend_video_with_tpad(video_path, extended_video_tmp, audio_duration + 0.5):
                    working_video = extended_video_tmp
    
    # Check if video has audio stream BEFORE deciding which command to use
    # PyCaps outputs typically have NO audio stream
    has_audio = _video_has_audio_stream(working_video)
    
    if has_audio:
        # FFmpeg command to mix audio tracks
        # - Use duration=longest to preserve TTS when it's longer than video
        filter_complex = (
            f"[0:a]volume={game_audio_volume}[game];"
            f"[1:a]volume={voiceover_volume}[voice];"
            "[game][voice]amix=inputs=2:duration=longest:dropout_transition=2[aout]"
        )
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(working_video),
            "-i", str(voiceover_path),
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",  # Copy video stream (already encoded)
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_path)
        ]
    else:
        # Video has no audio stream (e.g., PyCaps output)
        # Simply add TTS as the only audio track
        logging.info("Video has no audio stream, adding TTS as only audio track")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(working_video),
            "-i", str(voiceover_path),
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
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
        
        # Cleanup temp files
        for tmp_file in [extended_video_tmp, rerendered_video_tmp]:
            if tmp_file and tmp_file.exists():
                try:
                    tmp_file.unlink()
                except Exception:
                    pass
        
        if result.returncode != 0:
            logging.error(f"FFmpeg audio mix failed: {result.stderr[:500]}")
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
