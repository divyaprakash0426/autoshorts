"""Subtitle generation using Whisper and PyCaps.

This module provides functionality to:
1. Transcribe video audio using OpenAI Whisper
2. Apply animated subtitles using PyCaps with custom styling
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclass
class SubtitleConfig:
    """Configuration for subtitle styling."""
    
    font_family: str = "Bangers"
    font_size: int = 48
    text_color: str = "#FFFFFF"
    highlight_color: str = "#00ff88"  # Neon green for active word
    shadow_color: str = "#000000"
    shadow_offset: int = 2
    position: str = "bottom"  # "top", "center", "bottom"
    margin_bottom: int = 50
    
    @classmethod
    def from_env(cls) -> "SubtitleConfig":
        """Create config from environment variables."""
        return cls(
            font_family=os.getenv("SUBTITLE_FONT", "Bangers"),
            font_size=int(os.getenv("SUBTITLE_FONT_SIZE", "48")),
            text_color=os.getenv("SUBTITLE_TEXT_COLOR", "#FFFFFF"),
            highlight_color=os.getenv("SUBTITLE_HIGHLIGHT_COLOR", "#00ff88"),
            shadow_color=os.getenv("SUBTITLE_SHADOW_COLOR", "#000000"),
            shadow_offset=int(os.getenv("SUBTITLE_SHADOW_OFFSET", "2")),
            position=os.getenv("SUBTITLE_POSITION", "bottom"),
            margin_bottom=int(os.getenv("SUBTITLE_MARGIN_BOTTOM", "50")),
        )


def is_subtitles_enabled() -> bool:
    """Check if subtitle generation is enabled via env."""
    return os.getenv("ENABLE_SUBTITLES", "true").lower() in ("true", "1", "yes")


def get_whisper_model() -> str:
    """Get the configured Whisper model name."""
    return os.getenv("WHISPER_MODEL", "medium")


def transcribe_audio(video_path: Path, output_srt: Optional[Path] = None) -> Path:
    """Transcribe video audio using Whisper.
    
    Args:
        video_path: Path to video file
        output_srt: Optional path for SRT output. If None, creates alongside video.
        
    Returns:
        Path to generated SRT file
    """
    if output_srt is None:
        output_srt = video_path.with_suffix(".srt")
    
    model_name = get_whisper_model()
    
    logging.info(f"Transcribing audio with Whisper ({model_name})...")
    
    try:
        import whisper
        
        # Load model
        model = whisper.load_model(model_name)
        
        # Transcribe
        result = model.transcribe(
            str(video_path),
            task="transcribe",
            verbose=False,
            word_timestamps=True  # Enable word-level timestamps for PyCaps
        )
        
        # Generate SRT content
        srt_content = _generate_srt(result)
        
        # Write SRT file
        with open(output_srt, "w", encoding="utf-8") as f:
            f.write(srt_content)
        
        logging.info(f"Transcription saved to: {output_srt}")
        

        
        return output_srt
        
    except ImportError:
        logging.error("whisper not installed. Run: pip install openai-whisper")
        raise
    except Exception as e:
        logging.error(f"Transcription failed: {e}")
        raise


def _generate_srt(whisper_result: dict) -> str:
    """Convert Whisper result to SRT format."""
    segments = whisper_result.get("segments", [])
    srt_lines = []
    
    for i, segment in enumerate(segments, 1):
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        text = segment.get("text", "").strip()
        
        if not text:
            continue
        
        start_ts = _format_timestamp(start)
        end_ts = _format_timestamp(end)
        
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_ts} --> {end_ts}")
        srt_lines.append(text)
        srt_lines.append("")
    
    return "\n".join(srt_lines)


def _format_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def apply_pycaps_subtitles(
    video_path: Path,
    srt_path: Path,
    output_path: Path,
    config: Optional[SubtitleConfig] = None,
    extra_css: str = "",
    word_lists: dict = None
) -> bool:
    """Apply animated subtitles using PyCaps (TemplateLoader API).
    
    Args:
        video_path: Path to input video
        srt_path: Path to SRT subtitle file
        output_path: Path for output video with subtitles
        config: Subtitle styling configuration
        extra_css: Additional CSS for tagging
        word_lists: Dictionary of word lists for tagging
        
    Returns:
        True if successful
    """
    if config is None:
        config = SubtitleConfig.from_env()
    
    logging.info(f"Applying PyCaps subtitles to: {video_path.name}")
    
    # Run PyCaps in a separate process to avoid Playwright Sync API conflict with asyncio
    # and to ensure a clean environment.
    import multiprocessing
    
    # We need to pass strings, not Path objects, to be safe across processes
    ctx = multiprocessing.get_context("spawn")  # Use spawn for better compatibility
    p = ctx.Process(
        target=_run_pycaps_worker,
        args=(str(video_path), str(srt_path), str(output_path), config, extra_css, word_lists)
    )
    p.start()
    p.join()
    
    if p.exitcode == 0 and (output_path.exists() or _check_for_fallback_output(video_path, output_path)):
        logging.info(f"PyCaps subtitled video saved to: {output_path}")
        return True
    
    logging.warning("PyCaps process failed or produced no output. Falling back to FFmpeg.")
    return _apply_ffmpeg_subtitles(video_path, srt_path, output_path, config)


def _check_for_fallback_output(video_path: Path, output_path: Path) -> bool:
    """Check if PyCaps saved to a default filename and move it."""
    import time
    
    # Check current directory
    cwd = Path.cwd()
    recent_outputs = sorted(
        cwd.glob("output_*.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    for recent in recent_outputs[:3]:
        if time.time() - recent.stat().st_mtime < 60:
            import shutil
            shutil.move(str(recent), str(output_path))
            return True
            
    # Check video parent directory
    parent_outputs = sorted(
        video_path.parent.glob("output_*.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    for recent in parent_outputs[:3]:
        if time.time() - recent.stat().st_mtime < 60:
            import shutil
            shutil.move(str(recent), str(output_path))
            return True
            
    return False


def _run_pycaps_worker(video_path_str: str, srt_path_str: str, output_path_str: str, config: SubtitleConfig, extra_css: str = "", word_lists: dict = None):
    """Worker function to run PyCaps in a separate process."""
    # Re-import necessary modules in the spawned process
    import sys
    import os
    
    try:
        from pycaps import CapsPipelineBuilder
        from pycaps.transcriber import AudioTranscriber
        from pycaps.common import Document, Segment, Line, Word, TimeFragment
        from pycaps.tag import SemanticTagger
        from pycaps.template import TemplateLoader
        
        # Define SRTTranscriber locally to ensure it's available in the worker process
        class SRTTranscriber(AudioTranscriber):
            def __init__(self, srt_path):
                self.srt_path = srt_path
                
            def transcribe(self, audio_path: str) -> Document:
                document = Document()
                if not os.path.exists(self.srt_path):
                    print(f"SRT file not found: {self.srt_path}")
                    return document
                    
                try:
                    with open(self.srt_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Normalize newlines
                    content = content.replace('\r\n', '\n').replace('\r', '\n')
                    
                    def parse_time(t_str):
                        t_str = t_str.strip().replace(',', '.')
                        parts = t_str.split(':')
                        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

                    import re
                    # Regex to find blocks: ID, Time, Text
                    # Matches:
                    # 1
                    # 00:00:01,000 --> 00:00:02,000
                    # Text content
                    block_pattern = re.compile(r'(\d+)\s*\n(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*\n(.*?)(?=\n\s*\d+\s*\n|\Z)', re.DOTALL)
                    
                    matches = list(block_pattern.finditer(content))
                    print(f"Parsing SRT: Found {len(matches)} blocks in {self.srt_path}")
                    
                    for match in matches:
                        idx = match.group(1)
                        start_str = match.group(2)
                        end_str = match.group(3)
                        text = match.group(4).strip()
                        
                        if not text: continue
                        
                        start = parse_time(start_str)
                        end = parse_time(end_str)
                        
                        segment_time = TimeFragment(start=start, end=end)
                        segment = Segment(time=segment_time)
                        line = Line(time=segment_time)
                        segment.lines.add(line)
                        
                        # Pre-process words to merge detached emojis
                        # "BETRAYAL" + "🗡️💔" -> "BETRAYAL 🗡️💔"
                        # This prevents emojis from getting their own time slice and lingering
                        raw_words = text.split()
                        words = []
                        if raw_words:
                            words.append(raw_words[0])
                            for w in raw_words[1:]:
                                # Check if word is likely just emojis/symbols (non-alphanumeric and high unicode)
                                # isalnum() handles "café" correctly as True, so we skip merging it.
                                # Emojis "🗡️" returns isalnum() -> False.
                                is_symbol_or_emoji = (not w.isalnum()) and any(ord(c) > 2000 for c in w)
                                
                                if is_symbol_or_emoji and words:
                                    words[-1] += " " + w
                                else:
                                    words.append(w)
                        
                        if not words: continue
                        
                        # Interpolate word timings
                        duration = end - start
                        word_duration = duration / len(words)
                        
                        from pycaps.common import Tag
                        
                        current_time = start
                        for i, w_text in enumerate(words):
                            w_start = current_time
                            w_end = current_time + word_duration
                            # Ensure no overlap issues or zero duration
                            if w_end > end: w_end = end
                            
                            # Add space to word text to prevent "SOCLOSE" issues
                            final_text = w_text + " "
                            
                            word = Word(
                                text=final_text, 
                                time=TimeFragment(start=w_start, end=w_end)
                            )
                            
                            # Explicitly tag key words to prevent merging
                            if i == len(words) - 1:
                                word.structure_tags.add(Tag(name="last-word-in-segment"))
                                word.structure_tags.add(Tag(name="last-word-in-line"))
                            if i == 0:
                                word.structure_tags.add(Tag(name="first-word-in-segment"))
                                word.structure_tags.add(Tag(name="first-word-in-line"))
                                
                            line.words.add(word)
                            current_time += word_duration
                            
                        # Explicitly tag the line
                        line.structure_tags.add(Tag(name="last-line-in-segment"))
                        segment.structure_tags.add(Tag(name="segment"))
                            
                        document.segments.add(segment)
                        
                    print(f"SRTTranscriber populated {len(document.segments)} segments.")
                    
                except Exception as e:
                    print(f"Error parsing SRT: {e}")
                    import traceback
                    traceback.print_exc()
                
                return document

        # Get template from environment or use default
        template_name = os.getenv("PYCAPS_TEMPLATE", "hype")
        
        # Use native PyCaps template styling (no custom CSS overrides)
        # The template handles fonts, colors, animations automatically
        print(f"Loading PyCaps template: {template_name}")
        builder = TemplateLoader(template_name).with_input_video(video_path_str).load(should_build_pipeline=False)
        
        if hasattr(builder, "_caps_pipeline"):
            print("Force-clearing PyCaps segment splitters to enforce strict SRT timing.")
            if hasattr(builder._caps_pipeline, "_segment_splitters"):
                 builder._caps_pipeline._segment_splitters = []
                 print("Cleared splitters.")

        # EXPLICITLY tell PyCaps to use our SRT file instead of transcribing audio
        # This fixes the issue where it ignores AI captions and transcribes game audio
        print(f"Using custom SRTTranscriber with: {srt_path_str}")
        builder = builder.with_custom_audio_transcriber(SRTTranscriber(str(srt_path_str)))

        builder = (
            builder
            .with_output_video(output_path_str)
        )
        
        # Only add extra_css if AI tagging provides it (for word highlighting)
        if extra_css and extra_css.strip():
            builder = builder.add_css_content(extra_css)
        
        # Configure Semantic Tagger for AI tags
        if word_lists:
            tagger = SemanticTagger()
            from pycaps.common import Tag
            for tag_name, w_list in word_lists.items():
                if w_list:
                    tagger.add_wordlist_rule(Tag(name=tag_name), w_list)
            builder.with_semantic_tagger(tagger)
        
        # Build pipeline
        pipeline = builder.build()
        
        pipeline.run()
        
    except Exception as e:
        # Use print because logging might not be configured in child process
        print(f"PyCaps worker failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _apply_ffmpeg_subtitles(
    video_path: Path,
    srt_path: Path,
    output_path: Path,
    config: SubtitleConfig
) -> bool:
    """Fallback: Apply subtitles using FFmpeg (less fancy, but reliable).
    
    Uses the ASS subtitle filter for styled text with the Bangers font.
    """
    logging.info("Using FFmpeg subtitle burn-in (fallback mode)")
    
    # Build FFmpeg subtitle filter with styling
    # Note: FFmpeg uses a different syntax for fonts and colors
    font_style = f"FontName={config.font_family},FontSize={config.font_size}"
    color_hex = config.text_color.lstrip("#")
    outline_hex = config.shadow_color.lstrip("#")
    highlight_hex = config.highlight_color.lstrip("#")
    
    # Convert SRT to styled ASS for better control
    ass_path = srt_path.with_suffix(".ass")
    _convert_srt_to_ass(srt_path, ass_path, config)
    
    # Use the ASS file for subtitle burn-in
    subtitle_filter = f"ass='{ass_path}'"
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", subtitle_filter,
        "-c:v", "hevc_nvenc",  # Use GPU encoding
        "-preset", "slow",
        "-cq", "23",
        "-c:a", "copy",
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logging.error(f"FFmpeg subtitle burn-in failed: {result.stderr[:200]}")
        return False
    
    # Clean up temp ASS file
    try:
        ass_path.unlink()
    except Exception:
        pass
    
    return output_path.exists()


def _convert_srt_to_ass(srt_path: Path, ass_path: Path, config: SubtitleConfig) -> None:
    """Convert SRT to ASS format with custom styling."""
    
    # Read SRT content
    with open(srt_path, "r", encoding="utf-8") as f:
        srt_content = f.read()
    
    # Parse SRT
    segments = []
    current_segment = {}
    
    for line in srt_content.strip().split("\n"):
        line = line.strip()
        
        if not line:
            if current_segment:
                segments.append(current_segment)
                current_segment = {}
        elif "-->" in line:
            parts = line.split(" --> ")
            current_segment["start"] = _srt_to_ass_time(parts[0])
            current_segment["end"] = _srt_to_ass_time(parts[1])
        elif not line.isdigit():
            current_segment["text"] = current_segment.get("text", "") + line + " "
    
    if current_segment:
        segments.append(current_segment)
    
    # Generate ASS content
    text_color = _hex_to_ass_color(config.text_color)
    outline_color = _hex_to_ass_color(config.shadow_color)
    
    ass_header = f"""[Script Info]
Title: Generated Subtitles
ScriptType: v4.00+
WrapStyle: 0
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{config.font_family},{config.font_size},{text_color},&H000000FF,{outline_color},&H00000000,1,0,0,0,100,100,0,0,1,{config.shadow_offset},0,2,10,10,{config.margin_bottom},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    ass_lines = [ass_header]
    
    for seg in segments:
        text = seg.get("text", "").strip()
        if text:
            # Use uppercase for impact (common in short-form content)
            text = text.upper()
            ass_lines.append(
                f"Dialogue: 0,{seg['start']},{seg['end']},Default,,0,0,0,,{text}"
            )
    
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ass_lines))


def _srt_to_ass_time(srt_time: str) -> str:
    """Convert SRT timestamp to ASS format."""
    # SRT: 00:00:01,500 -> ASS: 0:00:01.50
    srt_time = srt_time.strip().replace(",", ".")
    parts = srt_time.split(":")
    hours = int(parts[0])
    minutes = parts[1]
    seconds = parts[2][:5]  # Truncate to 2 decimal places
    return f"{hours}:{minutes}:{seconds}"


def _hex_to_ass_color(hex_color: str) -> str:
    """Convert hex color to ASS format (&HAABBGGRR)."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        r = hex_color[0:2]
        g = hex_color[2:4]
        b = hex_color[4:6]
        # ASS uses BGR order with alpha prefix
        return f"&H00{b}{g}{r}"
    return "&H00FFFFFF"


def get_subtitle_mode() -> str:
    """Get the configured subtitle mode.
    
    Returns:
        One of: "speech", "ai_captions", "none"
    """
    return os.getenv("SUBTITLE_MODE", "ai_captions").lower()


def get_caption_style() -> str:
    """Get the configured AI caption style.
    
    Returns:
        One of: "gaming", "dramatic", "funny", "minimal"
    """
    return os.getenv("CAPTION_STYLE", "gaming").lower()


def generate_subtitles(
    video_path: Path, 
    output_path: Optional[Path] = None,
    detected_category: Optional[str] = None
) -> Optional[Path]:
    """Full subtitle pipeline with mode selection.
    
    Modes (set via SUBTITLE_MODE env var):
    - "speech": Use Whisper to transcribe voice/commentary
    - "ai_captions": Use AI to generate contextual captions (for gameplay without voice)
    - "none": Skip subtitle generation entirely
    
    Args:
        video_path: Path to input video
        output_path: Optional output path. If None, creates alongside input.
        detected_category: Optional category from AI analysis ("action", "funny", "highlight")
                          Used when CAPTION_STYLE=auto to match style to content.
        
    Returns:
        Path to subtitled video, or None if subtitles are disabled/failed
    """
    if not is_subtitles_enabled():
        logging.info("Subtitles disabled via ENABLE_SUBTITLES env var")
        return None
    
    mode = get_subtitle_mode()
    
    if mode == "none":
        logging.info("Subtitle mode set to 'none'. Skipping.")
        return None
    
    if output_path is None:
        output_path = video_path.with_stem(video_path.stem + "_subtitled")
    
    config = SubtitleConfig.from_env()
    srt_path = video_path.with_suffix(".srt")
    
    # Initialize tagging data
    tag_css = ""
    word_lists = {}
    
    try:
        if mode == "speech":
            # Whisper transcription mode
            logging.info("Using speech transcription mode (Whisper)")
            srt_path = transcribe_audio(video_path, srt_path)
            
        elif mode == "ai_captions":
            # AI-generated captions mode
            logging.info("Using AI caption generation mode")
            
            from ai_providers import (
                generate_ai_captions, 
                captions_to_srt, 
                batch_tag_captions, 
                apply_tags_to_pycaps, 
                add_emojis_to_caption,
                Caption
            )
            
            # Determine caption style
            style = get_caption_style()
            
            # Auto style matching based on detected category
            if style == "auto" and detected_category:
                style_map = {
                    "action": "gaming",
                    "funny": "funny",
                    "highlight": "dramatic",
                }
                style = style_map.get(detected_category, "gaming")
                logging.info(f"Auto-matched caption style: {style} (from category: {detected_category})")
            elif style == "auto":
                style = "gaming"  # Default fallback
                
            max_captions = int(os.getenv("MAX_CAPTIONS", "8"))
            
            result = generate_ai_captions(video_path, style=style, max_captions=max_captions)
            
            if not result.success or not result.captions:
                logging.warning(f"AI caption generation failed: {result.error}")
                logging.info("Falling back to speech mode...")
                
                # Fallback to speech mode
                try:
                    srt_path = transcribe_audio(video_path, srt_path)
                except Exception as e:
                    logging.error(f"Speech fallback also failed: {e}")
                    return None
            else:
                # Enhance captions with AI-powered emojis and tagging
                category = detected_category or style
                
                # 1. Generate tags and emojis
                tag_results = batch_tag_captions(result.captions, category=category)
                
                # 2. Add emojis to text
                add_emojis = os.getenv("ENABLE_CAPTION_EMOJIS", "true").lower() in ("true", "1", "yes")
                enhanced_captions = []
                for cap in result.captions:
                    tag_res = tag_results.get(cap.text)
                    if tag_res and add_emojis:
                        new_text = add_emojis_to_caption(cap.text, tag_res)
                        enhanced_captions.append(Caption(cap.start_time, cap.end_time, new_text, cap.style))
                    else:
                        enhanced_captions.append(cap)
                
                # 3. Generate PyCaps Tagging Data
                highlight_color = config.highlight_color or "#00ff88"
                tag_css, word_lists = apply_tags_to_pycaps(result.captions, tag_results, highlight_color)
                
                # Convert enhanced AI captions to SRT
                captions_to_srt(enhanced_captions, srt_path)
                logging.info(f"Generated {len(enhanced_captions)} AI captions ({style} style)")
        else:
            logging.warning(f"Unknown subtitle mode: {mode}. Using ai_captions.")
            return generate_subtitles(video_path, output_path, detected_category)
        
        # Check if we got any subtitles
        if not srt_path.exists() or srt_path.stat().st_size == 0:
            logging.warning("No subtitles generated (empty SRT)")
            return None
        
        # Apply subtitles using PyCaps (or FFmpeg fallback)
        success = apply_pycaps_subtitles(video_path, srt_path, output_path, config, extra_css=tag_css, word_lists=word_lists)
        
        if not success:
            logging.error("Subtitle burn-in failed")
            return None
        
        # --- TTS Voiceover (Optional) ---
        try:
            from tts_generator import is_tts_enabled, ChatterBoxTTS, TTSConfig, mix_audio_with_video
            
            if is_tts_enabled():
                logging.info("Generating TTS voiceover...")
                
                # Parse SRT with timing for proper audio alignment
                captions = _parse_srt_with_timing(srt_path)
                
                if captions:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                        voiceover_path = Path(tmp_audio.name)
                    
                    # === MEMORY CLEANUP BEFORE TTS ===
                    # ChatterBox TTS requires ~4-6GB VRAM. Clear any cached tensors
                    # from PyCaps/Playwright/video processing before loading.
                    import gc
                    gc.collect()
                    
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            logging.info("GPU memory cleared before TTS loading")
                    except Exception:
                        pass
                    # === END MEMORY CLEANUP ===
                    
                    # Generate per-caption TTS with timing alignment
                    tts = ChatterBoxTTS.get_instance()
                    result_audio = tts.generate_for_captions(
                        captions,
                        voiceover_path,
                        detected_category=detected_category or "action"
                    )
                    
                    if result_audio and result_audio.exists():
                        # Mix voiceover with video
                        mixed_output = output_path.with_stem(output_path.stem + "_voiced")
                        
                        game_vol = float(os.getenv("TTS_GAME_AUDIO_VOLUME", "0.3"))
                        voice_vol = float(os.getenv("TTS_VOICEOVER_VOLUME", "1.0"))
                        
                        if mix_audio_with_video(output_path, voiceover_path, mixed_output, game_vol, voice_vol):
                            # Replace subtitled version with voiced version
                            output_path.unlink()
                            mixed_output.rename(output_path)
                            logging.info(f"TTS voiceover added successfully")
                        else:
                            logging.warning("TTS audio mixing failed, keeping subtitles only")
                    
                    # Cleanup temp voiceover file
                    try:
                        voiceover_path.unlink()
                    except Exception:
                        pass
                    
                    # === CLEANUP TTS MEMORY AFTER USE ===
                    # Free VRAM for subsequent clips
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    # === END TTS CLEANUP ===
                    
        except ImportError:
            logging.debug("TTS module not available, skipping voiceover")
        except Exception as e:
            logging.warning(f"TTS generation failed: {e}, keeping subtitles only")
        
        # Clean up temporary files
        try:
            srt_path.unlink()
            json_path = srt_path.with_suffix(".json")
            if json_path.exists():
                json_path.unlink()
        except Exception:
            pass
        
        return output_path
            
    except Exception as e:
        logging.error(f"Subtitle pipeline failed: {e}")
        return None


def _extract_srt_text(srt_path: Path) -> str:
    """Extract all text from an SRT file for TTS."""
    try:
        with open(srt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        text_lines = []
        for line in lines:
            line = line.strip()
            # Skip empty lines, index numbers, and timestamps
            if not line or line.isdigit() or "-->" in line:
                continue
            text_lines.append(line)
        
        return " ".join(text_lines)
    except Exception:
        return ""


def _parse_srt_with_timing(srt_path: Path) -> list:
    """Parse SRT file and return captions with timing info.
    
    Returns:
        List of {"start": float, "end": float, "text": str}
    """
    def parse_timestamp(ts: str) -> float:
        """Convert SRT timestamp to seconds."""
        ts = ts.strip().replace(",", ".")
        parts = ts.split(":")
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    
    captions = []
    try:
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Split by double newlines (caption blocks)
        blocks = content.strip().split("\n\n")
        
        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) < 3:
                continue
            
            # Find timestamp line
            timestamp_line = None
            text_start_idx = 0
            for i, line in enumerate(lines):
                if "-->" in line:
                    timestamp_line = line
                    text_start_idx = i + 1
                    break
            
            if not timestamp_line:
                continue
            
            # Parse timestamps
            parts = timestamp_line.split("-->")
            if len(parts) != 2:
                continue
            
            start = parse_timestamp(parts[0])
            end = parse_timestamp(parts[1])
            
            # Get text (remaining lines)
            text = " ".join(lines[text_start_idx:]).strip()
            
            # Strip emojis and non-ascii characters for clean TTS
            import re
            # Keep basic punctuation for better TTS prosody (pacing), but remove emojis/symbols
            # This regex keeps ASCII letters, numbers, spaces, and basic punctuation
            text = re.sub(r'[^\x20-\x7E]+', '', text) 
            # Further strip specific symbols that TTS might try to 'read' literally
            text = re.sub(r'[*#_~`]', '', text)
            # Cleanup extra spaces
            text = re.sub(r'\s+', ' ', text).strip()
            
            if text:
                captions.append({
                    "start": start,
                    "end": end,
                    "text": text
                })
        
        return captions
    except Exception as e:
        logging.warning(f"Failed to parse SRT for timing: {e}")
        return []
