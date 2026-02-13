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
    word_lists: dict = None
) -> bool:
    """Apply animated subtitles using PyCaps (TemplateLoader API).
    
    Args:
        video_path: Path to input video
        srt_path: Path to SRT subtitle file
        output_path: Path for output video with subtitles
        config: Subtitle styling configuration
        word_lists: Dictionary of word lists for SemanticTagger
        
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
        args=(str(video_path), str(srt_path), str(output_path), config, word_lists)
    )
    p.start()
    p.join()
    
    # Check for output - PyCaps derives output name from SRT file, not video file
    # So when we pass extended video, it still outputs based on original SRT name
    # SRT path = scene-0.srt → PyCaps outputs scene-0_sub.mp4
    srt_stem = srt_path.stem  # e.g., "scene-0" from "scene-0.srt"
    pycaps_output = srt_path.parent / f"{srt_stem}_sub.mp4"
    
    # Also check based on video path (for non-extended cases)
    pycaps_output_from_video = video_path.with_stem(video_path.stem + "_sub").with_suffix(".mp4")
    
    if p.exitcode == 0:
        # Check our expected path first
        if output_path.exists():
            logging.info(f"PyCaps subtitled video saved to: {output_path}")
            return True
        # Check PyCaps output based on SRT name (most common case when video was extended)
        elif pycaps_output.exists():
            import shutil
            shutil.move(str(pycaps_output), str(output_path))
            logging.info(f"PyCaps subtitled video (from SRT-based _sub.mp4) saved to: {output_path}")
            return True
        # Check PyCaps output based on video name
        elif pycaps_output_from_video.exists():
            import shutil
            shutil.move(str(pycaps_output_from_video), str(output_path))
            logging.info(f"PyCaps subtitled video (from video-based _sub.mp4) saved to: {output_path}")
            return True
        # Check fallback locations
        elif _check_for_fallback_output(video_path, output_path):
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


def _run_pycaps_worker(video_path_str: str, srt_path_str: str, output_path_str: str, config: SubtitleConfig, word_lists: dict = None):
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
        
        # Configure Semantic Tagger for AI tags (word highlighting handled by PyCaps templates)
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
    detected_category: Optional[str] = None,
    story_narration: Optional[str] = None,
    render_meta: Optional[dict] = None
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
        story_narration: Pre-generated narration text for story modes (cross-clip narrative)
        render_meta: Optional dict with source_path, start_time, duration, crop params for re-rendering
        
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
        # Use .mp4 extension since PyCaps always outputs mp4 container
        output_path = video_path.with_stem(video_path.stem + "_subtitled").with_suffix(".mp4")
    else:
        # Ensure output is mp4 even if caller specified different extension
        output_path = output_path.with_suffix(".mp4")
    
    config = SubtitleConfig.from_env()
    srt_path = video_path.with_suffix(".srt")
    
    # Initialize tagging data (word lists for PyCaps SemanticTagger)
    word_lists = {}
    
    # Determine caption style (for both ai_captions and TTS voice selection)
    caption_style = None
    if mode == "ai_captions":
        caption_style = get_caption_style()
        
        # Auto style matching based on detected category
        if caption_style == "auto" and detected_category:
            from ai_providers import ClipScore
            caption_style = ClipScore.CAPTION_STYLE_MAP.get(detected_category, "gaming")
            logging.info(f"Auto-matched caption style: {caption_style} (from category: {detected_category})")
        elif caption_style == "auto":
            caption_style = "gaming"  # Default fallback
    
    try:
        if mode == "speech":
            # Whisper transcription mode
            logging.info("Using speech transcription mode (Whisper)")
            srt_path = transcribe_audio(video_path, srt_path)
            
        elif mode == "ai_captions":
            # AI-generated captions mode
            from ai_providers import (
                generate_ai_captions, 
                captions_to_srt, 
                batch_tag_captions, 
                apply_tags_to_pycaps, 
                add_emojis_to_caption,
                Caption
            )
            
            # Check if we have pre-generated story narration
            if story_narration:
                logging.info("Using pre-generated story narration")
                
                # Get video duration
                import subprocess
                import re
                ffprobe_cmd = [
                    "ffprobe", "-v", "error", 
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(video_path)
                ]
                try:
                    duration = float(subprocess.check_output(ffprobe_cmd).decode().strip())
                except:
                    duration = 30.0
                
                # Split narration into sentences
                # Enhanced regex to handle:
                # 1. English: [.!?] followed by whitespace
                # 2. Japanese/CJK: [。！？] (no whitespace needed)
                # 3. Newlines
                sentences = re.split(r'(?<=[.!?])\s+|(?<=[。！？])', story_narration.strip())
                sentences = [s.strip() for s in sentences if s.strip()]
                
                if not sentences:
                    sentences = [story_narration]
                
                # For story mode: Generate TTS FIRST to get actual durations
                # Then create SRT based on actual audio timing
                from tts_generator import is_tts_enabled, QwenTTS, TTSConfig, generate_voice_description
                
                if is_tts_enabled():
                    import gc
                    import torch
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logging.info("GPU memory cleared before TTS loading")
                    
                    try:
                        tts_config = TTSConfig.from_env()
                        tts = QwenTTS(tts_config)
                        
                        # Load the model explicitly
                        tts._ensure_initialized()
                        
                        # Get voice description for story mode
                        # For story modes, caption_style takes priority over detected_category
                        # because story modes have specific voice presets (e.g., story_dramatic = Female)
                        voice_context = caption_style if caption_style and caption_style.startswith("story_") else (detected_category or caption_style)
                        voice_desc = generate_voice_description(voice_context)
                        logging.info(f"Using voice preset for: {voice_context}")
                        
                        # Preprocess slang in sentences for TTS
                        from tts_generator import preprocess_text_for_tts
                        processed_sentences = [preprocess_text_for_tts(s) for s in sentences]
                        
                        # Generate TTS for each sentence and SAVE the audio for later
                        # This avoids the sync issue of generating TTS twice
                        sentence_durations = []
                        sentence_audio_segments = []  # Store actual audio to reuse
                        tts_sample_rate = 24000  # Default, will be updated
                        
                        for sentence in processed_sentences:
                            # Generate TTS
                            wavs, sr = tts._model.generate_voice_design(
                                text=sentence,
                                instruct=voice_desc,
                                language=tts.config.get_language_name(),
                            )
                            
                            tts_sample_rate = sr
                            
                            if wavs and len(wavs) > 0:
                                audio_duration = len(wavs[0]) / sr
                                sentence_durations.append(audio_duration)
                                sentence_audio_segments.append(wavs[0])  # Save audio
                            else:
                                # Fallback to word-count estimate
                                word_count = len(sentence.split())
                                sentence_durations.append(max(1.5, word_count * 0.4))
                                sentence_audio_segments.append(None)  # No audio
                        
                        # Now create SRT with ACTUAL TTS durations
                        result_captions = []
                        current_time = 0.5  # Small buffer
                        
                        # Split long sentences into smaller visual chunks specifically for story mode
                        # This prevents the "wall of text" issue in PyCaps
                        MAX_WORDS_PER_CAPTION = 7
                        
                        for sentence, tts_duration in zip(sentences, sentence_durations):
                            # --- CJK Detection ---
                            is_cjk = any("\u4e00" <= char <= "\u9fff" or "\u3040" <= char <= "\u30ff" for char in sentence)
                            MAX_CJK_CHARS = 18  # Characters per line for CJK
                            
                            if is_cjk:
                                # Character-based splitting for CJK
                                if len(sentence) <= MAX_CJK_CHARS:
                                    # Short sentence
                                    end_time = current_time + tts_duration
                                    result_captions.append(Caption(
                                        start_time=current_time,
                                        end_time=end_time,
                                        text=sentence,
                                        style="narrative"
                                    ))
                                    current_time = end_time
                                else:
                                    # Long sentence -> Split by chars
                                    chunks = [sentence[i:i+MAX_CJK_CHARS] for i in range(0, len(sentence), MAX_CJK_CHARS)]
                                    
                                    total_chars = len(sentence)
                                    chunk_start = current_time
                                    
                                    for i, chunk in enumerate(chunks):
                                        chunk_ratio = len(chunk) / total_chars if total_chars > 0 else 1.0
                                        chunk_dur = tts_duration * chunk_ratio
                                        
                                        chunk_end = chunk_start + chunk_dur
                                        if i == len(chunks) - 1:
                                            chunk_end = current_time + tts_duration
                                            
                                        result_captions.append(Caption(
                                            start_time=chunk_start,
                                            end_time=chunk_end,
                                            text=chunk,
                                            style="narrative"
                                        ))
                                        chunk_start = chunk_end
                                    
                                    # Update current_time for the next sentence
                                    current_time = current_time + tts_duration
                            else:
                                # --- Standard Word-based Logic ---
                                words = sentence.split()
                                if not words:
                                    continue
                                
                                # If sentence is short enough, keep as one
                                if len(words) <= MAX_WORDS_PER_CAPTION:
                                    end_time = current_time + tts_duration
                                    result_captions.append(Caption(
                                        start_time=current_time,
                                        end_time=end_time,
                                        text=sentence,
                                        style="narrative"
                                    ))
                                    current_time = end_time
                                else:
                                    # Split into chunks
                                    chunks = []
                                    for i in range(0, len(words), MAX_WORDS_PER_CAPTION):
                                        chunk_words = words[i:i + MAX_WORDS_PER_CAPTION]
                                        chunks.append(" ".join(chunk_words))
                                    
                                    # Distribute duration proportionally
                                    total_chars = len(sentence)
                                    chunk_start = current_time
                                    
                                    for i, chunk in enumerate(chunks):
                                        # Calculate duration based on character count ratio
                                        # (characters correlate better with speaking time than word count)
                                        chunk_ratio = len(chunk) / total_chars if total_chars > 0 else 1.0
                                        chunk_dur = tts_duration * chunk_ratio
                                        
                                        # For the last chunk, ensure we essentially align with the end
                                        # (floating point fix)
                                        chunk_end = chunk_start + chunk_dur
                                        if i == len(chunks) - 1:
                                            chunk_end = current_time + tts_duration
                                            
                                        result_captions.append(Caption(
                                            start_time=chunk_start,
                                            end_time=chunk_end,
                                            text=chunk,
                                            style="narrative"
                                        ))
                                        chunk_start = chunk_end
                                    
                                    # Update current_time for the next sentence
                                    current_time = current_time + tts_duration
                        
                        # === SAVE PRE-GENERATED TTS AUDIO ===
                        # Build the final audio from saved segments to avoid regenerating
                        import numpy as np
                        import scipy.io.wavfile as wav
                        
                        audio_parts = []
                        audio_current_time = 0.0
                        
                        for i, (tts_dur, audio_seg) in enumerate(zip(sentence_durations, sentence_audio_segments)):
                            # Calculate start time for this sentence (same logic as captions)
                            # Sentences start at 0.5s buffer, then sequentially
                            expected_start = 0.5 + sum(sentence_durations[:i])
                            
                            # Add silence gap if needed
                            silence_needed = expected_start - audio_current_time
                            if silence_needed > 0:
                                silence_samples = int(silence_needed * tts_sample_rate)
                                audio_parts.append(np.zeros(silence_samples, dtype=np.float32))
                                audio_current_time += silence_needed
                            
                            # Add audio segment
                            if audio_seg is not None:
                                audio_parts.append(audio_seg.astype(np.float32))
                                audio_current_time += len(audio_seg) / tts_sample_rate
                            else:
                                # Add silence for failed generations
                                silence_samples = int(tts_dur * tts_sample_rate)
                                audio_parts.append(np.zeros(silence_samples, dtype=np.float32))
                                audio_current_time += tts_dur
                        
                        # Save pre-generated TTS to a temp file for later use
                        if audio_parts:
                            story_tts_audio = np.concatenate(audio_parts)
                            story_tts_audio_int16 = (story_tts_audio * 32767).astype(np.int16)
                            
                            # Save to temp file that will be used instead of regenerating
                            story_tts_path = srt_path.with_suffix(".story_tts.wav")
                            wav.write(str(story_tts_path), tts_sample_rate, story_tts_audio_int16)
                            logging.info(f"Pre-generated story TTS saved: {audio_current_time:.1f}s")
                        
                        # Clean up TTS model (but keep the saved audio file)
                        del tts
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                    except Exception as e:
                        logging.warning(f"TTS pre-generation failed, falling back to word-count estimate: {e}")
                        # Fallback to original word-count based timing
                        total_words = sum(len(s.split()) for s in sentences)
                        result_captions = []
                        current_time = 0.5
                        
                        # Split into chunks logic for fallback
                        MAX_WORDS_PER_CAPTION = 7
                        
                        for sentence in sentences:
                            # Calculate TOTAL duration for this sentence
                            word_count = len(sentence.split())
                            if total_words > 0:
                                proportion = word_count / total_words
                                total_segment_duration = max(1.5, (duration - 1.0) * proportion)
                            else:
                                total_segment_duration = max(1.5, word_count * 0.4)
                            
                            words = sentence.split()
                            if not words: continue

                            if len(words) <= MAX_WORDS_PER_CAPTION:
                                end_time = min(current_time + total_segment_duration, duration - 0.5)
                                result_captions.append(Caption(
                                    start_time=current_time,
                                    end_time=end_time,
                                    text=sentence,
                                    style="narrative"
                                ))
                                current_time = end_time
                            else:
                                # Chunk logic
                                chunks = []
                                for i in range(0, len(words), MAX_WORDS_PER_CAPTION):
                                    chunk_words = words[i:i + MAX_WORDS_PER_CAPTION]
                                    chunks.append(" ".join(chunk_words))
                                
                                total_chars = len(sentence)
                                chunk_start = current_time
                                
                                for i, chunk in enumerate(chunks):
                                    chunk_ratio = len(chunk) / total_chars if total_chars > 0 else 1.0
                                    chunk_dur = total_segment_duration * chunk_ratio
                                    
                                    # Alignment fix for last chunk
                                    if i == len(chunks) - 1:
                                        overall_end_time = min(current_time + total_segment_duration, duration - 0.5)
                                        chunk_end = overall_end_time
                                    else:
                                         chunk_end = chunk_start + chunk_dur
                                    
                                    result_captions.append(Caption(
                                        start_time=chunk_start,
                                        end_time=chunk_end,
                                        text=chunk,
                                        style="narrative"
                                    ))
                                    chunk_start = chunk_end
                                
                                current_time = chunk_start
                else:
                    # No TTS - use word-count estimate
                    total_words = sum(len(s.split()) for s in sentences)
                    result_captions = []
                    current_time = 0.5
                    
                    # Split into chunks logic for No-TTS mode
                    MAX_WORDS_PER_CAPTION = 7
                    
                    for sentence in sentences:
                        # Calculate TOTAL duration for this sentence
                        word_count = len(sentence.split())
                        if total_words > 0:
                            proportion = word_count / total_words
                            total_segment_duration = max(1.5, (duration - 1.0) * proportion)
                        else:
                            total_segment_duration = max(1.5, word_count * 0.4)
                        
                        words = sentence.split()
                        if not words: continue

                        if len(words) <= MAX_WORDS_PER_CAPTION:
                            end_time = min(current_time + total_segment_duration, duration - 0.5)
                            result_captions.append(Caption(
                                start_time=current_time,
                                end_time=end_time,
                                text=sentence,
                                style="narrative"
                            ))
                            current_time = end_time
                        else:
                            # Chunk logic
                            chunks = []
                            for i in range(0, len(words), MAX_WORDS_PER_CAPTION):
                                chunk_words = words[i:i + MAX_WORDS_PER_CAPTION]
                                chunks.append(" ".join(chunk_words))
                            
                            total_chars = len(sentence)
                            chunk_start = current_time
                            
                            for i, chunk in enumerate(chunks):
                                chunk_ratio = len(chunk) / total_chars if total_chars > 0 else 1.0
                                chunk_dur = total_segment_duration * chunk_ratio
                                
                                # Alignment fix for last chunk
                                if i == len(chunks) - 1:
                                    overall_end_time = min(current_time + total_segment_duration, duration - 0.5)
                                    chunk_end = overall_end_time
                                else:
                                     chunk_end = chunk_start + chunk_dur
                                
                                result_captions.append(Caption(
                                    start_time=chunk_start,
                                    end_time=chunk_end,
                                    text=chunk,
                                    style="narrative"
                                ))
                                chunk_start = chunk_end
                            
                            current_time = chunk_start
                
                # Save SRT
                captions_to_srt(result_captions, srt_path)
                logging.info(f"Generated {len(result_captions)} story narration segments ({caption_style} style)")
                
            else:
                # Regular AI caption generation
                logging.info("Using AI caption generation mode")
                
                # Get configured max_captions (0 = auto/dynamic)
                max_captions_env = int(os.getenv("MAX_CAPTIONS", "0"))
                is_auto_captions = (max_captions_env == 0)
                
                # Get video duration for dynamic calculation
                import subprocess
                import json
                try:
                    probe_cmd = [
                        "ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", str(video_path)
                    ]
                    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
                    probe_data = json.loads(probe_result.stdout)
                    duration = float(probe_data["format"]["duration"])
                    
                    # Dynamic caption count: ~1 caption per 2-4 seconds (varies by style)
                    # Story modes: 1 caption per 5-6 seconds (narrative pacing)
                    # Regular modes: 1 caption per 2-3 seconds (punchy)
                    is_story_mode = caption_style.startswith("story_")
                    
                    if is_story_mode:
                        # Story modes: narrative pacing, but not too sparse
                        # Aim for ~1 caption per 5-6 seconds for good coverage
                        ideal_captions = max(4, int(duration / 5.5))  # 1 caption per ~5.5 seconds
                    else:
                        # Regular modes: shorter punchy captions
                        ideal_captions = max(2, int(duration / 2.5))  # 1 caption per ~2.5 seconds
                    
                    if is_auto_captions:
                        # Auto mode: use ideal count (no cap)
                        max_captions = ideal_captions
                        logging.info(f"Dynamic caption count: {max_captions} captions for {duration:.1f}s video "
                                   f"(~{duration/max_captions:.1f}s per caption)")
                    else:
                        # Manual mode: cap at configured maximum
                        max_captions = min(ideal_captions, max_captions_env)
                        logging.info(f"Dynamic caption count: {max_captions} captions for {duration:.1f}s video "
                                   f"(~{duration/max_captions:.1f}s per caption, max={max_captions_env})")
                    
                except Exception as e:
                    # Fallback to env variable or default if duration detection fails
                    logging.warning(f"Could not determine video duration: {e}")
                    max_captions = max_captions_env if max_captions_env > 0 else 10
                
                result = generate_ai_captions(video_path, style=caption_style, max_captions=max_captions)
                
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
                    category = detected_category or caption_style
                    
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
                    
                    # 3. Generate PyCaps Tagging Data (word lists only, no custom CSS)
                    word_lists = apply_tags_to_pycaps(result.captions, tag_results)
                    
                    # Convert enhanced AI captions to SRT
                    captions_to_srt(enhanced_captions, srt_path)
                    logging.info(f"Generated {len(enhanced_captions)} AI captions ({caption_style} style)")
        else:
            logging.warning(f"Unknown subtitle mode: {mode}. Using ai_captions.")
            return generate_subtitles(video_path, output_path, detected_category, render_meta=render_meta)
        
        # Check if we got any subtitles
        if not srt_path.exists() or srt_path.stat().st_size == 0:
            logging.warning("No subtitles generated (empty SRT)")
            return None
        
        # === PRE-GENERATE TTS FOR AUTO MODE (non-story) ===
        # For auto/regular modes, generate TTS BEFORE PyCaps so we can:
        # 1. Adjust SRT timings to match actual TTS audio durations
        # 2. Extend video if TTS is longer than video
        # This mirrors story mode's approach and prevents subtitle/voiceover desync
        auto_tts_path = srt_path.with_suffix(".auto_tts.wav")
        
        if mode == "ai_captions" and not story_narration:  # Only for non-story caption flows
            try:
                from tts_generator import is_tts_enabled, QwenTTS, preprocess_text_for_tts, generate_voice_description
                
                if is_tts_enabled():
                    import gc
                    import torch
                    import numpy as np
                    import scipy.io.wavfile as wav_io
                    
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logging.info("GPU memory cleared before TTS loading")
                    
                    tts = QwenTTS.get_instance()
                    tts._ensure_initialized()
                    if tts._model is None:
                        logging.warning("TTS model unavailable for auto pre-generation, keeping subtitles only")
                    else:
                        # Get voice description
                        voice_context = caption_style if caption_style else (detected_category or "action")
                        voice_desc = generate_voice_description(voice_context)
                        first_line = voice_desc.split('\n')[0] if '\n' in voice_desc else voice_desc[:60]
                        logging.info(f"Generating voice for context: {voice_context}")
                        logging.info(f"Voice: {first_line}...")
                        
                        # Parse the SRT to get captions
                        captions_for_tts = _parse_srt_with_timing(srt_path)
                        
                        if captions_for_tts:
                            audio_segments = []
                            updated_captions = []
                            current_audio_time = 0.0
                            sample_rate = 24000
                            
                            logging.info(f"Generating TTS for {len(captions_for_tts)} captions ({detected_category or 'auto'} mode)")
                            
                            for i, cap in enumerate(captions_for_tts):
                                raw_text = cap.get("text", "").strip()
                                start_time = cap.get("start", 0.0)
                                end_time = cap.get("end", start_time + 2.0)
                                fallback_start = max(start_time, current_audio_time)
                                fallback_end = max(fallback_start + 0.1, fallback_start + (end_time - start_time))
                                
                                text = preprocess_text_for_tts(_sanitize_tts_input(raw_text))
                                if not text:
                                    updated_captions.append({"start": fallback_start, "end": fallback_end, "text": raw_text})
                                    continue
                                
                                # Add silence gap to reach this caption's start
                                silence_dur = max(0, start_time - current_audio_time)
                                if silence_dur > 0:
                                    silence = np.zeros(int(silence_dur * sample_rate), dtype=np.float32)
                                    audio_segments.append(silence)
                                    current_audio_time += silence_dur
                                
                                # Track true audio start for this caption after any carry-over shift
                                actual_start = current_audio_time
                                
                                try:
                                    wavs, sr = tts._model.generate_voice_design(
                                        text=text,
                                        instruct=voice_desc,
                                        language=tts.config.get_language_name(),
                                    )
                                    sample_rate = sr
                                    
                                    if wavs and len(wavs) > 0:
                                        audio = wavs[0].astype(np.float32)
                                        tts_dur = len(audio) / sample_rate
                                        audio_segments.append(audio)
                                        current_audio_time += tts_dur
                                        updated_captions.append({
                                            "start": actual_start,
                                            "end": max(actual_start + 0.1, current_audio_time),
                                            "text": raw_text
                                        })
                                    else:
                                        updated_captions.append({"start": fallback_start, "end": fallback_end, "text": raw_text})
                                except Exception as e:
                                    logging.warning(f"TTS failed for caption {i}: {e}")
                                    updated_captions.append({"start": fallback_start, "end": fallback_end, "text": raw_text})
                            
                            if audio_segments:
                                final_audio = np.concatenate(audio_segments)
                                final_audio_int16 = (final_audio * 32767).astype(np.int16)
                                wav_io.write(str(auto_tts_path), sample_rate, final_audio_int16)
                                total_tts_dur = len(final_audio) / sample_rate
                                logging.info(f"TTS audio generated: {total_tts_dur:.1f}s for {len(captions_for_tts)} captions")
                                
                                # Rewrite SRT using actual generated audio timing to keep subtitle/voice sync exact.
                                original_end = max((cap.get("end", 0.0) for cap in captions_for_tts), default=0.0)
                                adjusted_end = max((cap.get("end", 0.0) for cap in updated_captions), default=0.0)
                                shift = max(0.0, adjusted_end - original_end)
                                logging.info(f"Adjusted SRT timing: shifted {shift:.1f}s to match TTS durations")
                                
                                srt_lines = []
                                for idx, cap in enumerate(updated_captions, 1):
                                    start_ts = _format_timestamp(cap["start"])
                                    end_ts = _format_timestamp(cap["end"])
                                    srt_lines.append(str(idx))
                                    srt_lines.append(f"{start_ts} --> {end_ts}")
                                    srt_lines.append(cap["text"])
                                    srt_lines.append("")
                                
                                with open(srt_path, "w", encoding="utf-8") as f:
                                    f.write("\n".join(srt_lines))
                        
                        # Clean up TTS model memory
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
            except ImportError:
                logging.debug("TTS module not available, skipping pre-generation")
            except Exception as e:
                logging.warning(f"TTS pre-generation failed for auto mode: {e}")
                # Clean up failed TTS file
                if auto_tts_path.exists():
                    try:
                        auto_tts_path.unlink()
                    except Exception:
                        pass
        
        # === EXTEND VIDEO TO MATCH TTS DURATION (if needed) ===
        # This MUST happen BEFORE PyCaps so subtitles are rendered on the full-length video
        story_tts_path = srt_path.with_suffix(".story_tts.wav")
        # Use whichever TTS audio exists (story mode or auto mode)
        pre_generated_tts_path = story_tts_path if story_tts_path.exists() else (auto_tts_path if auto_tts_path.exists() else None)
        video_for_pycaps = video_path
        extended_video_temp = None
        
        if pre_generated_tts_path and render_meta:
            from tts_generator import get_audio_duration, get_video_duration, rerender_video_longer, RenderMeta
            
            tts_duration = get_audio_duration(pre_generated_tts_path)
            video_duration = get_video_duration(video_path)
            
            if tts_duration > video_duration + 1.0:  # TTS is significantly longer
                logging.info(f"TTS ({tts_duration:.1f}s) longer than video ({video_duration:.1f}s)")
                logging.info("Re-rendering video to TTS duration BEFORE applying subtitles...")
                
                # Re-render video to match TTS duration + buffer
                extended_video_temp = video_path.with_stem(video_path.stem + "_extended_for_subs")
                target_duration = tts_duration + 1.0
                
                try:
                    tts_render_meta = RenderMeta(
                        source_path=Path(render_meta["source_path"]),
                        start_time=render_meta["start_time"],
                        original_duration=render_meta["duration"],
                        output_width=render_meta["output_width"],
                        output_height=render_meta["output_height"],
                        crop_x=render_meta["crop_x"],
                        crop_y=render_meta["crop_y"],
                        crop_w=render_meta["crop_w"],
                        crop_h=render_meta["crop_h"],
                        bg_width=render_meta.get("bg_width", render_meta["output_width"]),
                        bg_height=render_meta.get("bg_height", render_meta["output_height"]),
                        is_vertical_bg=render_meta.get("is_vertical_bg", True),
                    )
                    
                    if rerender_video_longer(tts_render_meta, target_duration, extended_video_temp):
                        video_for_pycaps = extended_video_temp
                        logging.info(f"Video extended to {target_duration:.1f}s for subtitle rendering")
                    else:
                        logging.warning("Video extension failed, subtitles may be truncated")
                except Exception as e:
                    logging.warning(f"Could not extend video: {e}")
        
        # Apply subtitles using PyCaps (to the potentially extended video)
        success = apply_pycaps_subtitles(video_for_pycaps, srt_path, output_path, config, word_lists=word_lists)
        
        # DON'T clean up extended video yet - we need its audio track for TTS mixing!
        # PyCaps strips audio, so we need it from the source video
        audio_source_video = extended_video_temp if extended_video_temp and extended_video_temp.exists() else video_path
        
        if not success:
            logging.error("Subtitle burn-in failed")
            # Cleanup extended video on failure
            if extended_video_temp and extended_video_temp.exists():
                try:
                    extended_video_temp.unlink()
                except Exception:
                    pass
            return None
        
        # --- TTS Voiceover (Optional) ---
        try:
            from tts_generator import is_tts_enabled
            
            if is_tts_enabled():
                logging.info("Generating TTS voiceover...")
                
                # Check if we have pre-generated TTS audio (story mode or auto mode)
                # This avoids regenerating TTS which would cause sync issues
                story_tts_path = srt_path.with_suffix(".story_tts.wav")
                
                # Pick whichever pre-generated TTS exists
                if story_tts_path.exists():
                    pre_gen_tts = story_tts_path
                elif auto_tts_path.exists():
                    pre_gen_tts = auto_tts_path
                else:
                    pre_gen_tts = None
                
                if pre_gen_tts:
                    # === USE PRE-GENERATED TTS (story or auto mode) ===
                    logging.info(f"Using pre-generated TTS audio (perfect sync)")
                    voiceover_path = pre_gen_tts
                    
                    # Mix voiceover with video
                    mixed_output = output_path.with_stem(output_path.stem + "_voiced")
                    
                    game_vol = float(os.getenv("TTS_GAME_AUDIO_VOLUME", "0.3"))
                    voice_vol = float(os.getenv("TTS_VOICEOVER_VOLUME", "1.0"))
                    
                    # PyCaps output has NO AUDIO, so we need to mix 3 streams:
                    # 1. Video from PyCaps output (output_path)
                    # 2. Audio from source video (audio_source_video)
                    # 3. TTS voiceover
                    logging.info(f"Mixing: video from PyCaps, audio from source, TTS voiceover")
                    
                    filter_complex = (
                        f"[1:a]volume={game_vol}[game];"
                        f"[2:a]volume={voice_vol}[voice];"
                        "[game][voice]amix=inputs=2:duration=longest:dropout_transition=2[aout]"
                    )
                    
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", str(output_path),         # Video from PyCaps (no audio)
                        "-i", str(audio_source_video),  # Audio source (original/extended video)
                        "-i", str(voiceover_path),      # TTS voiceover
                        "-filter_complex", filter_complex,
                        "-map", "0:v",                  # Video from PyCaps
                        "-map", "[aout]",               # Mixed audio
                        "-c:v", "copy",
                        "-c:a", "aac",
                        "-b:a", "192k",
                        str(mixed_output)
                    ]
                    
                    import subprocess
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0 and mixed_output.exists():
                        # Replace subtitled version with voiced version
                        output_path.unlink()
                        mixed_output.rename(output_path)
                        logging.info(f"TTS voiceover added successfully")
                    else:
                        logging.warning(f"TTS audio mixing failed: {result.stderr[:300]}")
                    
                    # Cleanup the pre-generated TTS file
                    try:
                        pre_gen_tts.unlink()
                    except Exception:
                        pass
                        
                    # Cleanup extended video temp file if it was used
                    if extended_video_temp and extended_video_temp.exists():
                        try:
                            extended_video_temp.unlink()
                        except Exception:
                            pass
                else:
                    logging.warning("No pre-generated TTS audio found; skipping legacy on-the-fly TTS fallback to preserve sync")
                    if extended_video_temp and extended_video_temp.exists():
                        try:
                            extended_video_temp.unlink()
                        except Exception:
                            pass
                    
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


def _sanitize_tts_input(text: str) -> str:
    """Sanitize caption text for TTS while preserving visible subtitle text."""
    import unicodedata

    if not text:
        return ""

    cleaned = re.sub(r"\*[^*]+\*", "", text).strip()
    cleaned = cleaned.replace("\u200d", "").replace("\ufe0f", "").replace("\ufe0e", "")
    cleaned = "".join(ch for ch in cleaned if unicodedata.category(ch) not in ("So", "Sk"))
    cleaned = re.sub(r"[*#_~`]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


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
            
            # Preserve original subtitle text (including emojis) for rendering.
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
