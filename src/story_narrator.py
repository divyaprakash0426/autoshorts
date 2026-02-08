"""Cross-clip story narration for unified narrative flow.

This module generates cohesive narratives that span across multiple clips,
creating story arcs for News, Roast, Creepypasta, and Dramatic modes.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclass
class ClipNarration:
    """Narration for a single clip in a multi-clip story."""
    
    clip_index: int
    narration_text: str
    detected_category: str
    clip_path: Path


# Language name mapping for narration prompts
LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Chinese (Mandarin)",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}


def generate_unified_story(
    clips: List[Tuple[Path, str, Optional[dict]]],
    story_style: str,
    language: str = None,
) -> List[ClipNarration]:
    """Generate a unified narrative across multiple clips.
    
    Args:
        clips: List of (clip_path, detected_category, render_meta) tuples
        story_style: One of: story_news, story_roast, story_creepypasta, story_dramatic
        language: Language code (en, ja, ko, etc.) - defaults to TTS_LANGUAGE env var
        
    Returns:
        List of ClipNarration objects with per-clip narration text
    """
    if not clips:
        logging.warning("No clips provided for story narration")
        return []
    
    if len(clips) == 1:
        # Single clip - use regular per-clip narration
        logging.info("Only 1 clip - using standard narration instead of story mode")
        return []
    
    # Get language from parameter or environment
    if language is None:
        language = os.getenv("TTS_LANGUAGE", "en")
    
    logging.info(f"Generating unified {story_style} narrative for {len(clips)} clips (language: {language})")
    
    provider = os.getenv("AI_PROVIDER", "gemini").lower()
    
    # Extract just (path, category) for story generation (render_meta not needed)
    clips_for_story = [(p, cat) for p, cat, _ in clips]
    
    if provider == "openai":
        return _generate_story_openai(clips_for_story, story_style, language)
    elif provider == "local":
        logging.warning("Story mode not available in local mode")
        return []
    else:  # gemini
        return _generate_story_gemini(clips_for_story, story_style, language)


def _get_story_prompt(story_style: str, num_clips: int, language: str = "en") -> str:
    """Get the prompt for cross-clip story generation."""
    
    # Get full language name
    lang_name = LANGUAGE_NAMES.get(language, "English")
    
    style_guides = {
        "story_news": f"""You are a professional esports broadcaster creating a {num_clips}-part news segment.

Create a cohesive news narrative that flows across all {num_clips} clips:
- Clip 1: Opening/intro to the story
- Middle clips: Development and escalation
- Final clip: Conclusion or recap

Style: Professional, analytical, building excitement like a sports broadcast.
Tone: Clear, confident, measured pace.

IMPORTANT: Generate 4-6 sentences per clip segment to fill the video duration. Space narration throughout each clip - don't front-load all text at the start.""",

        "story_roast": f"""You are a sarcastic commentator creating a {num_clips}-part roast.

Create a playful roasting narrative that flows across all {num_clips} clips:
- Clip 1: Set up the situation
- Middle clips: Build the comedy/mockery
- Final clip: Deliver the punchline or conclusion

Style: Sarcastic but not mean, comedic timing, playful mockery.
Tone: Amused, slightly condescending but entertaining.

IMPORTANT: Generate 4-6 sentences per clip segment to fill the video duration. Space the roasting throughout each clip.""",

        "story_creepypasta": f"""You are a horror narrator creating a {num_clips}-part creepypasta.

Create an unsettling narrative that builds tension across all {num_clips} clips:
- Clip 1: Something feels wrong...
- Middle clips: The horror escalates
- Final clip: The terrifying reveal or cliffhanger

Style: Horror/tension building, slow reveals, ominous.
Tone: Deep, deliberate, building dread with ellipses.

IMPORTANT: Generate 4-6 sentences per clip segment to fill the video duration. Build tension throughout each clip with proper pacing.""",

        "story_dramatic": f"""You are an epic narrator creating a {num_clips}-part cinematic story.

Create a dramatic narrative arc across all {num_clips} clips:
- Clip 1: The stage is set
- Middle clips: The stakes rise
- Final clip: The climactic conclusion

Style: Epic, cinematic, movie-trailer intensity.
Tone: Powerful, inspiring, grand.

IMPORTANT: Generate 4-6 sentences per clip segment to fill the video duration. Space the epic narration throughout each clip."""
    }
    
    base_prompt = style_guides.get(story_style, style_guides["story_dramatic"])
    
    # Add language instruction for non-English
    if language != "en":
        language_instruction = f"""\n\nLANGUAGE REQUIREMENT:
- Generate ALL narration text in {lang_name}.
- Adapt the tone and style to be culturally appropriate for {lang_name} speakers.
- Use natural {lang_name} expressions and phrasing.
- Do NOT translate literally from English - create authentic {lang_name} narration."""
        return base_prompt + language_instruction
    
    return base_prompt


def _generate_story_gemini(
    clips: List[Tuple[Path, str]],
    story_style: str,
    language: str = "en"
) -> List[ClipNarration]:
    """Generate cross-clip story using Gemini API."""
    
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        logging.warning("GEMINI_API_KEY not set, skipping story mode")
        return []
    
    try:
        import time
        from google import genai
        client = genai.Client(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
        
        # Upload all clip videos with retry logic
        import time
        uploaded_files = []
        upload_attempts = 3
        
        for clip_path, _ in clips:
            video_file = None
            for attempt in range(upload_attempts):
                try:
                    with open(clip_path, "rb") as f:
                        video_file = client.files.upload(
                            file=f,
                            config={"mime_type": "video/mp4"}
                        )
                    uploaded_files.append(video_file)
                    break
                except Exception as e:
                    if attempt < upload_attempts - 1:
                        logging.warning(f"Gemini upload attempt {attempt + 1} for {clip_path.name} failed: {e}, retrying...")
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        # Cleanup any already uploaded files
                        for f in uploaded_files:
                            try:
                                client.files.delete(name=f.name)
                            except Exception:
                                pass
                        raise
        
        # Poll until all files are ACTIVE (Gemini processes uploads asynchronously)
        max_wait = 180  # Increased to 3 minutes per file
        poll_interval = 2
        
        for i, video_file in enumerate(uploaded_files):
            waited = 0
            logging.info(f"Waiting for Gemini to process file {i+1}/{len(uploaded_files)}: {video_file.name}")
            while waited < max_wait:
                try:
                    file_status = client.files.get(name=video_file.name)
                    state = file_status.state.name
                    logging.debug(f"File {i+1} state: {state} (waited {waited}s)")
                    
                    if state == "ACTIVE":
                        logging.info(f"File {i+1} ready after {waited}s")
                        break
                    elif state == "FAILED":
                        raise RuntimeError(f"Gemini file processing failed: {video_file.name}")
                    
                    time.sleep(poll_interval)
                    waited += poll_interval
                except Exception as e:
                    if "precondition" in str(e).lower() and waited < max_wait:
                        # File might still be processing, continue polling
                        logging.debug(f"Precondition error (file still processing), continuing... ({waited}s)")
                        time.sleep(poll_interval)
                        waited += poll_interval
                    else:
                        # Cleanup all uploaded files
                        for f in uploaded_files:
                            try:
                                client.files.delete(name=f.name)
                            except Exception:
                                pass
                        raise
            else:
                # Cleanup all uploaded files on timeout
                for f in uploaded_files:
                    try:
                        client.files.delete(name=f.name)
                    except Exception:
                        pass
                raise RuntimeError(f"Gemini file processing timed out after {max_wait}s")
        
        # Build prompt
        style_guide = _get_story_prompt(story_style, len(clips), language)
        
        prompt = f"""{style_guide}

I'm showing you {len(clips)} gameplay clips in sequence. Watch them all and create a unified narrative.

For each clip, provide:
1. A narration segment (2-3 sentences) that connects to the overall story
2. Ensure narrative continuity - each segment should flow from the previous

Return ONLY valid JSON:
{{
  "narrative_arc": "Brief description of the overall story",
  "segments": [
    {{"clip_index": 0, "narration": "Segment text for clip 1..."}},
    {{"clip_index": 1, "narration": "Segment text for clip 2..."}},
    {{"clip_index": 2, "narration": "Segment text for clip 3..."}}
  ]
}}

The clips:"""
        
        # Create content parts with all videos
        content_parts = [prompt]
        for i, video_file in enumerate(uploaded_files):
            content_parts.append(f"\n\nClip {i+1}:")
            content_parts.append(video_file)
        
        # Add extra delay before using the files to ensure they're fully ready
        time.sleep(1)
        
        # Retry logic for generate_content in case of transient errors
        max_retries = 3
        response = None
        for retry in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=content_parts
                )
                break
            except Exception as e:
                if "precondition" in str(e).lower() and retry < max_retries - 1:
                    logging.warning(f"Precondition error on attempt {retry + 1}, retrying after delay...")
                    time.sleep(3 * (retry + 1))  # Increasing delay: 3s, 6s, 9s
                else:
                    # Cleanup uploaded files before raising
                    for f in uploaded_files:
                        try:
                            client.files.delete(name=f.name)
                        except Exception:
                            pass
                    raise
        
        result_text = response.text.strip()
        
        # Clean up markdown code blocks if present
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        data = json.loads(result_text)
        
        # Parse response into ClipNarration objects
        narrations = []
        for segment in data.get("segments", []):
            idx = segment.get("clip_index", 0)
            text = segment.get("narration", "")
            
            if idx < len(clips):
                clip_path, category = clips[idx]
                narrations.append(ClipNarration(
                    clip_index=idx,
                    narration_text=text,
                    detected_category=category,
                    clip_path=clip_path
                ))
        
        # Cleanup uploaded files
        for video_file in uploaded_files:
            try:
                client.files.delete(name=video_file.name)
            except Exception:
                pass
        
        arc = data.get("narrative_arc", "")
        logging.info(f"Generated story arc: {arc}")
        logging.info(f"Created {len(narrations)} narrative segments")
        
        return narrations
        
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse Gemini response: {e}")
        logging.error(f"Response was: {result_text[:200]}")
        return []
    except Exception as e:
        logging.error(f"Gemini story generation failed: {e}")
        return []


def _generate_story_openai(
    clips: List[Tuple[Path, str]],
    story_style: str,
    language: str = "en"
) -> List[ClipNarration]:
    """Generate cross-clip story using OpenAI API (via frame extraction)."""
    
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logging.warning("OPENAI_API_KEY not set, skipping story mode")
        return []
    
    try:
        from openai import OpenAI
        import base64
        import subprocess
        import tempfile
        
        client = OpenAI(api_key=api_key)
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
        
        # Build prompt
        style_guide = _get_story_prompt(story_style, len(clips), language)
        
        prompt = f"""{style_guide}

I'm showing you {len(clips)} gameplay clips as a sequence of frames. Create a unified narrative across all clips.

For each clip, provide:
1. A narration segment (2-3 sentences) that connects to the overall story
2. Ensure narrative continuity - each segment should flow from the previous

Return ONLY valid JSON:
{{
  "narrative_arc": "Brief description of the overall story",
  "segments": [
    {{"clip_index": 0, "narration": "Segment text for clip 1..."}},
    {{"clip_index": 1, "narration": "Segment text for clip 2..."}},
    {{"clip_index": 2, "narration": "Segment text for clip 3..."}}
  ]
}}"""
        
        # Create message content with frames from each clip
        message_content = [{"type": "text", "text": prompt}]
        
        for i, (clip_path, _) in enumerate(clips):
            message_content.append({
                "type": "text",
                "text": f"\n\n--- Clip {i+1} ---"
            })
            
            # Extract keyframes dynamically (same as regular caption mode: 1 per 2 seconds, 8-20 frames)
            with tempfile.TemporaryDirectory() as tmpdir:
                output_pattern = Path(tmpdir) / f"clip{i}_frame_%02d.jpg"
                
                # Get clip duration
                probe_cmd = [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(clip_path)
                ]
                try:
                    duration = float(subprocess.check_output(probe_cmd).decode().strip())
                except:
                    duration = 30.0
                
                # Dynamic frame count: 1 per 2 seconds, clamped to 8-20 (same as regular captions)
                num_frames = min(max(8, int(duration / 2)), 20)
                interval = max(1, int(duration / num_frames))
                
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(clip_path),
                    "-vf", f"fps=1/{interval}",
                    "-frames:v", str(num_frames),
                    "-q:v", "2",
                    str(output_pattern)
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Add frames to message
                for j, frame_path in enumerate(sorted(Path(tmpdir).glob(f"clip{i}_frame_*.jpg"))):
                    timestamp = (j / num_frames) * duration
                    with open(frame_path, "rb") as f:
                        frame_b64 = base64.b64encode(f.read()).decode("utf-8")
                    
                    message_content.append({
                        "type": "text",
                        "text": f"[Clip {i+1}, {timestamp:.1f}s]"
                    })
                    message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_b64}",
                            "detail": "low"
                        }
                    })
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": message_content}],
            max_completion_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content
        data = json.loads(result_text)
        
        # Parse response into ClipNarration objects
        narrations = []
        for segment in data.get("segments", []):
            idx = segment.get("clip_index", 0)
            text = segment.get("narration", "")
            
            if idx < len(clips):
                clip_path, category = clips[idx]
                narrations.append(ClipNarration(
                    clip_index=idx,
                    narration_text=text,
                    detected_category=category,
                    clip_path=clip_path
                ))
        
        arc = data.get("narrative_arc", "")
        logging.info(f"Generated story arc: {arc}")
        logging.info(f"Created {len(narrations)} narrative segments")
        
        return narrations
        
    except Exception as e:
        logging.error(f"OpenAI story generation failed: {e}")
        return []
