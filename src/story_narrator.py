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


def generate_unified_story(
    clips: List[Tuple[Path, str]],
    story_style: str
) -> List[ClipNarration]:
    """Generate a unified narrative across multiple clips.
    
    Args:
        clips: List of (clip_path, detected_category) tuples
        story_style: One of: story_news, story_roast, story_creepypasta, story_dramatic
        
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
    
    logging.info(f"Generating unified {story_style} narrative for {len(clips)} clips")
    
    provider = os.getenv("AI_PROVIDER", "gemini").lower()
    
    if provider == "openai":
        return _generate_story_openai(clips, story_style)
    elif provider == "local":
        logging.warning("Story mode not available in local mode")
        return []
    else:  # gemini
        return _generate_story_gemini(clips, story_style)


def _get_story_prompt(story_style: str, num_clips: int) -> str:
    """Get the prompt for cross-clip story generation."""
    
    style_guides = {
        "story_news": f"""You are a professional esports broadcaster creating a {num_clips}-part news segment.

Create a cohesive news narrative that flows across all {num_clips} clips:
- Clip 1: Opening/intro to the story
- Middle clips: Development and escalation
- Final clip: Conclusion or recap

Style: Professional, analytical, building excitement like a sports broadcast.
Tone: Clear, confident, measured pace.

Each segment should be 2-3 sentences that connect to form one continuous story.""",

        "story_roast": f"""You are a sarcastic commentator creating a {num_clips}-part roast.

Create a playful roasting narrative that flows across all {num_clips} clips:
- Clip 1: Set up the situation
- Middle clips: Build the comedy/mockery
- Final clip: Deliver the punchline or conclusion

Style: Sarcastic but not mean, comedic timing, playful mockery.
Tone: Amused, slightly condescending but entertaining.

Each segment should be 2-3 sentences that escalate the roast.""",

        "story_creepypasta": f"""You are a horror narrator creating a {num_clips}-part creepypasta.

Create an unsettling narrative that builds tension across all {num_clips} clips:
- Clip 1: Something feels wrong...
- Middle clips: The horror escalates
- Final clip: The terrifying reveal or cliffhanger

Style: Horror/tension building, slow reveals, ominous.
Tone: Deep, deliberate, building dread with ellipses.

Each segment should be 2-3 sentences that build suspense.""",

        "story_dramatic": f"""You are an epic narrator creating a {num_clips}-part cinematic story.

Create a dramatic narrative arc across all {num_clips} clips:
- Clip 1: The stage is set
- Middle clips: The stakes rise
- Final clip: The climactic conclusion

Style: Epic, cinematic, movie-trailer intensity.
Tone: Powerful, inspiring, grand.

Each segment should be 2-3 sentences of epic narration."""
    }
    
    return style_guides.get(story_style, style_guides["story_dramatic"])


def _generate_story_gemini(
    clips: List[Tuple[Path, str]],
    story_style: str
) -> List[ClipNarration]:
    """Generate cross-clip story using Gemini API."""
    
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        logging.warning("GEMINI_API_KEY not set, skipping story mode")
        return []
    
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        
        # Upload all clip videos
        uploaded_files = []
        for clip_path, _ in clips:
            with open(clip_path, "rb") as f:
                video_file = client.files.upload(
                    file=f,
                    config={"mime_type": "video/mp4"}
                )
                uploaded_files.append(video_file)
        
        # Build prompt
        style_guide = _get_story_prompt(story_style, len(clips))
        
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
        
        response = client.models.generate_content(
            model=model_name,
            contents=content_parts
        )
        
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
    story_style: str
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
        style_guide = _get_story_prompt(story_style, len(clips))
        
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
