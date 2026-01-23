"""AI Provider abstraction for semantic video analysis.

This module provides a unified interface for analyzing video clips using
either Gemini or OpenAI APIs. The provider is selected via the AI_PROVIDER
environment variable.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclass
class ClipScore:
    """Represents a scored video clip with semantic analysis."""
    
    clip_path: Path
    original_start: float  # Start time in the original video
    original_end: float    # End time in the original video
    ai_score: float = 0.0  # Semantic score from AI (0-1)
    reason: str = ""       # Why this clip was selected
    heuristic_score: float = 0.0  # Original heuristic score
    detected_category: str = ""   # Detected category: "action", "funny", "highlight"
    category_scores: dict = None  # Scores per category: {"action": 0.8, "funny": 0.3, ...}
    
    def __post_init__(self):
        if self.category_scores is None:
            self.category_scores = {}
    
    @property
    def combined_score(self) -> float:
        """Weighted combination of AI and heuristic scores."""
        ai_weight = float(os.getenv("AI_SCORE_WEIGHT", "0.7"))
        return ai_weight * self.ai_score + (1 - ai_weight) * self.heuristic_score
    
    @property
    def caption_style(self) -> str:
        """Get the appropriate caption style based on detected category."""
        style_map = {
            "action": "gaming",
            "funny": "funny",
            "highlight": "dramatic",
        }
        return style_map.get(self.detected_category, "gaming")



@dataclass
class AnalysisResult:
    """Result from AI analysis containing ranked clips."""
    
    clips: List[ClipScore] = field(default_factory=list)
    provider: str = ""
    raw_response: str = ""


class SemanticAnalyzer(ABC):
    """Abstract base class for semantic video analyzers."""
    
    @abstractmethod
    def analyze_clips(
        self, 
        clip_infos: List[ClipScore], 
        goal: str = "action"
    ) -> AnalysisResult:
        """Analyze video clips and return semantic scores.
        
        Args:
            clip_infos: List of ClipScore objects with clip_path populated
            goal: Analysis goal - "action", "funny", "highlights"
            
        Returns:
            AnalysisResult with updated scores and reasons
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is properly configured."""
        pass


class GeminiAnalyzer(SemanticAnalyzer):
    """Semantic analyzer using Google Gemini API."""
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self._client = None
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                logging.error("google-genai package not installed. Run: pip install google-genai")
                raise
        return self._client
    
    def analyze_clips(
        self, 
        clip_infos: List[ClipScore], 
        goal: str = "action"
    ) -> AnalysisResult:
        """Analyze clips using Gemini's video understanding.
        
        If goal is "mixed", scores each clip for action, funny, and highlight
        categories and picks the best one.
        """
        
        if not self.is_available():
            logging.warning("Gemini API key not configured. Returning unmodified scores.")
            return AnalysisResult(clips=clip_infos, provider="gemini", raw_response="")
        
        client = self._get_client()
        is_mixed = goal.lower() == "mixed"
        
        goal_prompts = {
            "action": "intense action moments, exciting gameplay, skillful plays, or dramatic events",
            "funny": "funny moments, fails, unexpected events, comedic timing, or humorous situations",
            "highlight": "memorable highlights, impressive achievements, or shareable moments"
        }
        
        results = []
        
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _analyze_single_clip_gemini(clip_info):
            try:
                logging.info(f"Analyzing clip: {clip_info.clip_path.name}")
                client = self._get_client()

                # Upload the video file (Gemini handles uploads per request or reuse? For simplicity upload/delete per request)
                # Note: Parallel uploads might hit limits.
                with open(clip_info.clip_path, "rb") as f:
                    video_file = client.files.upload(
                        file=f,
                        config={"mime_type": "video/mp4"}
                    )
                
                # Reconstruct prompt locally
                if is_mixed:
                    prompt = """Analyze this video clip and rate it for THREE categories.

CATEGORIES:
1. ACTION - intense moments, exciting gameplay, skillful plays, combat, close calls
2. FUNNY - fails, unexpected events, comedic timing, humorous situations, glitches
3. HIGHLIGHT - impressive achievements, memorable moments, shareable content

Respond with ONLY valid JSON:
{"action": 0.85, "funny": 0.3, "highlight": 0.6, "best_category": "action", "reason": "Why this clip is best for the selected category"}

Scoring guide (per category):
- 0.0-0.3: Not relevant
- 0.4-0.6: Somewhat relevant
- 0.7-0.8: Good match
- 0.9-1.0: Exceptional
"""
                else:
                    goal_desc = goal_prompts.get(goal, goal_prompts["action"])
                    prompt = f"""Analyze this video clip for {goal_desc}.

Rate this clip on a scale of 0.0 to 1.0 based on how well it matches the goal.

Respond with ONLY valid JSON in this exact format:
{{"score": 0.85, "reason": "Brief explanation of why this clip scored this way"}}

Be strict in your scoring:
- 0.0-0.3: Not relevant to the goal
- 0.4-0.6: Somewhat relevant but not standout
- 0.7-0.8: Good content matching the goal
- 0.9-1.0: Exceptional content, highly shareable
"""
                
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=[video_file, prompt]
                )
                
                response_text = response.text.strip()
                
                try:
                    if "```json" in response_text:
                        response_text = response_text.split("```json")[1].split("```")[0]
                    elif "```" in response_text:
                        response_text = response_text.split("```")[1].split("```")[0]
                    
                    data = json.loads(response_text)
                    
                    if is_mixed:
                        clip_info.category_scores = {
                            "action": float(data.get("action", 0.5)),
                            "funny": float(data.get("funny", 0.5)),
                            "highlight": float(data.get("highlight", 0.5)),
                        }
                        best_cat = data.get("best_category", "").lower()
                        if best_cat not in clip_info.category_scores:
                            best_cat = max(clip_info.category_scores, key=clip_info.category_scores.get)
                        clip_info.detected_category = best_cat
                        clip_info.ai_score = clip_info.category_scores[best_cat]
                        clip_info.reason = data.get("reason", "")
                    else:
                        clip_info.ai_score = float(data.get("score", 0.5))
                        clip_info.reason = data.get("reason", "")
                        clip_info.detected_category = goal
                        
                except json.JSONDecodeError:
                    logging.warning(f"Failed to parse Gemini response: {response_text[:100]}")
                    clip_info.ai_score = 0.5
                    clip_info.reason = "Parse error"
                    clip_info.detected_category = "action"
                
                try:
                    client.files.delete(name=video_file.name)
                except Exception:
                    pass

                return clip_info

            except Exception as e:
                logging.error(f"Error analyzing clip {clip_info.clip_path.name}: {e}")
                clip_info.ai_score = 0.5
                clip_info.reason = f"Error: {str(e)[:50]}"
                return clip_info

        max_workers = 5 # Gemini might be stricter with concurrent uploads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_clip = {
                executor.submit(_analyze_single_clip_gemini, clip): clip 
                for clip in clip_infos
            }
            for future in as_completed(future_to_clip):
                results.append(future.result())
        
        return AnalysisResult(
            clips=results,
            provider="gemini",
            raw_response=""
        )



class OpenAIAnalyzer(SemanticAnalyzer):
    """Semantic analyzer using OpenAI API."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
        self._client = None
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                logging.error("openai package not installed. Run: pip install openai")
                raise
        return self._client
    
    def _extract_frames(self, video_path: Path, num_frames: int = 8) -> List[str]:
        """Extract keyframes from video as base64 images."""
        import base64
        
        frames = []
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract frames using ffmpeg
            output_pattern = Path(tmpdir) / "frame_%03d.jpg"
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vf", f"fps={num_frames}/30", # Extract roughly num_frames over 30s normalized, or just use fps
                "-vframes", str(num_frames),
                "-q:v", "2",
                str(output_pattern)
            ]
            # simplified robust command
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vf", f"fps=1",  # 1 frame per second is plenty
                "-frames:v", str(num_frames),
                "-q:v", "2",
                str(output_pattern)
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Read frames as base64
            for frame_path in sorted(Path(tmpdir).glob("frame_*.jpg")):
                with open(frame_path, "rb") as f:
                    frames.append(base64.b64encode(f.read()).decode("utf-8"))
        
        return frames
    
    def analyze_clips(
        self, 
        clip_infos: List[ClipScore], 
        goal: str = "action"
    ) -> AnalysisResult:
        """Analyze clips using OpenAI's vision API.
        
        If goal is "mixed", scores each clip for action, funny, and highlight
        categories and picks the best one.
        """
        
        if not self.is_available():
            logging.warning("OpenAI API key not configured. Returning unmodified scores.")
            return AnalysisResult(clips=clip_infos, provider="openai", raw_response="")
        
        client = self._get_client()
        is_mixed = goal.lower() == "mixed"
        
        goal_prompts = {
            "action": "intense action moments, exciting gameplay, skillful plays, or dramatic events",
            "funny": "funny moments, fails, unexpected events, comedic timing, or humorous situations",
            "highlight": "memorable highlights, impressive achievements, or shareable moments"
        }
        
        results = []
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Helper for parallel processing
        def _analyze_single_clip(clip_info):
            try:
                logging.info(f"Analyzing clip: {clip_info.clip_path.name}")
                
                # Extract keyframes
                frames = self._extract_frames(clip_info.clip_path, num_frames=8)
                
                if not frames:
                    logging.warning(f"No frames extracted from {clip_info.clip_path}")
                    clip_info.ai_score = 0.5
                    clip_info.reason = "No frames extracted"
                    clip_info.detected_category = "action"
                    return clip_info
                
                # Use same prompt logic as before...
                # Note: We need to reconstruct the prompt inside the thread or pass it
                if is_mixed:
                    prompt_text = """These are keyframes from a video clip. Analyze them for THREE categories.

CATEGORIES:
1. ACTION - intense moments, exciting gameplay, skillful plays, combat, close calls
2. FUNNY - fails, unexpected events, comedic timing, humorous situations, glitches
3. HIGHLIGHT - impressive achievements, memorable moments, shareable content

Respond with ONLY valid JSON:
{"action": 0.85, "funny": 0.3, "highlight": 0.6, "best_category": "action", "reason": "Why this clip is best for the selected category"}

Scoring guide (per category):
- 0.0-0.3: Not relevant
- 0.4-0.6: Somewhat relevant
- 0.7-0.8: Good match
- 0.9-1.0: Exceptional"""
                else:
                    goal_desc = goal_prompts.get(goal, goal_prompts["action"])
                    prompt_text = f"""These are keyframes from a video clip. Analyze them for {goal_desc}.

Rate this clip on a scale of 0.0 to 1.0 based on how well it matches the goal.

Respond with ONLY valid JSON in this exact format:
{{"score": 0.85, "reason": "Brief explanation of why this clip scored this way"}}

Be strict in your scoring:
- 0.0-0.3: Not relevant to the goal
- 0.4-0.6: Somewhat relevant but not standout
- 0.7-0.8: Good content matching the goal
- 0.9-1.0: Exceptional content, highly shareable"""
                
                # Build message with images
                content = [{"type": "text", "text": prompt_text}]
                
                for frame_b64 in frames:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_b64}",
                            "detail": "low"
                        }
                    })
                
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": content}],
                    max_completion_tokens=2000,  # Increased for gpt-5-mini
                    response_format={"type": "json_object"}
                )
                
                response_text = response.choices[0].message.content.strip()
                
                try:
                    if "```json" in response_text:
                        response_text = response_text.split("```json")[1].split("```")[0]
                    elif "```" in response_text:
                        response_text = response_text.split("```")[1].split("```")[0]
                    
                    data = json.loads(response_text)
                    
                    if is_mixed:
                        clip_info.category_scores = {
                            "action": float(data.get("action", 0.5)),
                            "funny": float(data.get("funny", 0.5)),
                            "highlight": float(data.get("highlight", 0.5)),
                        }
                        best_cat = data.get("best_category", "").lower()
                        if best_cat not in clip_info.category_scores:
                            best_cat = max(clip_info.category_scores, key=clip_info.category_scores.get)
                        clip_info.detected_category = best_cat
                        clip_info.ai_score = clip_info.category_scores[best_cat]
                        clip_info.reason = data.get("reason", "")
                    else:
                        clip_info.ai_score = float(data.get("score", 0.5))
                        clip_info.reason = data.get("reason", "")
                        clip_info.detected_category = goal
                        
                except json.JSONDecodeError:
                    logging.warning(f"Failed to parse OpenAI response: {response_text[:100]}")
                    clip_info.ai_score = 0.5
                    clip_info.reason = "Parse error"
                    clip_info.detected_category = "action"
                
                return clip_info

            except Exception as e:
                logging.error(f"Error analyzing clip {clip_info.clip_path.name}: {e}")
                clip_info.ai_score = 0.5
                clip_info.reason = f"Error: {str(e)[:50]}"
                clip_info.detected_category = "action"
                return clip_info

        # Execute in parallel
        # Max concurrency depends on API rate limits. 5-10 is usually safe for tier-1+.
        max_workers = 8
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_clip = {
                executor.submit(_analyze_single_clip, clip): clip 
                for clip in clip_infos
            }
            
            for future in as_completed(future_to_clip):
                result = future.result()
                results.append(result)
        
        # Sort results to match input order? Ideally, but ClipScore has paths we can map back.
        # The calling function handles sorting by score, so order here doesn't matter strictly.
        
        return AnalysisResult(
            clips=results,
            provider="openai",
            raw_response=""
        )



class LocalAnalyzer(SemanticAnalyzer):
    """Fallback analyzer that uses only heuristic scores (no API calls)."""
    
    def is_available(self) -> bool:
        return True
    
    def analyze_clips(
        self, 
        clip_infos: List[ClipScore], 
        goal: str = "action"
    ) -> AnalysisResult:
        """Return clips with AI score set to heuristic score."""
        
        logging.info("Using local-only mode (no AI API calls)")
        
        for clip_info in clip_infos:
            # Use heuristic score as AI score
            clip_info.ai_score = clip_info.heuristic_score
            clip_info.reason = "Local heuristic-based scoring"
            clip_info.detected_category = "action"  # Default category in local mode
        
        return AnalysisResult(
            clips=clip_infos,
            provider="local",
            raw_response=""
        )



def get_analyzer() -> SemanticAnalyzer:
    """Factory function to get the configured semantic analyzer.
    
    Selection is based on AI_PROVIDER environment variable:
    - "gemini" (default): Use Gemini API
    - "openai": Use OpenAI API  
    - "local": Use local heuristic-only mode (no API calls)
    
    Returns:
        SemanticAnalyzer instance
    """
    provider = os.getenv("AI_PROVIDER", "gemini").lower()
    
    if provider == "openai":
        analyzer = OpenAIAnalyzer()
        if not analyzer.is_available():
            logging.warning("OpenAI API key not set. Falling back to local mode.")
            return LocalAnalyzer()
        return analyzer
    
    elif provider == "local":
        return LocalAnalyzer()
    
    else:  # Default to Gemini
        analyzer = GeminiAnalyzer()
        if not analyzer.is_available():
            logging.warning("Gemini API key not set. Falling back to local mode.")
            return LocalAnalyzer()
        return analyzer


def extract_candidate_clip(
    source_path: Path,
    start_time: float,
    duration: float,
    output_path: Path
) -> bool:
    """Extract a clip from source video for AI analysis with GPU acceleration.
    
    Args:
        source_path: Path to source video
        start_time: Start time in seconds
        duration: Duration in seconds
        output_path: Path to save extracted clip
        
    Returns:
        True if extraction succeeded
    """
    # 1. Try GPU Extraction (Fast & Small)
    # Downscale to 640px wide on GPU, encode with NVENC
    gpu_cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_time:.3f}",
        "-t", f"{duration:.3f}",
        "-hwaccel", "cuda",
        "-hwaccel_output_format", "cuda",
        "-i", str(source_path),
        "-vf", "scale_cuda=640:-2",
        "-c:v", "hevc_nvenc",
        "-preset", "fast",     # 'fast' is good enough for analysis
        "-c:a", "aac",
        "-b:a", "96k",         # Low bitrate for analysis
        str(output_path)
    ]
    
    try:
        result = subprocess.run(
            gpu_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if result.returncode == 0 and output_path.exists():
            return True
    except Exception:
        pass # Fallback to CPU

    # 2. CPU Fallback (Reliable but Slow)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_time:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(source_path),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",
        "-vf", "scale=640:-2", # Still downscale for upload speed
        "-c:a", "aac",
        "-b:a", "96k",
        str(output_path)
    ]
    
    result = subprocess.run(
        cmd, 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL
    )
    
    return result.returncode == 0 and output_path.exists()


@dataclass
class Caption:
    """A single caption with timestamp."""
    
    start_time: float  # Start time in seconds
    end_time: float    # End time in seconds
    text: str          # Caption text
    style: str = ""    # Optional style hint (e.g., "emphasis", "action", "humor")


@dataclass
class CaptionResult:
    """Result from AI caption generation."""
    
    captions: List[Caption]
    provider: str
    success: bool = True
    error: str = ""


def generate_ai_captions(
    video_path: Path,
    style: str = "gaming",
    max_captions: int = 8,
) -> CaptionResult:
    """Generate AI-powered captions for a video clip.
    
    Instead of transcribing speech, the AI watches the video and generates
    contextual captions based on the visual content - perfect for gameplay
    videos without voice commentary.
    
    Args:
        video_path: Path to the video clip
        style: Caption style - "gaming", "dramatic", "funny", "minimal"
        max_captions: Maximum number of captions to generate
        
    Returns:
        CaptionResult with list of Caption objects
    """
    provider = os.getenv("AI_PROVIDER", "gemini").lower()
    
    if provider == "openai":
        return _generate_captions_openai(video_path, style, max_captions)
    elif provider == "local":
        logging.warning("AI captions not available in local mode")
        return CaptionResult(captions=[], provider="local", success=False, 
                           error="AI captions require Gemini or OpenAI")
    else:  # Default to Gemini
        return _generate_captions_gemini(video_path, style, max_captions)


def _get_caption_prompt(style: str, max_captions: int, duration: float) -> str:
    """Generate the prompt for AI caption generation."""
    
    style_guides = {
        "gaming": """Generate short, punchy captions like gaming content creators use.
Examples: "HEADSHOT!", "clutch play incoming...", "wait for it...", "GG EZ", "POV: you're cracked"
Keep captions 1-5 words. Use ALL CAPS for emphasis on action moments.""",
        
        "dramatic": """Generate dramatic, cinematic captions.
Examples: "The final stand.", "Everything changed.", "No turning back now."
Keep captions short and impactful. Use lowercase for tension, CAPS for climax.""",
        
        "funny": """Generate humorous, meme-style captions.
Examples: "skill issue tbh", "when the plan works (it never does)", "*chuckles* I'm in danger"
Be self-aware and slightly chaotic. Gen-Z humor welcome.""",
        
        "minimal": """Generate minimal, understated captions.
Examples: "nice.", "oh.", "well then."
Keep it subtle. Less is more. Max 3 words per caption."""
    }
    
    style_guide = style_guides.get(style, style_guides["gaming"])
    
    return f"""Watch this gameplay video and generate {max_captions} short captions for key moments.

STYLE GUIDE:
{style_guide}

VIDEO DURATION: {duration:.1f} seconds

RULES:
1. Space captions throughout the video (not all at the start)
2. Each caption should appear for 1-3 seconds
3. Focus on action moments, close calls, achievements, or funny situations
4. Captions should enhance the viewing experience, not describe obvious actions

Respond with ONLY valid JSON object:
{{
  "captions": [
    {{"start": 1.5, "end": 3.0, "text": "CAPTION TEXT", "style": "action"}},
    {{"start": 5.2, "end": 6.5, "text": "another caption", "style": "humor"}}
  ]
}}

The "style" field should be one of: "action", "humor", "tension", "emphasis", "normal"
"""


def _generate_captions_gemini(
    video_path: Path, 
    style: str, 
    max_captions: int
) -> CaptionResult:
    """Generate captions using Gemini API."""
    
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return CaptionResult(captions=[], provider="gemini", success=False,
                           error="GEMINI_API_KEY not set")
    
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        
        # Get video duration
        duration = _get_video_duration(video_path)
        
        logging.info(f"Generating AI captions for: {video_path.name} ({duration:.1f}s)")
        
        # Upload video
        with open(video_path, "rb") as f:
            video_file = client.files.upload(
                file=f,
                config={"mime_type": "video/mp4"}
            )
        
        prompt = _get_caption_prompt(style, max_captions, duration)
        
        response = client.models.generate_content(
            model=model_name,
            contents=[video_file, prompt]
        )
        
        # Parse response
        response_text = response.text.strip()
        captions = _parse_caption_response(response_text, duration)
        
        # Cleanup uploaded file
        try:
            client.files.delete(name=video_file.name)
        except Exception:
            pass
        
        logging.info(f"Generated {len(captions)} captions")
        return CaptionResult(captions=captions, provider="gemini", success=True)
        
    except Exception as e:
        logging.error(f"Gemini caption generation failed: {e}")
        return CaptionResult(captions=[], provider="gemini", success=False,
                           error=str(e))


def _generate_captions_openai(
    video_path: Path, 
    style: str, 
    max_captions: int
) -> CaptionResult:
    """Generate captions using OpenAI API (via keyframe extraction)."""
    
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return CaptionResult(captions=[], provider="openai", success=False,
                           error="OPENAI_API_KEY not set")
    
    try:
        from openai import OpenAI
        import base64
        
        client = OpenAI(api_key=api_key)
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
        
        # Get video duration
        duration = _get_video_duration(video_path)
        
        logging.info(f"Generating AI captions for: {video_path.name} ({duration:.1f}s)")
        
        # Extract more frames for caption timing (1 per 2 seconds)
        num_frames = min(max(8, int(duration / 2)), 20)
        
        frames = []
        with tempfile.TemporaryDirectory() as tmpdir:
            output_pattern = Path(tmpdir) / "frame_%03d.jpg"
            interval = max(1, int(duration / num_frames))
            
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vf", f"fps=1/{interval}",
                "-frames:v", str(num_frames),
                "-q:v", "2",
                str(output_pattern)
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            for frame_path in sorted(Path(tmpdir).glob("frame_*.jpg")):
                with open(frame_path, "rb") as f:
                    frames.append(base64.b64encode(f.read()).decode("utf-8"))
        
        if not frames:
            return CaptionResult(captions=[], provider="openai", success=False,
                               error="Failed to extract frames")
        
        # Build message
        prompt = _get_caption_prompt(style, max_captions, duration)
        prompt += f"\n\nThese {len(frames)} frames are evenly spaced across the {duration:.1f}s video."
        
        content = [{"type": "text", "text": prompt}]
        for i, frame_b64 in enumerate(frames):
            timestamp = (i / len(frames)) * duration
            content.append({
                "type": "text",
                "text": f"[Frame at {timestamp:.1f}s]"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_b64}",
                    "detail": "low"
                }
            })
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        response_text = response.choices[0].message.content
        
        # Debug logging for empty responses
        if not response_text or response_text.strip() == "":
            logging.warning(f"OpenAI returned empty response. Finish reason: {response.choices[0].finish_reason}")
            return CaptionResult(captions=[], provider="openai", success=False,
                               error="Empty response from model")
        
        response_text = response_text.strip()
        captions = _parse_caption_response(response_text, duration)
        
        logging.info(f"Generated {len(captions)} captions")
        return CaptionResult(captions=captions, provider="openai", success=True)
        
    except Exception as e:
        logging.error(f"OpenAI caption generation failed: {e}")
        return CaptionResult(captions=[], provider="openai", success=False,
                           error=str(e))


def _parse_caption_response(response_text: str, duration: float) -> List[Caption]:
    """Parse AI response into Caption objects."""
    
    captions = []
    
    try:
        # Handle markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        data = json.loads(response_text.strip())
        
        if isinstance(data, dict):
            # Handle {"captions": [...]} format
            if "captions" in data and isinstance(data["captions"], list):
                data = data["captions"]
            else:
                # Fallback if it's a single caption object or just flat dict
                data = [data]
        elif not isinstance(data, list):
            data = [data]
        
        for item in data:
            start = float(item.get("start", 0))
            end = float(item.get("end", start + 2))
            text = str(item.get("text", "")).strip()
            style = str(item.get("style", "normal"))
            
            # Validate timing
            if start < 0:
                start = 0
            if end > duration:
                end = duration
            if end <= start:
                end = min(start + 2, duration)
            
            if text:
                captions.append(Caption(
                    start_time=start,
                    end_time=end,
                    text=text,
                    style=style
                ))
        
        # Sort by start time
        captions.sort(key=lambda c: c.start_time)
        
    except json.JSONDecodeError as e:
        logging.warning(f"Failed to parse caption JSON: {e}")
        logging.warning(f"Raw Response: {response_text}")  # Changed to warning to ensure visibility
    
    if not captions:
        logging.warning(f"Parsed 0 captions. Raw Response: {response_text}")
        
    return captions


def _get_video_duration(video_path: Path) -> float:
    """Get video duration using ffprobe."""
    
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 30.0  # Default fallback


def captions_to_srt(captions: List[Caption], output_path: Path) -> Path:
    """Convert Caption list to SRT file format.
    
    Args:
        captions: List of Caption objects
        output_path: Path to save SRT file
        
    Returns:
        Path to the created SRT file
    """
    srt_lines = []
    
    for i, caption in enumerate(captions, 1):
        start_ts = _seconds_to_srt_time(caption.start_time)
        end_ts = _seconds_to_srt_time(caption.end_time)
        
        srt_lines.append(str(i))
        srt_lines.append(f"{start_ts} --> {end_ts}")
        srt_lines.append(caption.text)
        srt_lines.append("")  # Blank line after text
        
    with open(output_path, "w", encoding="utf-8") as f:
        # Join with newline.
        # Format becomes:
        # 1
        # 00:00:01 --> 00:00:02
        # Text
        # (blank line)
        f.write("\n".join(srt_lines))
    
    return output_path


def _seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# =============================================================================
# LOCAL AI TAGGING SYSTEM
# Replaces PyCaps paid AI features using your existing OpenAI/Gemini API
# =============================================================================

@dataclass
class TaggedWord:
    """A word with semantic tags and optional emoji."""
    
    word: str
    tags: List[str]  # e.g., ["action", "emphasis"]
    emoji: str = ""  # Optional emoji to add
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass 
class SemanticTagResult:
    """Result from AI semantic tagging."""
    
    tagged_words: List[TaggedWord]
    action_words: List[str]      # Words tagged as action
    emphasis_words: List[str]    # Words to emphasize
    emotion_words: List[str]     # Emotional words
    product_words: List[str]     # Product names, game names
    emoji_map: dict              # word -> emoji mapping
    success: bool = True
    error: str = ""


def get_semantic_tagging_prompt(text: str, category: str = "gaming") -> str:
    """Generate prompt for semantic tagging based on content category."""
    
    category_context = {
        "gaming": "gaming/esports content with action, kills, victories, fails",
        "funny": "comedy content with punchlines, fails, unexpected moments", 
        "dramatic": "dramatic/cinematic content with tension, reveals, climax",
        "action": "action-packed content with intense moments, combat, speed"
    }
    
    context = category_context.get(category, category_context["gaming"])
    
    return f"""Analyze this text from a video caption and tag words for styling.
    
TEXT: "{text}"

CONTENT TYPE: {context}

Tag each significant word with ONE OR MORE of these categories:
- action: Action words (kill, win, clutch, dodge, strike, hit, boom)  
- emphasis: Words that should be emphasized/highlighted (important moments)
- emotion: Emotional words (wow, crazy, insane, legendary, epic)
- product: Product names, game names, character names, brands
- number: Numbers, scores, counts (50, triple, first)

Also suggest relevant emojis (max 2 per caption).

Respond with ONLY valid JSON:
{{
    "tagged_words": [
        {{"word": "CLUTCH", "tags": ["action", "emphasis"], "emoji": ""}},
        {{"word": "win", "tags": ["action"], "emoji": ""}},
        {{"word": "insane", "tags": ["emotion", "emphasis"], "emoji": ""}}
    ],
    "suggested_emojis": ["🔥", "💀"],
    "emoji_placement": "after"
}}

RULES:
1. Only tag important words (skip common words like "the", "a", "is")
2. Max 4-5 tagged words per caption
3. Emojis should match the content energy
4. Be selective - not every word needs a tag
"""


def tag_caption_semantically(
    caption_text: str,
    category: str = "gaming",
    provider: str = None
) -> SemanticTagResult:
    """Tag a caption's words with semantic categories using AI.
    
    This is a LOCAL implementation that replaces PyCaps paid AI tagging.
    Uses your existing OpenAI/Gemini API keys.
    
    Args:
        caption_text: The caption text to analyze
        category: Content category for context ("gaming", "funny", "dramatic")
        provider: AI provider to use (None = use AI_PROVIDER env var)
        
    Returns:
        SemanticTagResult with tagged words and categorized lists
    """
    if provider is None:
        provider = os.getenv("AI_PROVIDER", "gemini").lower()
    
    # Skip AI call for very short captions
    if len(caption_text.split()) <= 2:
        return SemanticTagResult(
            tagged_words=[],
            action_words=[],
            emphasis_words=[],
            emotion_words=[],
            product_words=[],
            emoji_map={},
            success=True
        )
    
    prompt = get_semantic_tagging_prompt(caption_text, category)
    
    try:
        if provider == "openai":
            result = _tag_with_openai(prompt)
        else:
            result = _tag_with_gemini(prompt)
        
        return result
        
    except Exception as e:
        logging.warning(f"Semantic tagging failed: {e}")
        return SemanticTagResult(
            tagged_words=[],
            action_words=[],
            emphasis_words=[],
            emotion_words=[],
            product_words=[],
            emoji_map={},
            success=False,
            error=str(e)
        )


def _tag_with_gemini(prompt: str) -> SemanticTagResult:
    """Use Gemini API for semantic tagging."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    
    from google import genai
    client = genai.Client(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt]
    )
    
    return _parse_tagging_response(response.text)


def _tag_with_openai(prompt: str) -> SemanticTagResult:
    """Use OpenAI API for semantic tagging."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Use cheaper model for tagging
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=2000,  # Increased for gpt-5-mini
        response_format={"type": "json_object"}
    )
    
    return _parse_tagging_response(response.choices[0].message.content)


def _parse_tagging_response(response_text: str) -> SemanticTagResult:
    """Parse the AI tagging response."""
    response_text = response_text.strip()
    
    # Clean up JSON
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0]
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0]
    
    data = json.loads(response_text)
    
    tagged_words = []
    action_words = []
    emphasis_words = []
    emotion_words = []
    product_words = []
    emoji_map = {}
    
    for item in data.get("tagged_words", []):
        word = item.get("word", "")
        tags = item.get("tags", [])
        emoji = item.get("emoji", "")
        
        if word:
            tagged_words.append(TaggedWord(word=word, tags=tags, emoji=emoji))
            
            if "action" in tags:
                action_words.append(word)
            if "emphasis" in tags:
                emphasis_words.append(word)
            if "emotion" in tags:
                emotion_words.append(word)
            if "product" in tags:
                product_words.append(word)
            if emoji:
                emoji_map[word.lower()] = emoji
    
    # Add suggested emojis to the map
    suggested_emojis = data.get("suggested_emojis", [])
    if suggested_emojis:
        emoji_map["__caption_emojis__"] = suggested_emojis
    
    return SemanticTagResult(
        tagged_words=tagged_words,
        action_words=action_words,
        emphasis_words=emphasis_words,
        emotion_words=emotion_words,
        product_words=product_words,
        emoji_map=emoji_map,
        success=True
    )


def batch_tag_captions(
    captions: List[Caption],
    category: str = "gaming"
) -> dict:
    """Batch process multiple captions for semantic tagging.
    
    More efficient than tagging one at a time.
    
    Args:
        captions: List of Caption objects
        category: Content category
        
    Returns:
        Dict mapping caption text -> SemanticTagResult
    """
    results = {}
    
    # Combine captions for single API call (more efficient)
    all_text = "\n".join([f"{i+1}. {c.text}" for i, c in enumerate(captions)])
    
    prompt = f"""Analyze these video captions and tag important words for each.

CAPTIONS:
{all_text}

CONTENT TYPE: {category} content

For each caption, identify:
- action: Action words (verbs, combat, movement)
- emphasis: Words to highlight (important moments)
- emotion: Emotional words (reactions, exclamations)
- product: Names (games, characters, brands)

Also suggest 1-2 relevant emojis per caption.

Respond with ONLY valid JSON:
{{
    "captions": [
        {{
            "index": 1,
            "tagged_words": [{{"word": "CLUTCH", "tags": ["action", "emphasis"]}}],
            "emojis": ["🔥"]
        }}
    ]
}}
"""
    
    provider = os.getenv("AI_PROVIDER", "gemini").lower()
    
    try:
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            # Use the dedicated tagging model if set, otherwise fallback
            tagging_model = os.getenv("OPENAI_TAGGING_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
            
            response = client.chat.completions.create(
                model=tagging_model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=4096,
                response_format={"type": "json_object"}
            )
            
            # Check for non-success finish reasons
            finish_reason = response.choices[0].finish_reason
            if finish_reason != "stop":
                logging.warning(f"OpenAI batch tagging finished with reason: {finish_reason}")
                
            response_text = response.choices[0].message.content
            
            if not response_text:
                logging.error(f"OpenAI returned empty content. Finish reason: {finish_reason}")
                return {}
                
        else:
            api_key = os.getenv("GEMINI_API_KEY", "")
            from google import genai
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
                contents=[prompt]
            )
            response_text = response.text
            
            # Additional safety check for Gemini
            if not response_text and hasattr(response, 'candidates') and response.candidates:
                finish_reason = getattr(response.candidates[0], 'finish_reason', 'UNKNOWN')
                logging.warning(f"Gemini returned empty text. Finish reason: {finish_reason}")
        
        # Parse response
        if not response_text or not response_text.strip():
             logging.error("Received empty response text from AI provider")
             return {}

        response_text = response_text.strip()
        
        # enhanced debug logging for failed parsing
        if debug_mode := os.getenv("DEBUG_TAGGING"):
             logging.info(f"Raw Tagging Response: {response_text}")

        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as e:
            logging.error(f"Batch tagging JSON decode failed: {e}")
            logging.error(f"Faulty Response Content: '{response_text}'")
            return {}
        
        for item in data.get("captions", []):
            idx = item.get("index", 1) - 1
            if 0 <= idx < len(captions):
                caption = captions[idx]
                
                tagged = []
                action_words = []
                emphasis_words = []
                emotion_words = []
                product_words = []
                
                for tw in item.get("tagged_words", []):
                    word = tw.get("word", "")
                    tags = tw.get("tags", [])
                    if word:
                        tagged.append(TaggedWord(word=word, tags=tags))
                        if "action" in tags:
                            action_words.append(word)
                        if "emphasis" in tags:
                            emphasis_words.append(word)
                        if "emotion" in tags:
                            emotion_words.append(word)
                        if "product" in tags:
                            product_words.append(word)
                
                emojis = item.get("emojis", [])
                emoji_map = {"__caption_emojis__": emojis} if emojis else {}
                
                results[caption.text] = SemanticTagResult(
                    tagged_words=tagged,
                    action_words=action_words,
                    emphasis_words=emphasis_words,
                    emotion_words=emotion_words,
                    product_words=product_words,
                    emoji_map=emoji_map,
                    success=True
                )
        
        logging.info(f"Batch tagged {len(results)} captions")
        
    except Exception as e:
        logging.error(f"Batch tagging failed with unexpected error: {e}")
        # Only log full response if we have it and haven't logged it yet
        if 'response_text' in locals() and response_text:
            logging.error(f"Response text involved in error: {response_text[:200]}...")
    
    return results


def apply_tags_to_pycaps(
    captions: List[Caption],
    tag_results: dict,
    config_highlight_color: str = "#00ff88"
) -> tuple:
    """Generate PyCaps-compatible CSS and word lists from AI tags.
    
    This allows you to use free PyCaps wordlist rules with AI-generated tags.
    
    Args:
        captions: Original captions
        tag_results: Dict from batch_tag_captions()
        config_highlight_color: Color for highlighted words
        
    Returns:
        Tuple of (custom_css, word_lists_dict)
        word_lists_dict can be passed to pycaps SemanticTagger.add_wordlist_rule()
    """
    all_action = set()
    all_emphasis = set()
    all_emotion = set()
    all_product = set()
    
    for text, result in tag_results.items():
        all_action.update(result.action_words)
        all_emphasis.update(result.emphasis_words)
        all_emotion.update(result.emotion_words)
        all_product.update(result.product_words)
    
    # Generate custom CSS for each tag type
    custom_css = f"""
/* Action words - bold and colored */
.tag-action {{
    color: #FF4444 !important;
    font-weight: 900;
    text-transform: uppercase;
}}

/* Emphasis words - highlighted background */
.tag-emphasis {{
    background-color: {config_highlight_color};
    color: #FFFFFF !important;
    padding: 2px 6px;
    border-radius: 4px;
}}

/* Emotion words - slightly larger */
.tag-emotion {{
    color: #FFD700 !important;
    font-size: 110%;
}}

/* Product/game names - distinct styling */
.tag-product {{
    color: #00BFFF !important;
    font-style: italic;
}}
"""
    
    word_lists = {
        "action": list(all_action),
        "emphasis": list(all_emphasis),
        "emotion": list(all_emotion),
        "product": list(all_product)
    }
    
    return custom_css, word_lists


def add_emojis_to_caption(caption_text: str, tag_result: SemanticTagResult) -> str:
    """Add AI-suggested emojis to a caption.
    
    Args:
        caption_text: Original caption text
        tag_result: Semantic tagging result with emoji suggestions
        
    Returns:
        Caption text with emojis added
    """
    emojis = tag_result.emoji_map.get("__caption_emojis__", [])
    
    if not emojis:
        return caption_text
    
    # Add emojis at the end (or customize placement)
    emoji_str = " ".join(emojis[:2])  # Max 2 emojis
    return f"{caption_text} {emoji_str}"


def enhance_captions_with_ai(
    captions: List[Caption],
    category: str = "gaming",
    add_emojis: bool = True
) -> List[Caption]:
    """Enhance captions with AI-powered emojis and return tagged info.
    
    This is the main entry point for local AI caption enhancement.
    
    Args:
        captions: Original Caption list
        category: Content category
        add_emojis: Whether to add emoji suggestions
        
    Returns:
        Enhanced Caption list with emojis added (if enabled)
    """
    if not captions:
        return captions
    
    enable_enhancement = os.getenv("ENABLE_AI_CAPTION_ENHANCEMENT", "true").lower() in ("true", "1", "yes")
    if not enable_enhancement:
        return captions
    
    logging.info(f"Enhancing {len(captions)} captions with AI tagging...")
    
    # Batch process all captions
    tag_results = batch_tag_captions(captions, category)
    
    enhanced = []
    for caption in captions:
        result = tag_results.get(caption.text)
        
        if result and add_emojis:
            new_text = add_emojis_to_caption(caption.text, result)
            enhanced.append(Caption(
                start_time=caption.start_time,
                end_time=caption.end_time,
                text=new_text,
                style=caption.style
            ))
        else:
            enhanced.append(caption)
    
    return enhanced

