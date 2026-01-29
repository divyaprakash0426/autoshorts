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
    
    # Semantic types for clip classification
    SEMANTIC_TYPES = {
        "action": "Combat, kills, intense gameplay, close calls",
        "funny": "Fails, glitches, unexpected humor, comedic timing",
        "clutch": "1vX situations, comebacks, last-second wins",
        "wtf": "Unexpected events, 'wait what?' moments, random chaos",
        "epic_fail": "Embarrassing deaths, tragic blunders, game-losing mistakes",
        "hype": "Celebrations, 'LET'S GO' energy, peak excitement",
        "skill": "Trick shots, IQ plays, advanced mechanics, impressive techniques",
    }
    
    # Map semantic types to caption styles
    CAPTION_STYLE_MAP = {
        "action": "gaming",
        "funny": "funny",
        "clutch": "dramatic",
        "wtf": "funny",
        "epic_fail": "funny",
        "hype": "gaming",
        "skill": "dramatic",
    }
    
    # Map semantic types to TTS voice presets (fallback)
    VOICE_PRESET_MAP = {
        "action": """gender: Male.
pitch: Mid-range male pitch with sharp upward inflections during exciting moments.
speed: Very fast-paced, rapid-fire delivery matching gaming action.
volume: Loud and projecting, nearly shouting during intense plays.
age: Young adult, 20s to early 30s.
clarity: Highly articulate, every word distinct even at speed.
fluency: Extremely fluent with no hesitations, continuous flow.
accent: American English, neutral.
texture: Bright, energetic vocal quality with slight rasp.
emotion: Intense excitement, hype, constant enthusiasm.
tone: Upbeat, authoritative, commanding attention.
personality: Confident, extroverted, engaging, competitive edge.""",
        
        "funny": """gender: Male.
pitch: Mid to slightly high male pitch with playful variations.
speed: Moderate pace with deliberate pauses for comedic timing.
volume: Conversational, occasionally louder for punchlines.
age: Young adult, early to mid 20s.
clarity: Clear but relaxed, not overly precise.
fluency: Fluent with intentional hesitations for humor, occasional 'uh', 'like'.
accent: American English, casual GenZ cadence.
texture: Smooth, light vocal quality with natural warmth.
emotion: Amused, ironic, playfully sarcastic.
tone: Laid-back, chill, slightly deadpan with smirk energy.
personality: Witty, self-aware, relatable, gently mocking.""",
        
        "clutch": """gender: Male.
pitch: Starting low and tense, building to high excited peaks.
speed: Starts slow and deliberate, accelerates rapidly to explosive climax.
volume: Begins quiet, builds to shouting, dramatic crescendo.
age: Middle-aged adult, 30s to 40s.
clarity: Crystal clear pronunciation, every syllable emphasized.
fluency: Perfect fluency with strategic pauses for dramatic tension.
accent: American English, sports broadcaster style.
texture: Rich, resonant voice with powerful projection.
emotion: Building anticipation transitioning to explosive celebration.
tone: Dramatic, tense, then triumphant and victorious.
personality: Professional yet passionate, masterful tension-builder.""",
        
        "wtf": """gender: Male.
pitch: Erratic pitch with sudden jumps and confused inflections.
speed: Irregular pacing, speeding up in disbelief, pausing in confusion.
volume: Moderate with sudden loud outbursts of surprise.
age: Young adult, mid 20s.
clarity: Clear but broken by genuine bewilderment.
fluency: Interrupted flow, hesitations, stammering from shock.
accent: American English, casual conversational.
texture: Light texture with air of disbelief.
emotion: Confused, bewildered, amused by absurdity.
tone: 'Wait what?', questioning everything, perplexed.
personality: Reactive, expressive, openly confused but entertained.""",
        
        "epic_fail": """gender: Male.
pitch: Mid-range descending to low defeated tones.
speed: Slow, heavy delivery with resigned sighs.
volume: Starts normal, drops to quiet disappointment.
age: Young adult to middle-aged, late 20s to 30s.
clarity: Very clear, articulating the tragedy dramatically.
fluency: Smooth but with emotional pauses and sighs.
accent: American English, neutral.
texture: Warm but tinged with sympathetic sadness.
emotion: Disappointed, sympathetic, comedically tragic.
tone: Dramatic despair mixed with humor, mock-serious.
personality: Empathetic, understanding, finding humor in failure.""",
        
        "hype": """gender: Male.
pitch: High energy male pitch, constantly elevated and excited.
speed: Extremely fast, breathless with adrenaline.
volume: Maximum volume, full-throated yelling and celebration.
age: Young adult, early 20s.
clarity: Surprisingly clear despite shouting intensity.
fluency: Non-stop flow, words tumbling out in excitement.
accent: American English, urban hype energy.
texture: Raw, unfiltered energy with slight hoarseness.
emotion: Pure unbridled excitement, celebration, joy.
tone: 'LET'S GOOO', maximalist hype, infectious energy.
personality: Extroverted to the extreme, contagiously enthusiastic.""",
        
        "skill": """gender: Male.
pitch: Low to mid-range, stable and controlled.
speed: Measured, deliberate pace allowing appreciation.
volume: Clear and projected, professional level.
age: Middle-aged adult, 35 to 45.
clarity: Impeccable articulation, refined speech.
fluency: Flawless flow, sophisticated vocabulary.
accent: British English, educated and polished.
texture: Smooth, cultured vocal quality with depth.
emotion: Impressed, appreciative, respectful admiration.
tone: Analytical yet enthusiastic, thoughtful praise.
personality: Intelligent, observant, genuine appreciation for craft.""",
    }
    
    # Map caption styles to TTS voice presets (for direct style selection)
    CAPTION_STYLE_VOICE_MAP = {
        "gaming": """gender: Male.
pitch: Mid-range male pitch with sharp upward inflections during exciting moments.
speed: Very fast-paced, rapid-fire delivery matching gaming action.
volume: Loud and projecting, nearly shouting during intense plays.
age: Young adult, 20s to early 30s.
clarity: Highly articulate, every word distinct even at speed.
fluency: Extremely fluent with no hesitations, continuous flow.
accent: American English, neutral.
texture: Bright, energetic vocal quality with slight rasp.
emotion: Intense excitement, hype, constant enthusiasm.
tone: Upbeat, authoritative, commanding attention.
personality: Confident, extroverted, engaging, competitive edge.""",
        
        "dramatic": """gender: Male.
pitch: Deep, resonant bass with powerful projection.
speed: Slow, deliberate pacing with strategic pauses for impact.
volume: Loud, commanding presence filling the space.
age: Mature adult, 40s to 50s.
clarity: Perfect diction, every word pronounced with gravitas.
fluency: Flawless delivery with cinematic timing.
accent: American English, neutral broadcast quality.
texture: Rich, velvety depth with cinematic warmth.
emotion: Inspiring, epic, grandiose.
tone: Heroic, momentous, like narrating legends.
personality: Authoritative, wise, larger-than-life presence.""",
        
        "funny": """gender: Male.
pitch: Mid to slightly high male pitch with playful variations.
speed: Moderate pace with deliberate pauses for comedic timing.
volume: Conversational, occasionally louder for punchlines.
age: Young adult, early to mid 20s.
clarity: Clear but relaxed, not overly precise.
fluency: Fluent with intentional hesitations for humor, occasional 'uh', 'like'.
accent: American English, casual GenZ cadence.
texture: Smooth, light vocal quality with natural warmth.
emotion: Amused, ironic, playfully sarcastic.
tone: Laid-back, chill, slightly deadpan with smirk energy.
personality: Witty, self-aware, relatable, gently mocking.""",
        
        "minimal": """gender: Male.
pitch: Low, steady pitch with minimal variation.
speed: Slow to moderate, unhurried and calm.
volume: Quiet to moderate, intimate and close.
age: Young to middle-aged adult, 25 to 35.
clarity: Clear but understated, effortless articulation.
fluency: Smooth and easy, natural flow.
accent: American English, neutral and unassuming.
texture: Soft, gentle vocal quality without harshness.
emotion: Calm, composed, subtly confident.
tone: Understated, reserved, quiet assurance.
personality: Introverted, thoughtful, self-contained.""",
        
        "genz": """gender: Male or Female (androgynous lean).
pitch: Mid-range with frequent upward inflections and vocal fry.
speed: Fast with casual slurring, modern speech patterns.
volume: Moderate, conversational social media energy.
age: Late teens to early 20s, Gen Z demographic.
clarity: Casual clarity, some words blend together naturally.
fluency: Very fluent but with filler words, 'literally', 'like', 'bruh'.
accent: American English, internet-influenced speech.
texture: Bright, youthful, slightly nasal quality.
emotion: Ironic detachment mixed with genuine enthusiasm.
tone: Casual, meme-aware, chronically online vibes.
personality: Self-aware, ironic, effortlessly cool, relatable chaos.""",
        
        "story_news": """gender: Female.
pitch: Mid-range female pitch, professional and steady.
speed: Moderate, measured pace with precise timing.
volume: Clear, projected, broadcast-quality loudness.
age: Middle-aged adult, 35 to 45.
clarity: Impeccable enunciation, television-standard precision.
fluency: Flawless professional delivery, no hesitations.
accent: American English, neutral broadcast accent.
texture: Polished, refined vocal quality with authority.
emotion: Objective, serious, professionally engaged.
tone: Journalistic, factual, trustworthy authority.
personality: Professional, composed, credible, commanding respect.""",
        
        "story_roast": """gender: Male.
pitch: Mid to high pitch with sarcastic inflections and exaggerated tones.
speed: Variable, speeding up for punchlines, slowing for emphasis.
volume: Moderate to loud, performative and theatrical.
age: Young adult, mid to late 20s.
clarity: Very clear, ensuring every barb lands perfectly.
fluency: Smooth with comedic pauses and timing.
accent: American English, comedy podcast energy.
texture: Bright with playful edge, slight smirk audible.
emotion: Amused mockery, playful cruelty, entertained.
tone: Sarcastic, teasing, roast-comedy style.
personality: Quick-witted, sharp-tongued, charismatic instigator.""",
        
        "story_creepypasta": """gender: Male.
pitch: Deep, low pitch with minimal variation, ominous undertones.
speed: Very slow, deliberate, each word drawn out for tension.
volume: Quiet to moderate, intimate and unsettling closeness.
age: Middle-aged to older adult, 40s to 50s.
clarity: Crystal clear whisper-level articulation.
fluency: Controlled, methodical pacing building dread.
accent: American English, neutral with slight rasp.
texture: Dark, gravelly with shadowy depth.
emotion: Foreboding, sinister, building unease.
tone: Ominous, creeping horror, inevitable dread.
personality: Mysterious, unsettling, knows something you don't.""",
        
        "story_dramatic": """gender: Male.
pitch: Deep, resonant bass with powerful projection.
speed: Slow, deliberate pacing with strategic pauses for impact.
volume: Loud, commanding presence filling the space.
age: Mature adult, 40s to 50s.
clarity: Perfect diction, every word pronounced with gravitas.
fluency: Flawless delivery with cinematic timing.
accent: American English, neutral broadcast quality.
texture: Rich, velvety depth with cinematic warmth.
emotion: Inspiring, epic, grandiose.
tone: Heroic, momentous, like narrating legends.
personality: Authoritative, wise, larger-than-life presence.""",
    }
    
    clip_path: Path
    original_start: float  # Start time in the original video
    original_end: float    # End time in the original video
    ai_score: float = 0.0  # Semantic score from AI (0-1)
    reason: str = ""       # Why this clip was selected
    heuristic_score: float = 0.0  # Original heuristic score
    detected_category: str = ""   # Detected category (one of SEMANTIC_TYPES keys)
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
        return self.CAPTION_STYLE_MAP.get(self.detected_category, "gaming")
    
    @property
    def voice_preset(self) -> str:
        """Get the fallback voice preset based on detected category."""
        return self.VOICE_PRESET_MAP.get(self.detected_category, self.VOICE_PRESET_MAP["action"])



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
        goal: str = "mixed"
    ) -> AnalysisResult:
        """Analyze clips using Gemini's video understanding.
        
        Always analyzes all semantic types and returns the best match.
        The goal parameter is ignored (kept for API compatibility).
        """
        
        if not self.is_available():
            logging.warning("Gemini API key not configured. Returning unmodified scores.")
            return AnalysisResult(clips=clip_infos, provider="gemini", raw_response="")
        
        client = self._get_client()
        
        # Build category descriptions from ClipScore
        categories_desc = "\n".join(
            f"{i+1}. {cat.upper()} - {desc}"
            for i, (cat, desc) in enumerate(ClipScore.SEMANTIC_TYPES.items())
        )
        categories_json = ", ".join(f'"{cat}": 0.5' for cat in ClipScore.SEMANTIC_TYPES.keys())
        
        results = []
        
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _analyze_single_clip_gemini(clip_info):
            try:
                logging.info(f"Analyzing clip: {clip_info.clip_path.name}")
                client = self._get_client()

                with open(clip_info.clip_path, "rb") as f:
                    video_file = client.files.upload(
                        file=f,
                        config={"mime_type": "video/mp4"}
                    )
                
                prompt = f"""Analyze this video clip and rate it for ALL categories.

CATEGORIES:
{categories_desc}

Respond with ONLY valid JSON:
{{{categories_json}, "best_category": "action", "reason": "Why this clip fits the selected category"}}

Scoring guide (per category):
- 0.0-0.3: Not relevant
- 0.4-0.6: Somewhat relevant
- 0.7-0.8: Good match
- 0.9-1.0: Exceptional
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
                    
                    # Parse scores for all semantic types
                    clip_info.category_scores = {
                        cat: float(data.get(cat, 0.5))
                        for cat in ClipScore.SEMANTIC_TYPES.keys()
                    }
                    best_cat = data.get("best_category", "").lower()
                    if best_cat not in clip_info.category_scores:
                        best_cat = max(clip_info.category_scores, key=clip_info.category_scores.get)
                    clip_info.detected_category = best_cat
                    clip_info.ai_score = clip_info.category_scores[best_cat]
                    clip_info.reason = data.get("reason", "")
                        
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
        goal: str = "mixed"
    ) -> AnalysisResult:
        """Analyze clips using OpenAI's vision API.
        
        Always analyzes all semantic types and returns the best match.
        The goal parameter is ignored (kept for API compatibility).
        """
        
        if not self.is_available():
            logging.warning("OpenAI API key not configured. Returning unmodified scores.")
            return AnalysisResult(clips=clip_infos, provider="openai", raw_response="")
        
        client = self._get_client()
        
        # Build category descriptions from ClipScore
        categories_desc = "\n".join(
            f"{i+1}. {cat.upper()} - {desc}"
            for i, (cat, desc) in enumerate(ClipScore.SEMANTIC_TYPES.items())
        )
        categories_json = ", ".join(f'"{cat}": 0.5' for cat in ClipScore.SEMANTIC_TYPES.keys())
        
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
                
                prompt_text = f"""These are keyframes from a video clip. Analyze them for ALL categories.

CATEGORIES:
{categories_desc}

Respond with ONLY valid JSON:
{{{categories_json}, "best_category": "action", "reason": "Why this clip fits the selected category"}}

Scoring guide (per category):
- 0.0-0.3: Not relevant
- 0.4-0.6: Somewhat relevant
- 0.7-0.8: Good match
- 0.9-1.0: Exceptional"""
                
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
                    max_completion_tokens=2000,
                    response_format={"type": "json_object"}
                )
                
                response_text = response.choices[0].message.content.strip()
                
                try:
                    if "```json" in response_text:
                        response_text = response_text.split("```json")[1].split("```")[0]
                    elif "```" in response_text:
                        response_text = response_text.split("```")[1].split("```")[0]
                    
                    data = json.loads(response_text)
                    
                    # Parse scores for all semantic types
                    clip_info.category_scores = {
                        cat: float(data.get(cat, 0.5))
                        for cat in ClipScore.SEMANTIC_TYPES.keys()
                    }
                    best_cat = data.get("best_category", "").lower()
                    if best_cat not in clip_info.category_scores:
                        best_cat = max(clip_info.category_scores, key=clip_info.category_scores.get)
                    clip_info.detected_category = best_cat
                    clip_info.ai_score = clip_info.category_scores[best_cat]
                    clip_info.reason = data.get("reason", "")
                        
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
    
    # Story modes use fewer but longer captions
    is_story_mode = style.startswith("story_")
    if is_story_mode:
        max_captions = min(max_captions, 4)  # Cap at 4 for story modes
    
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
Keep it subtle. Less is more. Max 3 words per caption.""",
        
        "genz": """Generate GenZ slang-heavy reactions and commentary.
Examples: "bruh 💀", "no cap this is insane", "he's locked in rn", "finna go crazy", "ate and left no crumbs"
Use modern slang naturally: bruh, no cap, finna, fr fr, ate, slay, lowkey, highkey, bussin, mid, L, W, rizz
Keep captions 1-6 words. Be authentic to GenZ internet culture. Emojis encouraged (💀🔥😭).""",
        
        "story_news": """Generate professional esports broadcaster narrative.
Examples: "And we're witnessing championship-level gameplay here.", "The positioning is absolutely impeccable.", "This could be the defining moment of the match."
Write 2-3 sentences per caption. Professional tone, clear analysis, building excitement. Fewer captions (2-4 max) with longer text.""",
        
        "story_roast": """Generate sarcastic, playful roasting commentary.
Examples: "Oh no. Oh no no no.", "Someone's definitely uninstalling after this.", "The audacity. The absolute audacity of this play."
Write 2-3 sentences per caption. Sarcastic but not mean, comedic timing, playful mockery. Fewer captions (2-4 max).""",
        
        "story_creepypasta": """Generate horror-style tension narrative.
Examples: "Something felt wrong.", "The game knew what was about to happen.", "And then... it did."
Write 2-3 sentences per caption. Build tension, ominous tone, slow reveals. Fewer captions (2-4 max). Use ellipses for suspense.""",
        
        "story_dramatic": """Generate epic cinematic narration.
Examples: "In the arena of champions, legends are born.", "The crowd holds its breath.", "One shot. One chance. Immortality awaits."
Write 2-3 sentences per caption. Epic tone, powerful delivery, movie trailer style. Fewer captions (2-4 max)."""
    }
    
    style_guide = style_guides.get(style, style_guides["gaming"])
    
    # Adjust caption duration guidance for story modes
    caption_duration = "4-8 seconds" if is_story_mode else "1-3 seconds"
    caption_type = "narrative segments" if is_story_mode else "short captions"
    
    return f"""Watch this gameplay video and generate {max_captions} {caption_type} for key moments.

STYLE GUIDE:
{style_guide}

VIDEO DURATION: {duration:.1f} seconds

RULES:
1. Space captions throughout the video (not all at the start)
2. Each caption should appear for {caption_duration}
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
        
        # Calculate appropriate max_tokens based on video length and style
        # Story modes need more tokens (2-3 sentences vs 1-5 words)
        is_story_mode = style.startswith("story_")
        base_tokens = 500 if is_story_mode else 300  # Tokens per caption
        max_output_tokens = min(base_tokens * max_captions + 500, 8000)  # +500 for JSON overhead
        
        logging.debug(f"OpenAI caption generation: {len(frames)} frames, {max_captions} captions, "
                     f"max_tokens={max_output_tokens} ({'story' if is_story_mode else 'standard'} mode)")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=max_output_tokens,
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

