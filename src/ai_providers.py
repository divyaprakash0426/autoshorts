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
        "action": "Standard combat, shooting, fighting - normal intense gameplay without special moments",
        "funny": "Fails, glitches, ragdoll physics, unexpected humor, comedic timing, awkward moments",
        "clutch": "1vX clutch wins, last-second saves, comebacks from low health, narrowly avoiding death",
        "wtf": "Unexpected chaos, 'wait what?!' moments, random events, physics breaking, confusing situations",
        "epic_fail": "Embarrassing deaths, tragic blunders, game-losing mistakes, throwing the match",
        "hype": "Victory celebrations, 'LET'S GO' energy, peak excitement, winning moments",
        "skill": "Trick shots, flick shots, IQ plays, advanced mechanics, pro-level techniques, impressive aim",
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
        
        "funny": """gender: Female.
pitch: Mid to slightly high female pitch with playful melodic variations.
speed: Moderate pace with perfect comedic timing and dramatic pauses.
volume: Conversational, occasionally louder for emphasis on punchlines.
age: Young adult, early to mid 20s.
clarity: Clear and expressive, not overly precise.
fluency: Fluent with intentional hesitations for humor, natural 'like', 'literally'.
accent: American English, casual GenZ cadence.
texture: Bright, warm vocal quality with infectious energy.
emotion: Amused, ironic, playfully sarcastic with eye-roll energy.
tone: Laid-back but sharp, witty commentary with smirk energy.
personality: Quick-witted, self-aware, relatable, gently roasting.""",
        
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
        
        "wtf": """gender: Female.
pitch: Expressive female pitch with dramatic jumps and confused inflections.
speed: Irregular pacing, speeding up in disbelief, pausing in utter confusion.
volume: Moderate with sudden loud outbursts of surprise and shock.
age: Young adult, early to mid 20s.
clarity: Clear but broken by genuine bewilderment and disbelief.
fluency: Interrupted flow, hesitations, stammering from pure shock.
accent: American English, casual conversational.
texture: Bright, expressive vocal quality with air of disbelief.
emotion: Confused, bewildered, amused by complete absurdity.
tone: 'Wait what?!', 'excuse me?!', questioning everything, hilariously perplexed.
personality: Reactive, dramatically expressive, openly confused but thoroughly entertained.""",
        
        "epic_fail": """gender: Female.
pitch: Mid-range descending to low sympathetic tones with dramatic flair.
speed: Slow, heavy delivery with resigned sighs and dramatic pauses.
volume: Starts normal, drops to quiet disappointment, occasional disbelief spikes.
age: Young adult, mid 20s.
clarity: Very clear, articulating the tragedy with dramatic emphasis.
fluency: Smooth but with emotional pauses, sighs, and sympathetic groans.
accent: American English, expressive and warm.
texture: Warm, empathetic vocal quality with theatrical flair.
emotion: Disappointed, sympathetic, finding humor in the tragedy.
tone: Dramatic despair mixed with 'oh no honey', mock-serious commentary.
personality: Empathetic storyteller, understanding, finding comedic gold in failure.""",
        
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
        
        "skill": """gender: Female.
pitch: Clear mid-range female pitch, stable and confident.
speed: Measured, deliberate pace allowing appreciation of the play.
volume: Clear and projected, professional analyst level.
age: Adult, late 20s to mid 30s.
clarity: Impeccable articulation, refined and precise speech.
fluency: Flawless flow, intelligent vocabulary.
accent: British English, educated and polished.
texture: Smooth, sophisticated vocal quality with warmth and depth.
emotion: Genuinely impressed, appreciative, respectful admiration.
tone: Analytical yet enthusiastic, thoughtful expert praise.
personality: Intelligent, observant, genuine appreciation for craft and technique.""",
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
        
        "story_news": """gender: Male.
pitch: Mid-range male pitch, energetic and dynamic.
speed: Fast-paced, building excitement with rapid-fire delivery.
volume: Loud, projected, arena-filling energy.
age: Young adult, late 20s to early 30s.
clarity: Sharp, punchy enunciation, esports casting precision.
fluency: Rapid-fire with hype pauses, building momentum.
accent: American English, energetic gaming culture.
texture: Bright, electric vocal quality with infectious enthusiasm.
emotion: Excited, hyped, passionate about the plays.
tone: Enthusiastic, analytical, building tension and release.
personality: Charismatic caster, knowledgeable, gets hyped with the audience.""",
        
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
        
        "story_dramatic": """gender: Female.
pitch: Rich, resonant mid-range with expressive depth.
speed: Measured, deliberate pacing with dramatic pauses for impact.
volume: Commanding presence, clear projection with emotional range.
age: Mature adult, late 30s to 40s.
clarity: Perfect diction, every word delivered with intention.
fluency: Flawless delivery with cinematic timing and gravitas.
accent: American English, theatrical broadcast quality.
texture: Warm, velvety depth with captivating allure.
emotion: Intense, evocative, drawing listeners into the story.
tone: Epic, momentous, like narrating legends and tragedies.
personality: Wise, commanding, magnetic storyteller presence.""",
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
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
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
                import time
                logging.info(f"Analyzing clip: {clip_info.clip_path.name}")
                client = self._get_client()

                # Upload with retry logic
                video_file = None
                upload_attempts = 3
                for attempt in range(upload_attempts):
                    try:
                        with open(clip_info.clip_path, "rb") as f:
                            video_file = client.files.upload(
                                file=f,
                                config={"mime_type": "video/mp4"}
                            )
                        break
                    except Exception as e:
                        if attempt < upload_attempts - 1:
                            logging.warning(f"Upload attempt {attempt + 1} failed for {clip_info.clip_path.name}: {e}, retrying...")
                            time.sleep(2 ** attempt)
                        else:
                            raise
                
                # Poll until file is ACTIVE
                max_wait = 120  # 2 minutes max
                poll_interval = 2
                waited = 0
                while waited < max_wait:
                    try:
                        file_status = client.files.get(name=video_file.name)
                        state = file_status.state.name
                        if state == "ACTIVE":
                            break
                        elif state == "FAILED":
                            raise RuntimeError(f"Gemini file processing failed: {video_file.name}")
                        time.sleep(poll_interval)
                        waited += poll_interval
                    except Exception as e:
                        if "precondition" in str(e).lower() and waited < max_wait:
                            time.sleep(poll_interval)
                            waited += poll_interval
                        else:
                            raise
                else:
                    raise RuntimeError(f"Gemini file processing timed out after {max_wait}s")
                
                # Small delay for stability
                time.sleep(0.5)
                
                prompt = f"""Analyze this gameplay video clip and classify it into the BEST matching category.

IMPORTANT: Be specific! Don't default to "action" - carefully consider ALL categories:

CATEGORIES:
{categories_desc}

CLASSIFICATION GUIDE:
- ACTION: Standard combat, shooting, fighting - the "normal" gameplay
- FUNNY: Look for fails, glitches, ragdoll physics, unexpected humor, awkward moments
- CLUTCH: Last-second saves, 1vX wins, comeback victories, surviving with low health
- WTF: "Wait, what just happened?!" moments - random, unexpected, confusing events
- EPIC_FAIL: Deaths that are embarrassing, tragic mistakes, throwing the game
- HYPE: Celebrations, victory moments, peak excitement, "LET'S GO" energy
- SKILL: Impressive plays - trick shots, perfect aim, smart strategies, pro-level moves

Ask yourself: What makes this clip SPECIAL? Is it just normal action, or is there something MORE?

Respond with ONLY valid JSON (score each category 0.0-1.0):
{{{categories_json}, "best_category": "action", "reason": "Why this clip fits the selected category"}}

Be honest with scores - if a clip is genuinely just normal action, that's fine. But if there's humor, skill, or a WTF moment, IDENTIFY IT!"""
                
                # Retry logic for generate_content
                max_retries = 3
                response = None
                for retry in range(max_retries):
                    try:
                        response = client.models.generate_content(
                            model=self.model_name,
                            contents=[video_file, prompt]
                        )
                        break
                    except Exception as e:
                        if "precondition" in str(e).lower() and retry < max_retries - 1:
                            logging.warning(f"Precondition error for {clip_info.clip_path.name}, retry {retry + 1}...")
                            time.sleep(3 * (retry + 1))
                        else:
                            raise
                
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

        max_workers = 3  # Reduced from 5 to avoid overwhelming Gemini API
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

    def analyze_full_video(
        self,
        video_path: Path,
        duration: float,
        target_count: int = 15
    ) -> List[Tuple[float, float, str, float]]:
        """Analyze the entire video to find the best moments directly.

        This method:
        1. Downscales the video to a small proxy (low res, low fps)
        2. Uploads the proxy to Gemini
        3. Asks Gemini to return a list of JSON timestamps for top moments
        4. Returns a list of (start, end, category, score) tuples

        Args:
            video_path: Path to the full source video
            duration: Video duration in seconds
            target_count: Number of moments to request

        Returns:
            List of (start_time, end_time, category, score)
        """
        if not self.is_available():
            logging.warning("Gemini API key not configured. Returning empty list.")
            return []

        # 1. Create a low-res proxy
        import cv2
        import numpy as np
        import subprocess

        logging.info(f"🚀 Starting Gemini Deep Analysis (Targeting ~{target_count} moments)...")

        # Create temp file for proxy
        import tempfile
        import hashlib
        
        # Generate hash based on file path and size/mtime to ensure uniqueness per video
        video_hash = hashlib.md5(f"{video_path}_{video_path.stat().st_size}".encode()).hexdigest()[:10]
        # Clean filename for safety but keep extension
        clean_name = "".join(c for c in video_path.stem if c.isalnum() or c in ('-', '_'))[:30]
        temp_proxy = Path(tempfile.gettempdir()) / f"proxy_{clean_name}_{video_hash}.mp4"
        
        # Try to use GPU-accelerated encoding if available (much faster)
        # Using hevc_nvenc for fast encoding, scale_cuda for GPU resizing
        gpu_cmd = [
            "ffmpeg", "-y",
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            "-i", str(video_path),
            "-vf", "scale_cuda=640:-2,fps=1", # 1fps on GPU
            "-c:v", "hevc_nvenc",
            "-preset", "fast",
            "-rc", "constqp",
            "-qp", "35", # High compression
            "-c:a", "aac", "-b:a", "32k", "-ac", "1",
            str(temp_proxy)
        ]
        
        # Fallback to CPU if GPU fails
        cpu_cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", "scale=-2:480,fps=1",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "30",
            "-c:a", "aac", "-b:a", "32k", "-ac", "1",
            str(temp_proxy)
        ]

        if temp_proxy.exists() and temp_proxy.stat().st_size > 1024:
            logging.info(f"Using cached proxy: {temp_proxy}")
        else:
            logging.info(f"Generating low-res proxy for AI analysis: {temp_proxy}")
            import time
            from tqdm import tqdm
            import re
            
            # Helper to get video duration
            def get_duration(path):
                try:
                    res = subprocess.run([
                        "ffprobe", "-v", "error", "-show_entries", "format=duration", 
                        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
                    ], capture_output=True, text=True)
                    return float(res.stdout.strip())
                except:
                    return 0
            
            total_duration = get_duration(video_path)
            
            start_time = time.time()
            # RegEx to capture time=HH:MM:SS.mm
            time_pattern = re.compile(r"time=(\d{2}):(\d{2}):(\d{2}\.\d{2})")
            
            # Try GPU first with progress
            try:
                # Add -progress pipe:1 to get progress info on stdout/stderr
                # But typically reading stderr for `time=` is easier/cross-platform compatible
                cmd = gpu_cmd
                
                logging.info(f"Rendering proxy with GPU (Duration: {total_duration:.1f}s)...")
                
                process = subprocess.Popen(
                    cmd,
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    universal_newlines=True
                )
                
                # Progress bar
                pbar = tqdm(total=total_duration, unit="s", desc="Generating Proxy", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}]")
                
                for line in process.stderr:
                    match = time_pattern.search(line)
                    if match:
                        hours, minutes, seconds = map(float, match.groups())
                        current_time = hours * 3600 + minutes * 60 + seconds
                        pbar.n = min(current_time, total_duration)
                        pbar.refresh()
                
                process.wait()
                pbar.close()
                
                if process.returncode != 0:
                    logging.warning("GPU proxy generation returned non-zero exit code")
                    raise subprocess.CalledProcessError(process.returncode, cmd)
                    
                logging.info(f"GPU Proxy generation took {time.time() - start_time:.1f}s")
                
            except (subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
                logging.warning(f"GPU proxy generation failed ({e}), falling back to CPU...")
                # Fallback to CPU with progress
                try:
                    cmd = cpu_cmd
                    process = subprocess.Popen(
                        cmd,
                        stderr=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        universal_newlines=True
                    )
                    
                    pbar = tqdm(total=total_duration, unit="s", desc="Generating Proxy (CPU)", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}]")
                    
                    for line in process.stderr:
                        match = time_pattern.search(line)
                        if match:
                            hours, minutes, seconds = map(float, match.groups())
                            current_time = hours * 3600 + minutes * 60 + seconds
                            pbar.n = min(current_time, total_duration)
                            pbar.refresh()
                    
                    process.wait()
                    pbar.close()
                    
                    if process.returncode != 0:
                        raise subprocess.CalledProcessError(process.returncode, cmd)
                        
                    logging.info(f"CPU Proxy generation took {time.time() - start_time:.1f}s")
                    
                except subprocess.CalledProcessError as e:
                    logging.error(f"Failed to generate proxy video: {e}")
                    return []

        # 2. Upload to Gemini
        client = self._get_client()
        uploaded_file = None
        
        try:
            logging.info(f"Uploading proxy ({temp_proxy.stat().st_size / 1024 / 1024:.1f} MB) to Gemini...")
            with open(temp_proxy, "rb") as f:
                uploaded_file = client.files.upload(
                    file=f,
                    config={"mime_type": "video/mp4"}
                )

            # Poll for active
            while True:
                file_status = client.files.get(name=uploaded_file.name)
                if file_status.state.name == "ACTIVE":
                    break
                elif file_status.state.name == "FAILED":
                    raise RuntimeError("Gemini processing failed")
                time.sleep(2)
            
            logging.info("Proxy processing complete. Prompting Gemini...")

            # 3. Prompt for timestamps
            categories_list = "\n".join(
                f"- {cat.upper()}: {desc}" 
                for cat, desc in ClipScore.SEMANTIC_TYPES.items()
            )

            prompt = f"""You are a professional video editor creating viral shorts.
Your task is to watch this entire video and identify the ABSOLUTE BEST moments.

CATEGORIES:
{categories_list}

INSTRUCTIONS:
1. Find {target_count} distinct, high-quality moments.
2. For each moment, provide the EXACT start and end timestamps (MM:SS).
3. Assign a 'score' (0.0 to 1.0) based on how viral/engaging it is.
4. Assign a 'category' from the list above.

IMPORTANT:
- Focus on moments that stand out (kills, funny fails, laughs, glitches, hypes).
- Ignore boring travel or looting.
- Clips should be 15-60 seconds long.
- Return ONLY valid JSON.

JSON FORMAT:
{{
    "moments": [
        {{ "start": "04:20", "end": "04:50", "category": "funny", "score": 0.95, "reason": "Streamer laughs at glitch" }},
        {{ "start": "12:15", "end": "12:45", "category": "skill", "score": 0.88, "reason": "Triple kill" }}
    ]
}}
"""
            # Use client.models.generate_content instead of client.GenerativeModel
            from google.api_core import exceptions
            
            max_retries = 5
            base_delay = 2
            response = None
            
            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model=self.model_name,
                        contents=[uploaded_file, prompt]
                    )
                    break
                except (exceptions.ServiceUnavailable, exceptions.ResourceExhausted) as e:
                    if attempt == max_retries - 1:
                        logging.error(f"Gemini API failed after {max_retries} attempts: {e}")
                        raise e
                    
                    delay = base_delay * (2 ** attempt)
                    logging.warning(f"Gemini API overloaded (503/429), retrying in {delay}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                except Exception as e:
                    logging.error(f"Gemini API unexpected error: {e}")
                    raise e
            
            # 4. Parse response
            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            data = json.loads(text)
            moments = data.get("moments", [])
            
            # Convert MM:SS to seconds
            parsed_moments = []
            
            def time_str_to_seconds(t_str):
                parts = list(map(int, t_str.split(":")))
                if len(parts) == 2:
                    return parts[0] * 60 + parts[1]
                elif len(parts) == 3:
                    return parts[0] * 3600 + parts[1] * 60 + parts[2]
                return 0

            for m in moments:
                try:
                    start_str = m["start"]
                    end_str = m["end"]
                    
                    start_sec = time_str_to_seconds(start_str)
                    end_sec = time_str_to_seconds(end_str)
                    
                    if end_sec > start_sec:
                        parsed_moments.append((
                            float(start_sec),
                            float(end_sec),
                            m["category"].lower(),
                            float(m["score"])
                        ))
                except Exception as e:
                    logging.warning(f"Failed to parse moment {m}: {e}")
            
            logging.info(f"Gemini found {len(parsed_moments)} notable moments.")
            return parsed_moments

        except Exception as e:
            logging.error(f"Deep Analysis failed: {e}")
            return []
        finally:
            # Cleanup
            if uploaded_file:
                try:
                    client.files.delete(name=uploaded_file.name)
                except:
                    pass
            # Don't delete cached proxy to speed up re-runs? 
            # Ideally we should clean it up, but for dev speed let's keep it if small?
            # No, let's clean up to avoid filling /tmp
            if temp_proxy.exists():
                try:
                    temp_proxy.unlink()
                except:
                    pass



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
                
                prompt_text = f"""These are keyframes from a gameplay video clip. Analyze and classify into the BEST matching category.

IMPORTANT: Be specific! Don't default to "action" - carefully consider ALL categories:

CATEGORIES:
{categories_desc}

CLASSIFICATION GUIDE:
- ACTION: Standard combat, shooting, fighting - the "normal" gameplay
- FUNNY: Look for fails, glitches, ragdoll physics, unexpected humor, awkward moments
- CLUTCH: Last-second saves, 1vX wins, comeback victories, surviving with low health
- WTF: "Wait, what just happened?!" moments - random, unexpected, confusing events
- EPIC_FAIL: Deaths that are embarrassing, tragic mistakes, throwing the game
- HYPE: Celebrations, victory moments, peak excitement, "LET'S GO" energy
- SKILL: Impressive plays - trick shots, perfect aim, smart strategies, pro-level moves

Ask yourself: What makes this clip SPECIAL? Is it just normal action, or is there something MORE?

Respond with ONLY valid JSON (score each category 0.0-1.0):
{{{categories_json}, "best_category": "action", "reason": "Why this clip fits the selected category"}}

Be honest with scores - if genuinely just normal action, that's fine. But if there's humor, skill, or a WTF moment, IDENTIFY IT!"""
                
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


# Language name mapping for caption/narration prompts
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


def generate_ai_captions(
    video_path: Path,
    style: str = "gaming",
    max_captions: int = 8,
    language: str = None,
) -> CaptionResult:
    """Generate AI-powered captions for a video clip.
    
    Instead of transcribing speech, the AI watches the video and generates
    contextual captions based on the visual content - perfect for gameplay
    videos without voice commentary.
    
    Args:
        video_path: Path to the video clip
        style: Caption style - "gaming", "dramatic", "funny", "minimal"
        max_captions: Maximum number of captions to generate
        language: Language code (en, ja, ko, etc.) - defaults to TTS_LANGUAGE env var
        
    Returns:
        CaptionResult with list of Caption objects
    """
    # Get language from parameter or environment
    if language is None:
        language = os.getenv("TTS_LANGUAGE", "en")
    
    provider = os.getenv("AI_PROVIDER", "gemini").lower()
    
    if provider == "openai":
        return _generate_captions_openai(video_path, style, max_captions, language)
    elif provider == "local":
        logging.warning("AI captions not available in local mode")
        return CaptionResult(captions=[], provider="local", success=False, 
                           error="AI captions require Gemini or OpenAI")
    else:  # Default to Gemini
        return _generate_captions_gemini(video_path, style, max_captions, language)


def _get_caption_prompt(style: str, max_captions: int, duration: float, language: str = "en") -> str:
    """Generate the prompt for AI caption generation."""
    
    # Get full language name
    lang_name = LANGUAGE_NAMES.get(language, "English")
    
    # Story modes use fewer but longer captions (but not as aggressively capped)
    is_story_mode = style.startswith("story_")
    if is_story_mode:
        # Story mode: aim for ~1 caption per 6-8 seconds instead of hard cap
        # Let dynamic calculation in subtitle_generator handle the count
        max_captions = min(max_captions, max(6, int(duration / 6)))  # At least 6, or 1 per 6s
    
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
Write 2-3 sentences per caption. Epic tone, powerful delivery, movie trailer style. Fewer captions (2-4 max).""",
        "podcast_quote": """Generate crisp pull-quote captions from podcast moments.
Examples: "That's the real problem.", "Clip this part.", "He actually said that."
Keep captions concise and quotable (3-10 words). Prioritize opinionated or insightful lines.""",

        "podcast_highlight": """Generate conversational podcast highlight captions.
Examples: "This is where it gets interesting.", "Here's the key takeaway.", "Wait, that's huge."
Use natural spoken phrasing. Keep captions 4-12 words with clear context.""",

        "sports_playbyplay": """Generate energetic sports play-by-play captions.
Examples: "HE'S THROUGH!", "LAST-SECOND FINISH!", "WHAT A COMEBACK!"
Short, high-energy, broadcast-like. Use CAPS for peak moments.""",

        "sports_hype": """Generate crowd-hype sports captions.
Examples: "THE STADIUM ERUPTS!", "ABSOLUTE SCENES!", "GOOSEBUMPS."
Very punchy captions (1-6 words) focused on emotion and momentum.""",

        "educational_explainer": """Generate clear educational explainer captions.
Examples: "Here's why this works.", "Step 1: isolate the variable.", "This is the core concept."
Be concise, structured, and easy to follow. Prioritize clarity over hype.""",

        "educational_aha": """Generate 'aha moment' educational captions.
Examples: "Now it clicks.", "This changes everything.", "That's the trick."
Highlight key insights and breakthroughs in simple language.""",

        "news_breaking": """Generate urgent breaking-news style captions.
Examples: "BREAKING UPDATE", "Developing story.", "Major reaction just in."
Keep captions short, factual, and high-impact.""",

        "news_debate": """Generate commentary/debate style captions.
Examples: "Strong take incoming.", "This point is controversial.", "Counterargument: ..."
Emphasize contrasting viewpoints and discussion beats.""",

        "vlog_story": """Generate personal vlog storytelling captions.
Examples: "I didn't expect this.", "Best part of the trip.", "Core memory unlocked."
Warm, relatable tone. Keep captions 3-10 words.""",

        "vlog_highlight": """Generate vlog highlight captions.
Examples: "Peak moment.", "Worth every second.", "This made my day."
Focus on memorable moments and emotional payoff.""",

        "music_hype": """Generate concert/music hype captions.
Examples: "THE DROP IS INSANE", "Crowd went wild.", "This hook is unreal."
Punchy, rhythmic, high-energy captions.""",

        "music_vibe": """Generate aesthetic music vibe captions.
Examples: "Late-night energy.", "On repeat.", "Pure vibes."
Minimal, mood-first captions (1-5 words).""",

        "interview_quote": """Generate interview pull-quote captions.
Examples: "That's a bold statement.", "He answered directly.", "Most honest line yet."
Keep captions quotable and context-rich.""",

        "interview_hot_take": """Generate hot-take interview captions.
Examples: "This will spark debate.", "Unfiltered opinion.", "People won't agree on this."
Emphasize controversy and reaction-worthy moments.""",

        "comedy_punchline": """Generate comedy punchline captions.
Examples: "AND THAT'S THE JOKE.", "Perfect timing.", "Crowd lost it."
Prioritize setup/payoff rhythm and punchline emphasis.""",

        "comedy_reaction": """Generate crowd/reaction comedy captions.
Examples: "He can't keep a straight face.", "Everyone's dying laughing.", "That landed."
Focus on reactions, laughter, and comedic aftermath.""",

        "cooking_step": """Generate cooking/DIY step captions.
Examples: "Step 2: fold gently.", "Don't overmix here.", "Now plate it."
Instructional and clear, with practical guidance.""",

        "cooking_reveal": """Generate cooking/DIY reveal captions.
Examples: "Final reveal.", "Before vs after.", "This turned out incredible."
Highlight transformation moments and satisfying results.""",

        "fitness_coach": """Generate coaching-style fitness captions.
Examples: "Keep your core tight.", "Drive through the heels.", "Control the tempo."
Actionable form cues in concise language.""",

        "fitness_motivation": """Generate motivational fitness captions.
Examples: "One more rep.", "No excuses today.", "Progress over perfection."
Short, motivating captions with high intent.""",

        "entertainment_reaction": """Generate entertainment reaction captions.
Examples: "I DID NOT SEE THAT COMING.", "That scene was wild.", "What a twist."
Focus on emotional reactions and surprise moments.""",

        "entertainment_recap": """Generate entertainment recap captions.
Examples: "Quick recap:", "Here's what happened.", "Key moment right here."
Summarize important scenes clearly and briefly.""",

        "tv_recap": """Generate TV show recap captions.
Examples: "Episode turning point.", "Character arc shift.", "Big reveal."
Concise narrative recap with context.""",

        "tv_dramatic": """Generate dramatic TV-style captions.
Examples: "No one was ready for this.", "Everything changes now.", "The tension is unreal."
Build tension and cliffhanger energy.""",

        "documentary_insight": """Generate documentary insight captions.
Examples: "Key insight:", "This explains the pattern.", "The data tells a story."
Informative, thoughtful, and precise.""",

        "documentary_wonder": """Generate documentary wonder captions.
Examples: "Nature at full scale.", "Hard to believe this is real.", "A rare moment captured."
Emphasize awe, discovery, and visual spectacle.""",

        "esports_playcast": """Generate esports caster-style captions.
Examples: "TEAMFIGHT NOW!", "Perfect timing on the engage.", "Objective secured."
Fast-paced esports broadcast tone with tactical callouts.""",

        "esports_clutch": """Generate esports clutch-moment captions.
Examples: "1v3. No fear.", "Ice-cold decision making.", "CLUTCH FACTOR."
High-tension competitive style, short and intense."""
    }
    
    style_guide = style_guides.get(style, style_guides["gaming"])
    
    # Adjust caption duration guidance for story modes
    caption_duration = "4-8 seconds" if is_story_mode else "1-3 seconds"
    caption_type = "narrative segments" if is_story_mode else "short captions"
    
    # Language instruction
    if language != "en":
        language_instruction = f"""\n\nLANGUAGE REQUIREMENT:
- Generate ALL caption text in {lang_name}.
- Adapt the style examples to be culturally appropriate for {lang_name} speakers.
- Use natural {lang_name} expressions and slang where appropriate.
- Do NOT translate literally - create authentic {lang_name} content."""
    else:
        language_instruction = ""
    
    return f"""Watch this gameplay video and generate {max_captions} {caption_type} for key moments.

STYLE GUIDE:
{style_guide}{language_instruction}

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
    max_captions: int,
    language: str = "en"
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
        
        # Upload video with retry logic
        import time
        video_file = None
        upload_attempts = 3
        for attempt in range(upload_attempts):
            try:
                with open(video_path, "rb") as f:
                    video_file = client.files.upload(
                        file=f,
                        config={"mime_type": "video/mp4"}
                    )
                break
            except Exception as e:
                if attempt < upload_attempts - 1:
                    logging.warning(f"Gemini upload attempt {attempt + 1} failed: {e}, retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
        
        # Poll until file is ACTIVE (Gemini processes uploads asynchronously)
        max_wait = 180  # Increased to 3 minutes
        poll_interval = 2
        waited = 0
        logging.info(f"Waiting for Gemini to process file: {video_file.name}")
        while waited < max_wait:
            try:
                file_status = client.files.get(name=video_file.name)
                state = file_status.state.name
                logging.debug(f"File state: {state} (waited {waited}s)")
                
                if state == "ACTIVE":
                    logging.info(f"File ready after {waited}s")
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
                    raise
        else:
            raise RuntimeError(f"Gemini file processing timed out after {max_wait}s")
        
        prompt = _get_caption_prompt(style, max_captions, duration, language)
        
        # Add extra delay before using the file to ensure it's fully ready
        time.sleep(1)
        
        # Retry logic for generate_content in case of transient errors
        max_retries = 3
        for retry in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[video_file, prompt]
                )
                break
            except Exception as e:
                if "precondition" in str(e).lower() and retry < max_retries - 1:
                    logging.warning(f"Precondition error on attempt {retry + 1}, retrying after delay...")
                    time.sleep(3 * (retry + 1))  # Increasing delay: 3s, 6s, 9s
                else:
                    raise
        
        # Parse response
        response_text = response.text.strip()
        captions = _parse_caption_response(response_text, duration)
        
        # Cleanup uploaded file
        try:
            client.files.delete(name=video_file.name)
            logging.debug(f"Cleaned up Gemini file: {video_file.name}")
        except Exception as e:
            logging.warning(f"Failed to cleanup Gemini file: {e}")
        
        logging.info(f"Generated {len(captions)} captions")
        return CaptionResult(captions=captions, provider="gemini", success=True)
        
    except Exception as e:
        logging.error(f"Gemini caption generation failed: {e}")
        logging.error(f"Error type: {type(e).__name__}")
        if hasattr(e, '__cause__') and e.__cause__:
            logging.error(f"Caused by: {e.__cause__}")
        return CaptionResult(captions=[], provider="gemini", success=False,
                           error=str(e))


def _generate_captions_openai(
    video_path: Path, 
    style: str, 
    max_captions: int,
    language: str = "en"
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
        prompt = _get_caption_prompt(style, max_captions, duration, language)
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
        "action": "action-packed content with intense moments, combat, speed",
        "podcasts": "podcast conversations with speaker turns, memorable quotes, and strong opinions",
        "entertainment": "movie or tv reaction content with emotional beats and memorable scenes",
        "sports": "sports content with clutch plays, goals, finishes, and crowd reactions",
        "vlogs": "vlog storytelling content with highlights and personal moments",
        "tv_shows": "tv show recap content with dramatic reveals and funny cuts",
        "documentaries": "documentary content with key insights and visual spectacle moments",
        "music": "music and concert content with drops, choruses, and audience energy",
        "educational": "educational explainer content with clear key takeaways and aha moments",
        "interviews": "interview content with quotable lines, reactions, and hot takes",
        "comedy": "comedy content with setup and punchline timing",
        "news_commentary": "news and commentary content with breaking moments and debates",
        "cooking_diy": "cooking and diy content with transformation reveals and actionable tips",
        "fitness": "fitness content with form cues, personal records, and motivation",
    }

    normalized_category = category
    if category.startswith("podcast_"):
        normalized_category = "podcasts"
    elif category.startswith("sports_"):
        normalized_category = "sports"
    elif category.startswith("educational_"):
        normalized_category = "educational"
    elif category.startswith("interview_"):
        normalized_category = "interviews"
    elif category.startswith("news_"):
        normalized_category = "news_commentary"
    elif category.startswith("vlog_"):
        normalized_category = "vlogs"
    elif category.startswith("music_"):
        normalized_category = "music"
    elif category.startswith("comedy_"):
        normalized_category = "comedy"
    elif category.startswith("cooking_"):
        normalized_category = "cooking_diy"
    elif category.startswith("fitness_"):
        normalized_category = "fitness"
    elif category.startswith("tv_"):
        normalized_category = "tv_shows"
    elif category.startswith("documentary_"):
        normalized_category = "documentaries"
    elif category.startswith("entertainment_"):
        normalized_category = "entertainment"
    
    context = category_context.get(normalized_category, category_context["gaming"])
    
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
) -> dict:
    """Generate PyCaps-compatible word lists from AI tags.
    
    This allows you to use PyCaps wordlist rules with AI-generated tags.
    
    Args:
        captions: Original captions
        tag_results: Dict from batch_tag_captions()
        config_highlight_color: Unused (kept for backward compatibility)
        
    Returns:
        word_lists_dict that can be passed to pycaps SemanticTagger.add_wordlist_rule()
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
    
    word_lists = {
        "action": list(all_action),
        "emphasis": list(all_emphasis),
        "emotion": list(all_emotion),
        "product": list(all_product)
    }
    
    return word_lists


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
