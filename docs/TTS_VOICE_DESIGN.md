# TTS Voice Design with Qwen3-TTS

## Overview

AutoShorts uses Qwen3-TTS's VoiceDesign feature with **detailed presets**:
- 12 acoustic dimensions per preset
- 16 unique presets (9 caption styles + 7 semantic types)
- Preset-only mode - no API calls required
- Result: Coherent, natural voices that match each clip's category

## Voice Control Dimensions

Based on Qwen3-TTS's VoiceDesign capabilities:

| Dimension | Description | Examples |
|-----------|-------------|----------|
| **gender** | Voice gender | Male, Female, Androgynous |
| **pitch** | Fundamental frequency | Low bass, Mid-range, High pitch, with inflections |
| **speed** | Speaking rate | Very slow, Moderate, Extremely fast, Variable |
| **volume** | Loudness | Quiet whisper, Conversational, Loud projection, Shouting |
| **age** | Speaker age | Teen, Young adult (20s-30s), Middle-aged (40s-50s) |
| **clarity** | Articulation precision | Clear, Highly articulate, Casual, Slurred |
| **fluency** | Speech smoothness | Flawless, Hesitations, Stammering, Filler words |
| **accent** | Regional accent | American, British, Neutral |
| **texture** | Vocal quality | Bright, Gravelly, Smooth, Nasal, Resonant |
| **emotion** | Emotional state | Excited, Calm, Confused, Amused, Foreboding |
| **tone** | Overall mood | Upbeat, Dramatic, Sarcastic, Ominous, Authoritative |
| **personality** | Character traits | Confident, Witty, Professional, Mysterious |

## Preset Examples

### Gaming Style (Action-Packed)
```
gender: Male.
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
personality: Confident, extroverted, engaging, competitive edge.
```

### Story Creepypasta (Horror)
```
gender: Male.
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
personality: Mysterious, unsettling, knows something you don't.
```

## About "code_predictor_config is None" Warning

**This is NORMAL and NOT an error!**

When you see these logs:
```
code_predictor_config is None. Initializing code_predictor model with default values
Setting `pad_token_id` to `eos_token_id`:2150 for open-end generation
```

**What it means:**
- Qwen3-TTS has internal components for voice generation
- "code_predictor" is a submodule that handles audio token prediction
- "default values" means using the model's built-in optimal settings
- These are **informational messages**, not errors or warnings

**Your voice descriptions ARE being used!**
- The VoiceDesign description is passed via the `instruct` parameter
- It controls the overall voice characteristics
- The "default values" refer to internal model parameters unrelated to your voice design

**Think of it like:**
- Your voice description = recipe ingredients (what you specify)
- Code predictor defaults = oven temperature settings (handled automatically)

## Voice Variety

With the new detailed presets, each style produces distinctly different voices:

- **Fastest:** HYPE (breathless, maximum energy)
- **Slowest:** CREEPYPASTA (deliberate tension-building)
- **Loudest:** HYPE (shouting celebration)
- **Quietest:** MINIMAL (intimate whisper)
- **Most Professional:** STORY_NEWS (broadcast quality)
- **Most Chaotic:** WTF (erratic confusion)
- **British Accent:** SKILL (cultured analyst)
- **GenZ Style:** GENZ (vocal fry, 'literally', 'bruh')

## How Voice Selection Works

1. **Caption Style Mode:**
   - User sets `CAPTION_STYLE=gaming` → uses GAMING preset
   - User sets `CAPTION_STYLE=story_news` → uses STORY_NEWS preset

2. **Auto Mode:**
   - AI detects semantic type (action, funny, clutch, etc.)
   - Maps to caption style via CAPTION_STYLE_MAP
   - Uses corresponding voice preset

3. **Voice Description Generation:**
   - For consistent voices, presets are used directly
   - For variety, AI can generate custom descriptions
   - All descriptions follow the 12-dimension format

## References

- [Qwen3-TTS Official Documentation](https://qwen.ai/blog?id=qwen3tts-0115)
- See `src/ai_providers.py` for all presets
- See `src/tts_generator.py` for voice generation logic
