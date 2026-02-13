from __future__ import annotations

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if parent.name == "src":
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

import streamlit as st

from dashboard.components.log_viewer import render_logs
from dashboard.state import get_process_manager, init_state, push_job_history
from dashboard.utils.config import load_env_values, save_env_values
from dashboard.utils.thumbnails import list_videos

for parent in Path(__file__).resolve().parents:
    if parent.name == "src":
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break


LOGO_PATH = Path("assets/logo.png")


def _build_env(overrides: dict) -> dict:
    env = os.environ.copy()
    env.update({k: str(v) for k, v in overrides.items()})
    return env


def render() -> None:
    st.set_page_config(page_title="Generate - AutoShorts", page_icon="🎬", layout="wide")
    
    # Sidebar logo
    if LOGO_PATH.exists():
        st.logo(str(LOGO_PATH))
    
    # Import shared theme
    from dashboard.utils.theme import get_shared_css
    
    # Apply shared theme
    st.markdown(get_shared_css(), unsafe_allow_html=True)
    
    init_state()

    # Header
    st.markdown("""
        <div class="page-header">
            <h1>🎬 Generate Shorts</h1>
            <p>Select gameplay, configure style, and generate viral clips</p>
        </div>
    """, unsafe_allow_html=True)

    values, extras = load_env_values()
    gameplay_videos = list_videos(Path("gameplay"))

    # Two-column layout
    col_left, col_right = st.columns([1, 1])
    
    with col_left:

        # Video Queue Management
        st.markdown("### 🎬 Video Queue")

        # Directories
        gameplay_dir = Path("gameplay")
        disabled_dir = gameplay_dir / ".disabled"
        disabled_dir.mkdir(parents=True, exist_ok=True)

        # Get list of active and disabled videos
        gameplay_videos = list_videos(gameplay_dir)
        disabled_videos_list = list_videos(disabled_dir)

        # Display current queue
        queue_container = st.container()
        if gameplay_videos:
            st.info(f"📁 {len(gameplay_videos)} video(s) in active queue (will be processed)")
            
            # Show thumbnails grid
            cols = st.columns(3)
            for idx, vid in enumerate(gameplay_videos):
                col = cols[idx % 3]
                with col:
                    if vid.thumbnail:
                        st.image(str(vid.thumbnail), width="stretch")
                    st.caption(f"**{vid.path.name}**")
                    st.caption(f"⏱️ {vid.duration:.0f}s | 💾 {vid.size_mb:.1f}MB")
                    
                    # Remove button (move to disabled)
                    if st.button("❌ Remove", key=f"disable_{vid.path.name}", help="Move to disabled (does not delete file)"):
                        try:
                             dest_path = disabled_dir / vid.path.name
                             vid.path.rename(dest_path)
                             st.toast(f"Moved '{vid.path.name}' to disabled")
                             st.rerun()
                        except Exception as e:
                             st.error(f"Failed to move: {e}")

        else:
            st.info("The queue is empty. Add a video to start.")

        # Display Disabled / Staged Videos
        if disabled_videos_list:
            with st.expander(f"🚫 Disabled / Staged Videos ({len(disabled_videos_list)})", expanded=False):
                st.caption("Videos hered are available but won't be processed.")
                d_cols = st.columns(3)
                for idx, vid in enumerate(disabled_videos_list):
                    col = d_cols[idx % 3]
                    with col:
                        # Optional: thumbnail for disabled too? Yes.
                        # if vid.thumbnail: st.image(str(vid.thumbnail), width="stretch")
                        st.text(vid.path.name)
                        if st.button("➕ Add back", key=f"enable_{vid.path.name}"):
                            try:
                                dest_path = gameplay_dir / vid.path.name
                                vid.path.rename(dest_path)
                                st.toast(f"Added '{vid.path.name}' back to queue")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to enable: {e}")

        # Action Buttons
        st.divider()
        col_add, col_clear = st.columns([1, 1])
        
        with col_add:
            if st.button("📂 Add Video", type="primary", width="stretch"):
                try:
                    # Run the helper script to open file dialog
                    result = subprocess.run(
                        [sys.executable, "src/dashboard/utils/file_dialog_helper.py"],
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    
                    if result.returncode == 0 and result.stdout.strip():
                        source_path = Path(result.stdout.strip())
                        if source_path.exists():
                            gameplay_dir.mkdir(exist_ok=True)
                            
                            link_path = gameplay_dir / source_path.name
                            if link_path.exists():
                                st.warning(f"⚠️ '{source_path.name}' is already in the queue")
                            else:
                                try:
                                    link_path.symlink_to(source_path.resolve())
                                    st.success(f"✅ Added to queue: {source_path.name}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Failed to link video: {e}")
                    else:
                        if result.stderr:
                             st.error(f"Dialog Error: {result.stderr}")
                        
                except Exception as e:
                    st.error(f"Failed to open file dialog: {e}")
        
        with col_clear:
            if st.button("🗑️ Clear Queue", type="secondary", width="stretch", disabled=not gameplay_videos):
                cleaned_count = 0
                moved_count = 0
                for vid in gameplay_videos:
                    try:
                        if vid.path.is_symlink():
                            vid.path.unlink()
                            cleaned_count += 1
                        else:
                            # Move actual files to disabled instead of deleting!
                            dest_path = disabled_dir / vid.path.name
                            vid.path.rename(dest_path)
                            moved_count += 1
                    except Exception:
                        pass
                
                msg = []
                if cleaned_count > 0: msg.append(f"Removed {cleaned_count} links")
                if moved_count > 0: msg.append(f"Moved {moved_count} files to disabled")
                
                if msg:
                    st.success(", ".join(msg))
                    st.rerun()
                else:
                    st.warning("Queue already empty")

        
        st.divider()
        
        # Quick settings
        st.markdown("### ⚡ Quick Settings")
        
        quick_cols = st.columns(2)
        with quick_cols[0]:
            def save_scene_limit():
                values["SCENE_LIMIT"] = st.session_state.quick_scene_limit
                save_env_values(values, extras)
            
            st.number_input(
                "🎬 Clips to generate",
                value=int(values.get("SCENE_LIMIT", 6)),
                min_value=1,
                max_value=20,
                key="quick_scene_limit",
                help="Number of short clips to create from the video",
                on_change=save_scene_limit
            )
        
        with quick_cols[1]:
            def save_tts_toggle():
                values["ENABLE_TTS"] = str(st.session_state.quick_enable_tts).lower()
                save_env_values(values, extras)
            
            st.toggle(
                "🔊 AI Voiceover",
                value=str(values.get("ENABLE_TTS", "true")).lower() in ("true", "1", "yes"),
                key="quick_enable_tts",
                help="Add AI-generated narration to clips",
                on_change=save_tts_toggle
            )
    
    with col_right:
        # Caption style selection
        st.markdown("### 💬 Caption Style")
        
        # CAPTION_STYLE may be in extras (not in schema)
        current_caption = str(extras.get("CAPTION_STYLE", values.get("CAPTION_STYLE", "auto")))
        
        GAMING_STYLES = ["gaming", "dramatic", "funny", "minimal"]
        STORY_STYLES = ["story_news", "story_roast", "story_creepypasta", "story_dramatic"]
        
        # Determine current mode
        if current_caption == "auto":
            default_mode_idx = 0
        elif current_caption in GAMING_STYLES:
            default_mode_idx = 1
        elif current_caption == "genz":
            default_mode_idx = 2
        elif current_caption in STORY_STYLES:
            default_mode_idx = 3
        else:
            default_mode_idx = 0
        
        # Style mode tabs
        mode_tab = st.radio(
            "Mode",
            ["🎯 Auto", "🎮 Gaming", "🔥 GenZ", "📖 Story"],
            index=default_mode_idx,
            horizontal=True,
            key="caption_mode",
            label_visibility="collapsed"
        )
        
        def save_caption(new_style):
            if new_style != current_caption:
                # Save to extras since CAPTION_STYLE isn't in schema
                extras["CAPTION_STYLE"] = new_style
                save_env_values(values, extras)
        
        if mode_tab == "🎯 Auto":
            st.info("🤖 AI will pick the best style based on content")
            st.caption("Detects: gaming, dramatic, funny, or minimal automatically")
            save_caption("auto")
                
        elif mode_tab == "🎮 Gaming":
            style_info = {
                "gaming": ("🎮", "HEADSHOT! • GG EZ • Punchy gaming captions"),
                "dramatic": ("🎭", "The final stand... • Cinematic narration"),
                "funny": ("😂", "skill issue tbh • Meme-style commentary"),
                "minimal": ("✨", "nice. • Clean, understated captions"),
            }
            
            current_gaming = current_caption if current_caption in GAMING_STYLES else "gaming"
            
            # If just switched to Gaming mode from another mode, save the default
            if current_caption not in GAMING_STYLES:
                save_caption("gaming")
            
            for style, (icon, desc) in style_info.items():
                is_selected = style == current_gaming
                col1, col2 = st.columns([0.15, 0.85])
                with col1:
                    if st.button(icon, key=f"gaming_{style}", type="primary" if is_selected else "secondary"):
                        save_caption(style)
                        st.rerun()
                with col2:
                    label = f"**{style.title()}**" if is_selected else style.title()
                    st.markdown(f"{label}: {desc}")
                
        elif mode_tab == "🔥 GenZ":
            st.success("💀 bruh • no cap • fr fr • Slang-heavy reactions")
            st.caption("Perfect for younger audiences and meme content")
            save_caption("genz")
                
        elif mode_tab == "📖 Story":
            story_info = {
                "story_news": ("📰", "Professional esports broadcaster narration"),
                "story_roast": ("🔥", "Sarcastic roasting commentary"),
                "story_creepypasta": ("👻", "Horror/tension narrative style"),
                "story_dramatic": ("🎭", "Epic cinematic storytelling"),
            }
            
            current_story = current_caption if current_caption in STORY_STYLES else "story_news"
            
            # If just switched to Story mode from another mode, save the default
            if current_caption not in STORY_STYLES:
                save_caption("story_news")
            
            for style, (icon, desc) in story_info.items():
                is_selected = style == current_story
                col1, col2 = st.columns([0.15, 0.85])
                with col1:
                    if st.button(icon, key=f"story_{style}", type="primary" if is_selected else "secondary"):
                        save_caption(style)
                        st.rerun()
                with col2:
                    label = f"**{style.replace('story_', '').title()}**" if is_selected else style.replace('story_', '').title()
                    st.markdown(f"{label}: {desc}")

        # --- Language & Subtitle Style row ---
        st.divider()
        sub_left, sub_right = st.columns([1, 1])

        # Language selector
        with sub_left:
            st.markdown("### 🌐 Language")
            TTS_LANGUAGE_LABELS = {
                "en": "🇺🇸 English", "zh": "🇨🇳 Chinese", "ja": "🇯🇵 Japanese",
                "ko": "🇰🇷 Korean", "de": "🇩🇪 German", "fr": "🇫🇷 French",
                "ru": "🇷🇺 Russian", "pt": "🇧🇷 Portuguese", "es": "🇪🇸 Spanish",
                "it": "🇮🇹 Italian",
            }
            lang_options = list(TTS_LANGUAGE_LABELS.keys())
            current_lang = str(values.get("TTS_LANGUAGE", extras.get("TTS_LANGUAGE", "en")))
            lang_idx = lang_options.index(current_lang) if current_lang in lang_options else 0

            def save_language():
                new_lang = st.session_state.gen_language
                # Persist language as an extra since TTS_LANGUAGE is managed in Generate UI now
                values["TTS_LANGUAGE"] = new_lang
                extras["TTS_LANGUAGE"] = new_lang
                save_env_values(values, extras)

            st.selectbox(
                "Voiceover & caption language",
                options=lang_options,
                index=lang_idx,
                key="gen_language",
                format_func=lambda v: TTS_LANGUAGE_LABELS.get(v, v),
                on_change=save_language,
                label_visibility="collapsed",
            )
            st.caption(f"Currently: {TTS_LANGUAGE_LABELS.get(current_lang, current_lang)}")

        # PyCaps Subtitle Style selector
        with sub_right:
            st.markdown("### ✨ Subtitle Style")

            PYCAPS_TEMPLATES = {
                "hype":         {"icon": "⚡", "desc": "Comic bold, yellow highlights", "color": "#FFFF00", "bg": "#1a1a2e", "shadow": "#000", "font": "Impact"},
                "vibrant":      {"icon": "🌈", "desc": "Neon chromatic aberration", "color": "#FFFFFF", "bg": "#1a1a2e", "shadow": "#FF00FF", "font": "Impact"},
                "explosive":    {"icon": "💥", "desc": "Orange glow, intense energy", "color": "#FFDD00", "bg": "#1a1a2e", "shadow": "#FF4400", "font": "Impact"},
                "word-focus":   {"icon": "🎯", "desc": "Orange word highlighting", "color": "#FFFFFF", "bg": "#f76f00", "shadow": "none", "font": "Arial Black"},
                "line-focus":   {"icon": "📌", "desc": "Blue line highlight bar", "color": "#FFFFFF", "bg": "#0055DD", "shadow": "none", "font": "Arial Black"},
                "retro-gaming": {"icon": "👾", "desc": "Pixel arcade aesthetic", "color": "#FFFF88", "bg": "#0a0a3a", "shadow": "#000", "font": "monospace"},
                "neo-minimal":  {"icon": "💻", "desc": "Code editor dark theme", "color": "#569CD6", "bg": "#1E1E1E", "shadow": "none", "font": "Consolas, monospace"},
                "fast":         {"icon": "💨", "desc": "Thick outline, bold cursive", "color": "#FFFFFF", "bg": "#1a1a2e", "shadow": "#000", "font": "cursive"},
                "classic":      {"icon": "📝", "desc": "Clean white on black shadow", "color": "#FFFFFF", "bg": "#1a1a2e", "shadow": "#000", "font": "Arial"},
                "minimalist":   {"icon": "🤍", "desc": "Subtle, semi-transparent", "color": "rgba(255,255,255,0.9)", "bg": "rgba(0,0,0,0.5)", "shadow": "none", "font": "Helvetica Neue, Arial"},
                "default":      {"icon": "⬜", "desc": "Standard white text", "color": "#FFFFFF", "bg": "#1a1a2e", "shadow": "#000", "font": "Arial Black"},
                "model":        {"icon": "🖤", "desc": "Clean minimal embedded", "color": "#FFFFFF", "bg": "#1a1a2e", "shadow": "none", "font": "Arial"},
            }

            tpl_names = list(PYCAPS_TEMPLATES.keys())
            current_template = str(values.get("PYCAPS_TEMPLATE", extras.get("PYCAPS_TEMPLATE", "hype")))
            tpl_idx = tpl_names.index(current_template) if current_template in tpl_names else 0

            def save_template():
                new_tpl = st.session_state.gen_template
                if new_tpl != current_template:
                    # Persist template as an extra setting (moved out of Settings page)
                    extras["PYCAPS_TEMPLATE"] = new_tpl
                    values["PYCAPS_TEMPLATE"] = new_tpl
                    save_env_values(values, extras)

            selected = st.selectbox(
                "Subtitle template",
                options=tpl_names,
                index=tpl_idx,
                key="gen_template",
                format_func=lambda v: f"{PYCAPS_TEMPLATES[v]['icon']} {v}",
                on_change=save_template,
                label_visibility="collapsed",
            )

            # Render preview for selected template
            info = PYCAPS_TEMPLATES[selected]
            if info["shadow"] == "none":
                txt_shadow = "none"
            else:
                s = info["shadow"]
                txt_shadow = f"2px 2px 0 {s}, -2px -2px 0 {s}, 2px -2px 0 {s}, -2px 2px 0 {s}"

            st.markdown(f"""
                <div style="
                    background: {info['bg']};
                    border: 2px solid #6C63FF;
                    border-radius: 10px;
                    padding: 20px 16px;
                    text-align: center;
                    margin-top: 8px;
                    min-height: 70px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                ">
                    <span style="
                        color: {info['color']};
                        font-family: {info['font']};
                        font-size: 20px;
                        font-weight: 700;
                        text-shadow: {txt_shadow};
                        text-transform: uppercase;
                        letter-spacing: 1px;
                    ">HEADSHOT!</span>
                </div>
            """, unsafe_allow_html=True)
            st.caption(f"{info['desc']}")

    st.divider()

    # Job control section
    manager = get_process_manager()
    status = manager.status()

    st.markdown("### 🚀 Processing")
    
    control_cols = st.columns([1, 1, 2])
    
    with control_cols[0]:
        start_disabled = status.running
        if st.button(
            "▶️ Start Processing",
            type="primary",
            width="stretch",
            disabled=start_disabled
        ):
            save_env_values(values, extras)
            manager.start(env=_build_env(values))
            push_job_history({
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "video": str(st.session_state.get("selected_gameplay") or "All gameplay videos"),
                "status": "running",
            })
            st.rerun()
    
    with control_cols[1]:
        stop_disabled = not status.running
        if st.button(
            "⏹️ Stop",
            type="secondary",
            width="stretch",
            disabled=stop_disabled
        ):
            manager.stop()
            st.rerun()
    
    with control_cols[2]:
        # Status indicator
        if status.running:
            st.success(f"🟢 Running (PID {status.pid})")
        elif status.exit_code is not None:
            if status.exit_code == 0:
                st.info(f"✅ Completed successfully")
            else:
                st.error(f"❌ Exited with code {status.exit_code}")
        else:
            st.warning("⏸️ Idle - Ready to process")

    # Logs section
    st.markdown("### 📋 Logs")
    show_all = st.toggle("Show full logs", value=False, key="show_all_logs")
    render_logs(status.tail, show_all=show_all)

    # Auto-refresh while running
    # Auto-refresh while running (only if not viewing full logs)
    if status.running and not show_all:
        import time
        time.sleep(2)
        st.rerun()


if __name__ == "__main__":
    render()
