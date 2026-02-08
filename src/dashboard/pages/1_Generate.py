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
                        st.image(str(vid.thumbnail), use_container_width=True)
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
                        # if vid.thumbnail: st.image(str(vid.thumbnail), use_container_width=True)
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
            if st.button("📂 Add Video", type="primary", use_container_width=True):
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
            if st.button("🗑️ Clear Queue", type="secondary", use_container_width=True, disabled=not gameplay_videos):
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
