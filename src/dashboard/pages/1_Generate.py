from __future__ import annotations

import os
import sys
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
        # Video selection
        st.markdown("### 🎮 Source Video")
        
        if gameplay_videos:
            options = {info.path.name: info for info in gameplay_videos}
            selection = st.selectbox(
                "Select gameplay video",
                options=list(options.keys()),
                label_visibility="collapsed"
            )
            selected = options[selection]
            st.session_state.selected_gameplay = selected.path
            
            # Video preview card
            with st.container():
                if selected.thumbnail:
                    st.image(str(selected.thumbnail), use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Duration", f"{selected.duration:.0f}s")
                col2.metric("Size", f"{selected.size_mb:.1f} MB")
                col3.metric("Resolution", selected.resolution or "N/A")
        else:
            st.warning("📁 No videos found in `./gameplay`")
            st.caption("Add your gameplay videos to the `gameplay/` folder or upload one below.")
        
        # Video upload / browse option
        with st.expander("📤 Add Video", expanded=not gameplay_videos):
            st.caption("Select a video to add to the gameplay folder")
            
            # Initialize session state for selected path
            if "selected_video_path" not in st.session_state:
                st.session_state.selected_video_path = ""
            
            col_input, col_browse = st.columns([3, 1])
            
            with col_browse:
                browse_clicked = st.button("📂 Browse", use_container_width=True, help="Open file browser")
            
            # Handle browse button click - open file dialog
            if browse_clicked:
                try:
                    import tkinter as tk
                    from tkinter import filedialog
                    root = tk.Tk()
                    root.withdraw()
                    root.attributes('-topmost', True)
                    root.focus_force()
                    file_path = filedialog.askopenfilename(
                        title="Select Video File",
                        filetypes=[
                            ("Video files", "*.mp4 *.mkv *.avi *.mov *.webm *.m4v"),
                            ("All files", "*.*")
                        ]
                    )
                    root.destroy()
                    if file_path:
                        st.session_state.selected_video_path = file_path
                        st.rerun()
                except Exception as e:
                    st.error(f"Could not open file browser: {e}")
                    st.caption("Please paste the full file path below")
            
            with col_input:
                # Use on_change to sync input with session state
                def update_path():
                    st.session_state.selected_video_path = st.session_state.video_path_input
                
                video_path = st.text_input(
                    "Video path",
                    value=st.session_state.selected_video_path,
                    placeholder="Click Browse or paste path here",
                    label_visibility="collapsed",
                    key="video_path_input",
                    on_change=update_path
                )
            
            # Show the current selected path for confirmation
            if st.session_state.selected_video_path:
                st.caption(f"📁 Selected: `{Path(st.session_state.selected_video_path).name}`")
            
            # Add button
            add_clicked = st.button(
                "➕ Add Video", 
                type="primary", 
                use_container_width=True, 
                disabled=not st.session_state.selected_video_path,
                help="Creates a link to the video (no copying, saves disk space)"
            )
            
            if add_clicked and st.session_state.selected_video_path:
                video_path = st.session_state.selected_video_path.strip()
                source_path = Path(video_path)
                
                # Debug info
                st.info(f"Attempting to link: {source_path}")
                
                if not source_path.exists():
                    st.error(f"❌ File not found: {video_path}")
                elif not source_path.is_file():
                    st.error(f"❌ Not a file: {video_path}")
                else:
                    gameplay_dir = Path("gameplay")
                    gameplay_dir.mkdir(exist_ok=True)
                    
                    link_path = gameplay_dir / source_path.name
                    
                    if link_path.exists() or link_path.is_symlink():
                        st.warning(f"⚠️ '{source_path.name}' already exists in gameplay folder")
                    else:
                        try:
                            # Create symlink with absolute path
                            abs_source = source_path.resolve()
                            link_path.symlink_to(abs_source)
                            
                            # Verify symlink was created
                            if link_path.exists() or link_path.is_symlink():
                                st.success(f"✅ Added '{source_path.name}' to gameplay folder")
                                st.session_state.selected_video_path = ""
                                # Clear any cached video list
                                if "video_list_cache" in st.session_state:
                                    del st.session_state.video_list_cache
                                st.rerun()
                            else:
                                st.error("❌ Symlink creation failed silently")
                        except PermissionError:
                            st.error("❌ Permission denied. Try running with administrator privileges.")
                        except OSError as e:
                            st.error(f"❌ Failed to create link: {e}")
                            st.caption("On Windows, you may need to enable Developer Mode or run as Administrator.")
        
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
            use_container_width=True,
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
            use_container_width=True,
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
    if status.running:
        import time
        time.sleep(2)
        st.rerun()


if __name__ == "__main__":
    render()
