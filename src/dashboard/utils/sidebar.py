import streamlit as st
from pathlib import Path

def render_sidebar():
    """Render the common sidebar elements including the logo."""
    # Place logo at the top of the sidebar (above navigation) using st.logo
    logo_path = Path("assets/logo.png")
    if logo_path.exists():
        st.logo(str(logo_path), size="large", icon_image=str(logo_path))
    else:
        # Fallback
        st.sidebar.title("AutoShorts 🎬")
    
    # You can add other common sidebar elements here if needed
    #st.sidebar.markdown("---")
