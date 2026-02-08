"""Shared theme and styling for the dashboard.

Provides consistent colors, spacing, and UI components across all pages.
"""

# Color palette
COLORS = {
    # Primary
    "primary": "#00ff88",
    "primary_dark": "#00d26a",
    "primary_light": "#66ffaa",
    
    # Background
    "bg_dark": "#0a0a0f",
    "bg_medium": "#1a1a2e",
    "bg_light": "#2a2a3e",
    "bg_card": "#1e1e2e",
    
    # Accents
    "accent_blue": "#0f3460",
    "accent_purple": "#7c3aed",
    "accent_orange": "#ff6b35",
    
    # Status
    "success": "#00d26a",
    "warning": "#fbbf24",
    "error": "#ef4444",
    "info": "#3b82f6",
    
    # Text
    "text_primary": "#ffffff",
    "text_secondary": "#a0a0a0",
    "text_muted": "#6b7280",
    
    # Borders
    "border_light": "#333344",
    "border_medium": "#444455",
}

# Spacing
SPACING = {
    "xs": "0.25rem",
    "sm": "0.5rem",
    "md": "1rem",
    "lg": "1.5rem",
    "xl": "2rem",
    "2xl": "3rem",
}


def get_shared_css() -> str:
    """Get shared CSS for consistent styling across all pages."""
    return f"""
        <style>
        /* Global resets */
        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        
        /* Remove Streamlit branding colors */
        .stApp {{
            background-color: {COLORS['bg_dark']};
        }}
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {{
            font-weight: 600;
            letter-spacing: -0.02em;
        }}
        
        h1 {{
            font-size: 2.25rem;
            margin-bottom: 1rem;
        }}
        
        h2 {{
            font-size: 1.875rem;
            margin-bottom: 0.875rem;
        }}
        
        h3 {{
            font-size: 1.5rem;
            margin-bottom: 0.75rem;
        }}
        
        /* Hero sections */
        .hero-section {{
            background: linear-gradient(135deg, {COLORS['bg_medium']} 0%, {COLORS['accent_blue']} 100%);
            padding: {SPACING['2xl']};
            border-radius: 16px;
            text-align: center;
            margin-bottom: {SPACING['xl']};
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        }}
        
        .hero-title {{
            font-size: 3rem;
            font-weight: 700;
            margin: {SPACING['md']} 0;
            background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['primary_light']} 50%, {COLORS['primary']} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1.2;
        }}
        
        .hero-subtitle {{
            font-size: 1.5rem;
            color: {COLORS['text_secondary']};
            margin-top: {SPACING['sm']};
            font-weight: 400;
        }}
        
        /* Cards */
        .feature-card {{
            background: linear-gradient(145deg, {COLORS['bg_card']} 0%, {COLORS['bg_light']} 100%);
            border: 1px solid {COLORS['border_light']};
            border-radius: 12px;
            padding: {SPACING['lg']};
            height: 100%;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .feature-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 255, 136, 0.1);
            border-color: {COLORS['primary_dark']};
        }}
        
        .feature-card h4 {{
            color: {COLORS['primary']};
            margin-top: 0;
            margin-bottom: {SPACING['md']};
            font-size: 1.25rem;
        }}
        
        .feature-card ul {{
            margin: 0;
            padding-left: 1.25rem;
        }}
        
        .feature-card li {{
            margin-bottom: {SPACING['sm']};
            line-height: 1.6;
        }}
        
        /* Stats cards */
        .stats-card {{
            background: linear-gradient(145deg, rgba(0, 255, 136, 0.05) 0%, rgba(0, 210, 106, 0.08) 100%);
            border: 1px solid {COLORS['primary_dark']};
            border-radius: 12px;
            padding: {SPACING['lg']};
            text-align: center;
            transition: transform 0.2s;
        }}
        
        .stats-card:hover {{
            transform: scale(1.02);
        }}
        
        .stats-value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: {COLORS['primary']};
            line-height: 1;
        }}
        
        .stats-label {{
            font-size: 0.875rem;
            color: {COLORS['text_secondary']};
            margin-top: {SPACING['sm']};
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        /* Header sections */
        .page-header {{
            background: linear-gradient(135deg, {COLORS['bg_medium']} 0%, {COLORS['accent_blue']} 100%);
            padding: {SPACING['xl']};
            border-radius: 12px;
            margin-bottom: {SPACING['xl']};
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }}
        
        .page-header h1 {{
            margin: 0;
            color: {COLORS['text_primary']};
        }}
        
        .page-header p {{
            margin: {SPACING['sm']} 0 0 0;
            color: {COLORS['text_secondary']};
            font-size: 1.125rem;
        }}
        
        /* Base button styles - target all buttons */
        .stButton > button {{
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-size: 1rem;
            transition: all 0.2s;
        }}

        /* --- Hero Container (New) --- */
        .hero-container {{
            display: flex;
            flex-direction: column;
            align_items: center;
            justify_content: center;
            text-align: center;
            margin-bottom: 3rem;
            padding: 3rem 2rem;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0) 100%);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}
        .hero-logo-img {{
            max-width: 150px;
            margin-bottom: 1rem;
        }}
        .hero-title-large {{
            font-size: 3.5rem;
            font-weight: 800;
            margin: 0;
            background: -webkit-linear-gradient(45deg, {COLORS['primary']}, #00c6ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1.2;
        }}
        .hero-subtitle-large {{
            font-size: 1.25rem;
            font-weight: 500;
            color: #e0e0e0;
            margin-top: 0.5rem;
            margin-bottom: 1.5rem;
        }}
        .hero-summary {{
            font-size: 1.1rem;
            color: {COLORS['text_secondary']};
            max-width: 600px;
            line-height: 1.6;
            margin: 0 auto;
        }}

        /* --- Feature Pills (New) --- */
        .feature-pill {{
            display: inline-block;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 50px;
            padding: 0.35rem 0.85rem;
            margin: 0.3rem;
            font-size: 0.85rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.2s ease;
        }}
        .feature-pill:hover {{
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.3);
        }}
        
        /* Stats cards */
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="baseButton-primary"],
        .stButton button[kind="primary"],
        button[kind="primary"] {{
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_dark']} 100%) !important;
            color: {COLORS['bg_dark']} !important;
            border: 2px solid {COLORS['primary']} !important;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(0, 255, 136, 0.3);
        }}
        
        .stButton > button[kind="primary"]:hover,
        .stButton > button[data-testid="baseButton-primary"]:hover,
        .stButton button[kind="primary"]:hover,
        button[kind="primary"]:hover {{
            background: linear-gradient(135deg, {COLORS['primary_light']} 0%, {COLORS['primary']} 100%) !important;
            box-shadow: 0 6px 20px rgba(0, 255, 136, 0.4);
            transform: translateY(-1px);
        }}
        
        /* Secondary buttons (unselected state) - multiple selectors for compatibility */
        .stButton > button[kind="secondary"],
        .stButton > button[data-testid="baseButton-secondary"],
        .stButton button[kind="secondary"],
        button[kind="secondary"] {{
            background: {COLORS['bg_card']} !important;
            color: {COLORS['text_secondary']} !important;
            border: 1px solid {COLORS['border_light']} !important;
            font-weight: 500;
            box-shadow: none;
        }}
        
        .stButton > button[kind="secondary"]:hover,
        .stButton > button[data-testid="baseButton-secondary"]:hover,
        .stButton button[kind="secondary"]:hover,
        button[kind="secondary"]:hover {{
            background: {COLORS['bg_light']} !important;
            color: {COLORS['text_primary']} !important;
            border-color: {COLORS['primary_dark']} !important;
            transform: translateY(-1px);
        }}
        
        .stButton > button:active {{
            transform: translateY(0);
        }}
        
        /* Status badges */
        .status-badge {{
            display: inline-block;
            padding: {SPACING['sm']} {SPACING['md']};
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        .status-success {{
            background: rgba(0, 210, 106, 0.2);
            color: {COLORS['success']};
            border: 1px solid {COLORS['success']};
        }}
        
        .status-warning {{
            background: rgba(251, 191, 36, 0.2);
            color: {COLORS['warning']};
            border: 1px solid {COLORS['warning']};
        }}
        
        .status-error {{
            background: rgba(239, 68, 68, 0.2);
            color: {COLORS['error']};
            border: 1px solid {COLORS['error']};
        }}
        
        .status-info {{
            background: rgba(59, 130, 246, 0.2);
            color: {COLORS['info']};
            border: 1px solid {COLORS['info']};
        }}
        
        /* Dividers */
        .stDivider {{
            margin: {SPACING['xl']} 0;
        }}
        
        /* Style cards for mode selection */
        .style-card {{
            background: linear-gradient(145deg, {COLORS['bg_card']} 0%, {COLORS['bg_light']} 100%);
            border: 1px solid {COLORS['border_light']};
            border-radius: 10px;
            padding: {SPACING['md']};
            margin: {SPACING['sm']} 0;
            transition: all 0.2s;
            cursor: pointer;
        }}
        
        .style-card:hover {{
            border-color: {COLORS['primary']};
            box-shadow: 0 4px 12px rgba(0, 255, 136, 0.15);
        }}
        
        .style-card.selected {{
            border-color: {COLORS['primary']};
            background: linear-gradient(145deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 210, 106, 0.05) 100%);
        }}
        
        /* Video preview */
        .video-preview {{
            border: 2px solid {COLORS['border_medium']};
            border-radius: 12px;
            padding: {SPACING['lg']};
            background: {COLORS['bg_card']};
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }}
        
        /* Form inputs */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > select,
        .stNumberInput > div > div > input {{
            background-color: {COLORS['bg_light']};
            border: 1px solid {COLORS['border_light']};
            border-radius: 8px;
            color: {COLORS['text_primary']};
        }}
        
        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus,
        .stNumberInput > div > div > input:focus {{
            border-color: {COLORS['primary']};
            box-shadow: 0 0 0 1px {COLORS['primary']};
        }}
        
        /* Metrics */
        div[data-testid="stMetricValue"] {{
            font-size: 2rem;
            color: {COLORS['primary']};
        }}
        
        /* Info boxes */
        .stAlert {{
            border-radius: 8px;
            border-left-width: 4px;
        }}
        
        /* Expanders */
        .streamlit-expanderHeader {{
            background-color: {COLORS['bg_card']};
            border-radius: 8px;
            border: 1px solid {COLORS['border_light']};
        }}
        
        .streamlit-expanderHeader:hover {{
            border-color: {COLORS['primary_dark']};
        }}
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 1rem;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background-color: {COLORS['bg_card']};
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            border: 1px solid {COLORS['border_light']};
        }}
        
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_dark']} 100%);
            color: {COLORS['bg_dark']};
        }}
        
        /* Remove top padding from main container */
        .main > div {{
            padding-top: 1rem;
        }}
        </style>
    """
