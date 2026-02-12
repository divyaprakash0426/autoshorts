"""Shared theme and styling for the dashboard.

Provides consistent colors, spacing, and UI components across all pages.
"""

# Color palette
COLORS = {
    # Primary
    "primary": "#6C63FF",
    "primary_dark": "#5A52D5",
    "primary_light": "#8B85FF",
    
    # Background
    "bg_dark": "#0F1117",
    "bg_medium": "#1A1D2E",
    "bg_light": "#252836",
    "bg_card": "#1E2030",
    
    # Accents
    "accent_blue": "#1E3A5F",
    "accent_purple": "#7C3AED",
    "accent_orange": "#FF6B35",
    
    # Status
    "success": "#34D399",
    "warning": "#FBBF24",
    "error": "#F87171",
    "info": "#60A5FA",
    
    # Text
    "text_primary": "#F1F5F9",
    "text_secondary": "#94A3B8",
    "text_muted": "#64748B",
    
    # Borders
    "border_light": "#2D3348",
    "border_medium": "#3D4463",
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
        /* Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* Global resets */
        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        
        /* Remove Streamlit branding colors */
        .stApp {{
            background-color: {COLORS['bg_dark']};
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-weight: 600;
            letter-spacing: -0.025em;
            color: {COLORS['text_primary']};
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

        p, span, div, li, label {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}
        
        /* Hero sections */
        .hero-section {{
            background: linear-gradient(135deg, {COLORS['bg_medium']} 0%, {COLORS['accent_blue']} 50%, {COLORS['bg_medium']} 100%);
            padding: {SPACING['2xl']};
            border-radius: 16px;
            text-align: center;
            margin-bottom: {SPACING['xl']};
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            border: 1px solid {COLORS['border_light']};
        }}
        
        .hero-title {{
            font-size: 3rem;
            font-weight: 700;
            margin: {SPACING['md']} 0;
            background: linear-gradient(135deg, {COLORS['primary_light']} 0%, {COLORS['primary']} 50%, {COLORS['accent_purple']} 100%);
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
            background: {COLORS['bg_card']};
            border: 1px solid {COLORS['border_light']};
            border-radius: 12px;
            padding: {SPACING['lg']};
            height: 100%;
            transition: all 0.25s ease;
        }}
        
        .feature-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 24px rgba(108, 99, 255, 0.12);
            border-color: {COLORS['primary']};
        }}
        
        .feature-card h4 {{
            color: {COLORS['primary_light']};
            margin-top: 0;
            margin-bottom: {SPACING['md']};
            font-size: 1.2rem;
        }}
        
        .feature-card ul {{
            margin: 0;
            padding-left: 1.25rem;
        }}
        
        .feature-card li {{
            margin-bottom: {SPACING['sm']};
            line-height: 1.6;
            color: {COLORS['text_secondary']};
        }}
        
        /* Stats cards */
        .stats-card {{
            background: linear-gradient(145deg, rgba(108, 99, 255, 0.06) 0%, rgba(90, 82, 213, 0.1) 100%);
            border: 1px solid {COLORS['border_light']};
            border-radius: 12px;
            padding: {SPACING['lg']};
            text-align: center;
            transition: all 0.25s ease;
        }}
        
        .stats-card:hover {{
            transform: scale(1.02);
            border-color: {COLORS['primary']};
            box-shadow: 0 4px 16px rgba(108, 99, 255, 0.1);
        }}
        
        .stats-value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: {COLORS['primary_light']};
            line-height: 1;
        }}
        
        .stats-label {{
            font-size: 0.85rem;
            color: {COLORS['text_secondary']};
            margin-top: {SPACING['sm']};
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 500;
        }}
        
        /* Header sections */
        .page-header {{
            background: linear-gradient(135deg, {COLORS['bg_medium']} 0%, {COLORS['accent_blue']} 100%);
            padding: {SPACING['xl']};
            border-radius: 12px;
            margin-bottom: {SPACING['xl']};
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
            border: 1px solid {COLORS['border_light']};
        }}
        
        .page-header h1 {{
            margin: 0;
            color: {COLORS['text_primary']};
        }}
        
        .page-header p {{
            margin: {SPACING['sm']} 0 0 0;
            color: {COLORS['text_secondary']};
            font-size: 1.1rem;
        }}
        
        /* Base button styles */
        .stButton > button {{
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-size: 0.95rem;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            transition: all 0.25s ease;
        }}

        /* --- Hero Container --- */
        .hero-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            margin-bottom: 3rem;
            padding: 3rem 2rem;
            background: linear-gradient(180deg, rgba(108, 99, 255, 0.04) 0%, rgba(108, 99, 255, 0) 100%);
            border-radius: 24px;
            border: 1px solid {COLORS['border_light']};
        }}
        .hero-logo-img {{
            max-width: 150px;
            margin-bottom: 1rem;
        }}
        .hero-title-large {{
            font-size: 3.5rem;
            font-weight: 800;
            margin: 0;
            background: linear-gradient(135deg, {COLORS['primary_light']}, {COLORS['primary']}, {COLORS['accent_purple']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1.2;
        }}
        .hero-subtitle-large {{
            font-size: 1.25rem;
            font-weight: 400;
            color: {COLORS['text_secondary']};
            margin-top: 0.5rem;
            margin-bottom: 1.5rem;
        }}
        .hero-summary {{
            font-size: 1.05rem;
            color: {COLORS['text_secondary']};
            max-width: 600px;
            line-height: 1.7;
            margin: 0 auto;
        }}

        /* --- Feature Pills --- */
        .feature-pill {{
            display: inline-block;
            background: rgba(108, 99, 255, 0.08);
            border-radius: 50px;
            padding: 0.35rem 0.85rem;
            margin: 0.3rem;
            font-size: 0.85rem;
            border: 1px solid rgba(108, 99, 255, 0.2);
            color: {COLORS['text_secondary']};
            transition: all 0.25s ease;
        }}
        .feature-pill:hover {{
            background: rgba(108, 99, 255, 0.15);
            border-color: rgba(108, 99, 255, 0.4);
            color: {COLORS['text_primary']};
        }}
        
        /* Primary buttons */
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="baseButton-primary"],
        .stButton button[kind="primary"],
        button[kind="primary"] {{
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_dark']} 100%) !important;
            color: #FFFFFF !important;
            border: 1px solid {COLORS['primary']} !important;
            font-weight: 600;
            box-shadow: 0 4px 14px rgba(108, 99, 255, 0.3);
        }}
        
        .stButton > button[kind="primary"]:hover,
        .stButton > button[data-testid="baseButton-primary"]:hover,
        .stButton button[kind="primary"]:hover,
        button[kind="primary"]:hover {{
            background: linear-gradient(135deg, {COLORS['primary_light']} 0%, {COLORS['primary']} 100%) !important;
            box-shadow: 0 6px 20px rgba(108, 99, 255, 0.4);
            transform: translateY(-1px);
        }}
        
        /* Secondary buttons */
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
            border-color: {COLORS['primary']} !important;
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
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        .status-success {{
            background: rgba(52, 211, 153, 0.15);
            color: {COLORS['success']};
            border: 1px solid rgba(52, 211, 153, 0.3);
        }}
        
        .status-warning {{
            background: rgba(251, 191, 36, 0.15);
            color: {COLORS['warning']};
            border: 1px solid rgba(251, 191, 36, 0.3);
        }}
        
        .status-error {{
            background: rgba(248, 113, 113, 0.15);
            color: {COLORS['error']};
            border: 1px solid rgba(248, 113, 113, 0.3);
        }}
        
        .status-info {{
            background: rgba(96, 165, 250, 0.15);
            color: {COLORS['info']};
            border: 1px solid rgba(96, 165, 250, 0.3);
        }}
        
        /* Dividers */
        .stDivider {{
            margin: {SPACING['xl']} 0;
        }}
        
        /* Style cards for mode selection */
        .style-card {{
            background: {COLORS['bg_card']};
            border: 1px solid {COLORS['border_light']};
            border-radius: 10px;
            padding: {SPACING['md']};
            margin: {SPACING['sm']} 0;
            transition: all 0.25s ease;
            cursor: pointer;
        }}
        
        .style-card:hover {{
            border-color: {COLORS['primary']};
            box-shadow: 0 4px 12px rgba(108, 99, 255, 0.12);
        }}
        
        .style-card.selected {{
            border-color: {COLORS['primary']};
            background: linear-gradient(145deg, rgba(108, 99, 255, 0.1) 0%, rgba(90, 82, 213, 0.05) 100%);
        }}
        
        /* Video preview */
        .video-preview {{
            border: 1px solid {COLORS['border_medium']};
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
            font-family: 'Inter', sans-serif;
        }}
        
        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus,
        .stNumberInput > div > div > input:focus {{
            border-color: {COLORS['primary']};
            box-shadow: 0 0 0 2px rgba(108, 99, 255, 0.2);
        }}
        
        /* Metrics */
        div[data-testid="stMetricValue"] {{
            font-size: 2rem;
            color: {COLORS['primary_light']};
            font-family: 'Inter', sans-serif;
            font-weight: 700;
        }}
        
        /* Info boxes */
        .stAlert {{
            border-radius: 10px;
            border-left-width: 4px;
        }}
        
        /* Expanders */
        .streamlit-expanderHeader {{
            background-color: {COLORS['bg_card']};
            border-radius: 8px;
            border: 1px solid {COLORS['border_light']};
        }}
        
        .streamlit-expanderHeader:hover {{
            border-color: {COLORS['primary']};
        }}
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.5rem;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background-color: {COLORS['bg_card']};
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            border: 1px solid {COLORS['border_light']};
            font-family: 'Inter', sans-serif;
            transition: all 0.25s ease;
        }}
        
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_dark']} 100%);
            color: #FFFFFF;
            border-color: {COLORS['primary']};
        }}

        /* Step cards for Getting Started */
        .step-card {{
            background: {COLORS['bg_card']};
            border: 1px solid {COLORS['border_light']};
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            transition: all 0.25s ease;
            height: 100%;
        }}
        .step-card:hover {{
            border-color: {COLORS['primary']};
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(108, 99, 255, 0.1);
        }}
        .step-number {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_dark']} 100%);
            color: #FFFFFF;
            border-radius: 50%;
            font-weight: 700;
            font-size: 1rem;
            margin-bottom: 0.75rem;
        }}
        .step-card h4 {{
            margin: 0.5rem 0 0.4rem 0;
            font-size: 1rem;
            color: {COLORS['text_primary']};
        }}
        .step-card p {{
            margin: 0;
            font-size: 0.88rem;
            color: {COLORS['text_secondary']};
            line-height: 1.5;
        }}
        
        /* Remove top padding from main container */
        .main > div {{
            padding-top: 1rem;
        }}

        /* Scrollbar styling */
        ::-webkit-scrollbar {{
            width: 6px;
            height: 6px;
        }}
        ::-webkit-scrollbar-track {{
            background: {COLORS['bg_dark']};
        }}
        ::-webkit-scrollbar-thumb {{
            background: {COLORS['border_medium']};
            border-radius: 3px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: {COLORS['primary']};
        }}
        </style>
    """
