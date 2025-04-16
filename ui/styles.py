# ui/styles.py
"""
CSS styling for the keyword clustering application.

This module provides functions to load CSS styles that are used 
throughout the application for consistent visual appearance.
"""

import streamlit as st

def load_css() -> None:
    """
    Load custom CSS styles for the application.
    """
    st.markdown("""
    <style>
        /* Main containers and layout */
        .main-header {
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid #eee;
        }
        
        .main-header h1 {
            font-size: 2.2rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #1E88E5;
        }
        
        .main-header .subtitle {
            font-size: 1.2rem;
            color: #666;
        }
        
        /* Cluster cards */
        .cluster-card {
            background-color: white;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .cluster-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.12);
        }
        
        .cluster-card h3 {
            margin-top: 0;
            font-size: 1.2rem;
            font-weight: 600;
            color: #333;
        }
        
        /* Metrics display */
        .metric-container {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 10px;
        }
        
        .metric-item {
            background-color: #f8f9fa;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 0.9rem;
            color: #555;
        }
        
        .metric-value {
            font-weight: 600;
            color: #1E88E5;
        }
        
        /* Intent badges */
        .intent-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .intent-informational {
            background-color: rgba(33, 150, 243, 0.2);
            color: #1565C0;
        }
        
        .intent-navigational {
            background-color: rgba(76, 175, 80, 0.2);
            color: #2E7D32;
        }
        
        .intent-transactional {
            background-color: rgba(255, 152, 0, 0.2);
            color: #E65100;
        }
        
        .intent-commercial {
            background-color: rgba(156, 39, 176, 0.2);
            color: #6A1B9A;
        }
        
        .intent-mixed-intent {
            background-color: rgba(117, 117, 117, 0.2);
            color: #424242;
        }
        
        /* Keyword pills */
        .keywords-container {
            margin-top: 12px;
        }
        
        .keyword-pill {
            display: inline-block;
            background-color: #f1f3f4;
            padding: 4px 10px;
            border-radius: 16px;
            margin: 2px;
            font-size: 0.85rem;
            color: #555;
            transition: background-color 0.2s;
        }
        
        .keyword-pill:hover {
            background-color: #e3f2fd;
        }
        
        /* Upload area */
        .upload-container {
            border: 2px dashed #ddd;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
            background-color: #f9f9f9;
        }
        
        .upload-container:hover {
            border-color: #1E88E5;
            background-color: #f5f5f5;
        }
        
        /* Filter controls */
        .filter-bar {
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        /* Journey visualization */
        .journey-phase {
            display: flex;
            width: 100%;
            margin: 10px 0;
        }
        
        .journey-step {
            flex: 1;
            text-align: center;
            padding: 8px 5px;
            font-size: 0.85rem;
            border-radius: 4px;
            margin: 0 2px;
        }
        
        .journey-step-active {
            font-weight: bold;
            color: white;
        }
        
        /* Improve spacing of default Streamlit elements */
        div[data-testid="stVerticalBlock"] > div:first-child {
            margin-top: 0;
        }
        
        /* Improve tabs styling */
        button[data-baseweb="tab"] {
            font-size: 1rem;
        }
        
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #1E88E5;
            font-weight: 600;
        }
        
        /* Make the Streamlit containers wider on large screens */
        .reportview-container .main .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Adjust header sizes */
        h1 {
            font-size: 2rem;
        }
        
        h2 {
            font-size: 1.5rem;
        }
        
        h3 {
            font-size: 1.2rem;
        }
        
        /* Tooltips */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        
        /* Content recommendations */
        .recommendation-item {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 8px;
            font-size: 0.9rem;
        }
        
        /* Opportunity highlight */
        .opportunity-section {
            border: 2px solid #ffd54f;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            background-color: #fffde7;
        }
        
        .opportunity-section h3 {
            color: #f57c00;
        }
    </style>
    """, unsafe_allow_html=True)

def inject_custom_js() -> None:
    """
    Inject custom JavaScript functionality if needed.
    """
    st.markdown("""
    <script>
    // This function can be used to add custom JavaScript
    document.addEventListener('DOMContentLoaded', function() {
        // Add any client-side functionality here
        
        // For example, we could add click animations to cards:
        const cards = document.querySelectorAll('.cluster-card');
        cards.forEach(card => {
            card.addEventListener('click', function() {
                this.style.transform = 'scale(0.98)';
                setTimeout(() => {
                    this.style.transform = 'translateY(-2px)';
                }, 100);
            });
        });
    });
    </script>
    """, unsafe_allow_html=True)

def apply_theme_overrides() -> None:
    """
    Apply theme overrides to match the application's design.
    
    Note: This should only be used for minor tweaks that can't be
    handled through Streamlit's config.toml theming.
    """
    # Unfortunately, we can't fully customize the Streamlit theme from code
    # without using external CSS libraries, but we can apply some basic tweaks
    st.markdown("""
    <style>
    /* Override Streamlit's default button colors to match our theme */
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        border: none;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #1976D2;
        color: white;
    }
    
    /* Improve slider appearance */
    .stSlider > div > div > div {
        background-color: #1E88E5 !important;
    }
    
    /* Override sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* Style radio buttons */
    .stRadio label {
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)
