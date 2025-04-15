# ui/__init__.py
"""
User interface components for the keyword clustering application.

This package contains all the UI-related functionality, including components,
page layouts, visualizations, and styling for the Streamlit-based application.
"""

# Import main UI functions
from ui.styles import load_css
from ui.components import display_cluster_card, show_filter_controls, show_metrics_summary
from ui.pages import show_welcome_page, show_results_dashboard
from ui.visualizations import show_intent_distribution, show_cluster_sizes

# Define UI version
__version__ = '1.0.0'
