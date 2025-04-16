# utils/state.py
"""
Session state management utilities for the keyword clustering application.

This module provides functions for initializing and updating the session state,
which is used to maintain data across re-runs in Streamlit.
"""

import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any

def initialize_session_state() -> None:
    """
    Initialize session state variables if they don't exist.
    """
    # Main state variables
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
        
    if 'cluster_insights' not in st.session_state:
        st.session_state.cluster_insights = None
    
    # View management
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "cluster_cards"
        
    if 'selected_cluster_id' not in st.session_state:
        st.session_state.selected_cluster_id = None
    
    # Cache state
    if 'df_cache' not in st.session_state:
        st.session_state.df_cache = {}
        
    if 'embedding_cache' not in st.session_state:
        st.session_state.embedding_cache = {}
        
    if 'api_cache' not in st.session_state:
        st.session_state.api_cache = {}
    
    # User settings
    if 'user_settings' not in st.session_state:
        st.session_state.user_settings = {
            "csv_format": "no_header",
            "num_clusters": 10,
            "use_openai": False,
            "openai_api_key": None,
            "language": "English",
            "use_gpt": False,
            "gpt_model": "gpt-3.5-turbo"
        }

def update_session_state(df: Optional[pd.DataFrame] = None,
                        cluster_insights: Optional[Dict[int, Dict[str, Any]]] = None,
                        processing_complete: Optional[bool] = None) -> None:
    """
    Update session state variables with new values.
    
    Args:
        df: Updated dataframe with clustering results
        cluster_insights: Updated cluster insights dictionary
        processing_complete: Updated processing status
    """
    if df is not None:
        st.session_state.results_df = df
    
    if cluster_insights is not None:
        st.session_state.cluster_insights = cluster_insights
    
    if processing_complete is not None:
        st.session_state.processing_complete = processing_complete

def reset_session_state() -> None:
    """
    Reset the session state to initial values, except for user settings.
    """
    user_settings = st.session_state.user_settings if 'user_settings' in st.session_state else None
    
    # Clear all session state
    for key in list(st.session_state.keys()):
        if key != 'user_settings':
            del st.session_state[key]
    
    # Re-initialize
    initialize_session_state()
    
    # Restore user settings
    if user_settings is not None:
        st.session_state.user_settings = user_settings

def get_current_view() -> str:
    """
    Get the current view from session state.
    
    Returns:
        Current view name
    """
    return st.session_state.get('current_view', 'cluster_cards')

def set_current_view(view_name: str) -> None:
    """
    Set the current view in session state.
    
    Args:
        view_name: Name of the view to set
    """
    st.session_state.current_view = view_name
