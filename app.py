import os
import time
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Module imports
from modules.preprocessing import preprocess_keywords
from modules.embeddings import generate_embeddings, check_embedding_models
from modules.clustering import run_clustering, refine_clusters
from modules.search_intent import classify_cluster_intents
from modules.naming import generate_cluster_names
from modules.evaluation import evaluate_cluster_quality, analyze_cluster_insights

# UI imports
from ui.styles import load_css
from ui.components import display_cluster_card, show_filter_controls, show_metrics_summary
from ui.pages import show_welcome_page, show_results_dashboard
from ui.visualizations import show_intent_distribution, show_cluster_sizes

# Utility imports
from utils.file_handlers import process_uploaded_file, export_results
from utils.constants import INTENT_COLORS, DEFAULT_NUM_CLUSTERS, APP_NAME, APP_DESCRIPTION
from utils.helpers import generate_sample_csv
from utils.state import initialize_session_state, update_session_state

# Page configuration
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS styles
load_css()

# Initialize session state
initialize_session_state()

# Check model availability
openai_available, sentence_transformers_available, spacy_available = check_embedding_models()

def main():
    # Sidebar
    with st.sidebar:
        st.markdown(f"## {APP_NAME}")
        st.markdown("### 1. Import Keywords")
        
        # CSV format options
        csv_format = st.radio(
            "CSV Format:",
            options=["No header (one keyword per line)", "With header (Keyword Planner format)"],
            index=0,
            help="Select the format of your CSV file"
        )
        format_type = "no_header" if "No header" in csv_format else "with_header"
        
        # File upload
        uploaded_file = st.file_uploader("Upload your keywords CSV file", type=['csv'])
        
        # Sample CSV download
        if st.button("Download sample CSV"):
            sample_csv = generate_sample_csv()
            st.download_button(
                label="Download sample",
                data=sample_csv,
                file_name="sample_keywords.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        st.markdown("### 2. Clustering Options")
        
        # Basic options
        num_clusters = st.slider(
            "Number of clusters",
            min_value=2, 
            max_value=50, 
            value=DEFAULT_NUM_CLUSTERS,
            help="Higher number = more specific clusters, Lower number = broader clusters"
        )
        
        # Embedding options
        embedding_option = st.selectbox(
            "Embedding method:",
            options=[
                "SentenceTransformers (free, good quality)" if sentence_transformers_available else "SentenceTransformers (not installed)",
                "OpenAI Embeddings (API key required, highest quality)" if openai_available else "OpenAI (not installed)",
                "TF-IDF (basic, always available)"
            ],
            index=0 if sentence_transformers_available else 2
        )
        
        use_openai = "OpenAI" in embedding_option and openai_available
        
        if use_openai:
            openai_api_key = st.text_input("OpenAI API Key:", type="password")
        else:
            openai_api_key = None
            
        # Advanced options in an expander
        with st.expander("Advanced Options", expanded=False):
            spacy_language = st.selectbox(
                "Language for text processing:",
                options=["English", "Spanish", "French", "German", "Others (basic processing)"],
                index=0
            )
            
            st.markdown("#### AI-powered cluster naming")
            use_gpt = st.checkbox("Use OpenAI to name clusters", value=use_openai)
            
            if use_gpt:
                if not use_openai:
                    openai_api_key = st.text_input("OpenAI API Key for naming:", type="password")
                
                gpt_model = st.selectbox(
                    "OpenAI model for naming:",
                    options=["gpt-3.5-turbo", "gpt-4"],
                    index=0
                )
            else:
                gpt_model = None
        
        # Start processing button
        start_button = st.button(
            "Start Clustering" if uploaded_file else "Upload a file to begin",
            disabled=not uploaded_file,
            use_container_width=True
        )
        
        # Reset button (only shown after processing)
        if st.session_state.processing_complete:
            if st.button("Reset & Start Over", use_container_width=True):
                for key in list(st.session_state.keys()):
                    if key != "user_settings":
                        del st.session_state[key]
                st.session_state.processing_complete = False
                st.experimental_rerun()
    
    # Main content area
    if not st.session_state.processing_complete:
        show_welcome_page()
        
        # If no file uploaded, do nothing more
        if not uploaded_file:
            return
            
        # Processing when button is clicked
        if start_button:
            with st.spinner("Processing your keywords..."):
                # 1. Load and process the file
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                progress_text.text("Reading and preprocessing your keywords...")
                df = process_uploaded_file(uploaded_file, format_type)
                progress_bar.progress(0.2)
                
                if df is not None:
                    # Save user settings
                    st.session_state.user_settings = {
                        "csv_format": format_type,
                        "num_clusters": num_clusters,
                        "use_openai": use_openai,
                        "openai_api_key": openai_api_key,
                        "language": spacy_language,
                        "use_gpt": use_gpt,
                        "gpt_model": gpt_model
                    }
                    
                    # 2. Preprocess keywords
                    progress_text.text("Preprocessing keywords...")
                    df = preprocess_keywords(df, spacy_language)
                    progress_bar.progress(0.3)
                    
                    # 3. Generate embeddings
                    progress_text.text("Generating semantic embeddings...")
                    embeddings = generate_embeddings(
                        df, 
                        use_openai=use_openai, 
                        api_key=openai_api_key
                    )
                    progress_bar.progress(0.5)
                    
                    # 4. Apply clustering
                    progress_text.text("Clustering keywords by semantic similarity...")
                    clustered_df = run_clustering(df, embeddings, num_clusters)
                    progress_bar.progress(0.6)
                    
                    # 5. Refine clusters
                    progress_text.text("Refining clusters...")
                    clustered_df = refine_clusters(clustered_df, embeddings)
                    progress_bar.progress(0.7)
                    
                    # 6. Evaluate cluster quality
                    progress_text.text("Evaluating cluster quality...")
                    clustered_df = evaluate_cluster_quality(clustered_df, embeddings)
                    progress_bar.progress(0.8)
                    
                    # 7. Classify search intent
                    progress_text.text("Analyzing search intent...")
                    cluster_intents = classify_cluster_intents(clustered_df)
                    progress_bar.progress(0.85)
                    
                    # 8. Generate cluster names if using GPT
                    if use_gpt and openai_api_key:
                        progress_text.text("Generating cluster names with AI...")
                        cluster_names = generate_cluster_names(
                            clustered_df, 
                            api_key=openai_api_key, 
                            model=gpt_model
                        )
                        progress_bar.progress(0.9)
                    else:
                        # Generic names if not using AI
                        cluster_names = {
                            cluster_id: f"Cluster {cluster_id}"
                            for cluster_id in clustered_df['cluster_id'].unique()
                        }
                    
                    # 9. Add names to clusters
                    for cluster_id, name in cluster_names.items():
                        clustered_df.loc[clustered_df['cluster_id'] == cluster_id, 'cluster_name'] = name
                    
                    # 10. Generate additional insights
                    progress_text.text("Generating advanced insights...")
                    cluster_insights = analyze_cluster_insights(
                        clustered_df, 
                        cluster_intents, 
                        cluster_names
                    )
                    progress_bar.progress(1.0)
                    
                    # Save results to session
                    update_session_state(
                        df=clustered_df,
                        cluster_insights=cluster_insights,
                        processing_complete=True
                    )
                    
                    # Clear indicators
                    progress_text.empty()
                    progress_bar.empty()
                    
                    # Force rerun to show results
                    st.experimental_rerun()
                else:
                    st.error("There was an error processing your file. Please check the format and try again.")
    else:
        # Show results dashboard
        show_results_dashboard(
            st.session_state.results_df,
            st.session_state.cluster_insights
        )

if __name__ == "__main__":
    main()
