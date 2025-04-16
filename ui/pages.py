# ui/pages.py
"""
Page layout and view components for the keyword clustering application.

This module provides functions for rendering the main pages and views
of the application, such as the welcome page, results dashboard, etc.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple

# Import UI components
from ui.components import (
    display_cluster_card, 
    show_filter_controls,
    apply_filters,
    show_metrics_summary,
    show_cluster_detail_view,
    show_export_options,
    show_opportunity_clusters
)

# Import visualizations
from ui.visualizations import (
    show_intent_distribution,
    show_cluster_sizes,
    show_search_volume_by_intent,
    show_journey_distribution
)

# Import utility functions
from utils.helpers import generate_sample_csv
from modules.evaluation import identify_opportunity_clusters

def show_welcome_page() -> None:
    """
    Display the welcome/landing page with introduction and instructions.
    """
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Semantic Keyword Clustering</h1>
        <p class="subtitle">Group keywords by meaning and search intent for better content strategy</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content in two columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ## Why Use Keyword Clustering?
        
        Semantic keyword clustering helps you:
        
        - **Understand search intent** behind different groups of keywords
        - **Plan content more effectively** by addressing related topics together
        - **Identify content gaps** in your current strategy
        - **Optimize existing pages** with semantically related keywords
        - **Prioritize content creation** based on search volume and intent
        
        This tool uses advanced NLP and machine learning to group your keywords into
        meaningful clusters, analyze search intent, and provide actionable insights
        for your content strategy.
        """)
        
        # Show feature highlights
        st.markdown("""
        ## Key Features
        
        - **Semantic Clustering**: Group keywords by meaning, not just lexical similarity
        - **Search Intent Analysis**: Automatically classify keywords by user intent
        - **Customer Journey Mapping**: Understand where keywords fit in the buyer journey
        - **Content Recommendations**: Get suggestions for content types based on intent
        - **Visualization & Export**: Explore your keyword landscape and export results
        """)
    
    with col2:
        st.markdown("""
        ## Getting Started
        
        1. **Upload a CSV file** with your keywords
        2. **Select options** for clustering
        3. **Start the process** and let the algorithm work
        4. **Explore the results** and export your clusters
        
        Your CSV file should contain a list of keywords, either as:
        - A single column (no header)
        - Keywords with metadata (with header row)
        """)
        
        # File format examples
        st.markdown("""
        ### Example CSV Formats
        
        **Simple format (no header):**
        ```
        running shoes
        best running shoes
        nike running shoes
        ```
        
        **With header and metrics:**
        ```
        Keyword,search_volume,competition
        running shoes,5400,0.75
        best running shoes,2900,0.82
        ```
        """)
        
        # Sample CSV download
        st.markdown("### Need a sample file?")
        sample_csv = generate_sample_csv()
        st.download_button(
            "Download Sample CSV",
            sample_csv,
            file_name="sample_keywords.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Additional instructions in an expander
    with st.expander("Advanced Usage Tips", expanded=False):
        st.markdown("""
        ### Tips for Better Results
        
        - **Use related keywords**: For best results, use keywords from a single topic area
        - **Include sufficient data**: At least 50-100 keywords is recommended for meaningful clusters
        - **Include search volume data**: If available, include search volume for better insights
        - **Try different cluster counts**: The optimal number of clusters varies by keyword set
        - **Consider language**: The tool works best with English keywords, but supports other languages
        
        ### Working with Large Keyword Sets
        
        For very large keyword sets (>10,000 keywords), consider:
        
        - Breaking into smaller related groups
        - Increasing the number of clusters
        - Using OpenAI embeddings for higher quality (requires API key)
        
        ### Understanding Search Intent
        
        The tool classifies keywords into four intent categories:
        
        - **Informational**: Users seeking information or answers ("how to", "what is")
        - **Navigational**: Users trying to find a specific website or resource
        - **Transactional**: Users looking to make a purchase or complete an action
        - **Commercial**: Users researching products before making a purchase decision
        """)

def handle_cluster_click(cluster_id: int) -> None:
    """
    Handle a click on a cluster card.
    
    Args:
        cluster_id: ID of the clicked cluster
    """
    # Store the selected cluster ID in session state
    st.session_state.selected_cluster_id = cluster_id
    
    # Set view to cluster detail
    st.session_state.current_view = "cluster_detail"

def show_results_dashboard(df: pd.DataFrame, 
                         cluster_insights: Dict[int, Dict[str, Any]]) -> None:
    """
    Display the main results dashboard after clustering is complete.
    
    Args:
        df: Dataframe with clustering results
        cluster_insights: Dictionary of cluster insights
    """
    # Initialize session state for view management if needed
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "cluster_cards"
    
    if 'selected_cluster_id' not in st.session_state:
        st.session_state.selected_cluster_id = None
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Clustering Results</h1>
        <p class="subtitle">Explore and analyze your keyword clusters</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different views
    tabs = ["Cluster Cards", "Visualizations", "Export Data"]
    selected_tab = st.radio("View:", tabs, horizontal=True)
    
    # Handle view switching based on tab selection
    if selected_tab == "Cluster Cards":
        st.session_state.current_view = "cluster_cards" if st.session_state.current_view != "cluster_detail" else st.session_state.current_view
    else:
        st.session_state.current_view = selected_tab.lower().replace(" ", "_")
    
    # Back button for cluster detail view
    if st.session_state.current_view == "cluster_detail" and st.session_state.selected_cluster_id is not None:
        if st.button("← Back to All Clusters"):
            st.session_state.current_view = "cluster_cards"
            st.session_state.selected_cluster_id = None
            st.experimental_rerun()
    
    # Show appropriate view based on state
    if st.session_state.current_view == "cluster_detail" and st.session_state.selected_cluster_id is not None:
        # Show detailed view for selected cluster
        show_cluster_detail_view(
            st.session_state.selected_cluster_id,
            df,
            cluster_insights.get(st.session_state.selected_cluster_id, {})
        )
    elif st.session_state.current_view == "cluster_cards":
        show_clusters_view(df, cluster_insights)
    elif st.session_state.current_view == "visualizations":
        show_visualizations_view(df, cluster_insights)
    elif st.session_state.current_view == "export_data":
        show_export_view(df, cluster_insights)

def show_clusters_view(df: pd.DataFrame, 
                     cluster_insights: Dict[int, Dict[str, Any]]) -> None:
    """
    Display the main clusters view with cards.
    
    Args:
        df: Dataframe with clustering results
        cluster_insights: Dictionary of cluster insights
    """
    # Show summary metrics
    show_metrics_summary(cluster_insights)
    
    # Add filter controls
    filters = show_filter_controls(df, cluster_insights)
    
    # Apply filters
    filtered_cluster_ids = apply_filters(cluster_insights, filters)
    
    # Show filter results summary
    st.markdown(f"### Showing {len(filtered_cluster_ids)} of {len(cluster_insights)} clusters")
    
    # Show opportunity clusters if we have enough data
    if len(cluster_insights) >= 5:
        opportunities = identify_opportunity_clusters(cluster_insights)
        show_opportunity_clusters(
            opportunities, 
            cluster_insights, 
            df,
            handle_cluster_click
        )
    
    # Create grid layout for cluster cards
    if filtered_cluster_ids:
        st.markdown("### All Clusters")
        
        # Use columns to create a grid
        cols = st.columns(2)
        for i, cluster_id in enumerate(filtered_cluster_ids):
            with cols[i % 2]:
                display_cluster_card(
                    cluster_id, 
                    cluster_insights[cluster_id], 
                    df,
                    handle_cluster_click
                )
    else:
        st.info("No clusters match your current filters. Try adjusting your filter criteria.")

def show_visualizations_view(df: pd.DataFrame, 
                           cluster_insights: Dict[int, Dict[str, Any]]) -> None:
    """
    Display visualizations of the clustering results.
    
    Args:
        df: Dataframe with clustering results
        cluster_insights: Dictionary of cluster insights
    """
    st.markdown("## Keyword Clustering Insights")
    
    # Create a 2x2 grid of visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Intent distribution
        show_intent_distribution(cluster_insights)
    
    with col2:
        # Cluster sizes visualization
        show_cluster_sizes(cluster_insights)
    
    # Second row
    col1, col2 = st.columns(2)
    
    with col1:
        # Search volume by intent (if available)
        has_volume = any('total_volume' in insights for insights in cluster_insights.values())
        if has_volume:
            show_search_volume_by_intent(cluster_insights)
        else:
            # Journey distribution as alternative
            show_journey_distribution(cluster_insights)
    
    with col2:
        # Additional visualization - Journey phases
        if has_volume:
            show_journey_distribution(cluster_insights)
        else:
            # Quality score distribution
            st.markdown("### Cluster Quality Distribution")
            st.info("This visualization requires search volume data, which is not available in your current dataset.")
    
    # Add additional insights in an expander
    with st.expander("Additional Insights", expanded=False):
        st.markdown("### Keyword Insights")
        
        # Word count distribution
        df['word_count'] = df['keyword'].apply(lambda x: len(str(x).split()))
        avg_word_count = df['word_count'].mean()
        
        st.markdown(f"**Average words per keyword:** {avg_word_count:.1f}")
        
        # Intent breakdown
        if 'primary_intent' in df.columns:
            intent_counts = df['primary_intent'].value_counts()
            st.markdown("**Intent breakdown:**")
            
            for intent, count in intent_counts.items():
                percentage = (count / len(df)) * 100
                st.markdown(f"- {intent}: {count} keywords ({percentage:.1f}%)")
        
        # Keyword examples by intent
        if 'primary_intent' in df.columns:
            st.markdown("**Example keywords by intent:**")
            
            for intent in df['primary_intent'].unique():
                examples = df[df['primary_intent'] == intent]['keyword'].tolist()[:3]
                if examples:
                    examples_str = ", ".join([f"`{ex}`" for ex in examples])
                    st.markdown(f"- **{intent}**: {examples_str}")

def show_export_view(df: pd.DataFrame, 
                   cluster_insights: Dict[int, Dict[str, Any]]) -> None:
    """
    Display export options and data preview.
    
    Args:
        df: Dataframe with clustering results
        cluster_insights: Dictionary of cluster insights
    """
    st.markdown("## Export Clustered Keywords")
    
    # Show export options
    show_export_options(df, cluster_insights)
    
    # Show data preview
    with st.expander("Data Preview", expanded=True):
        st.dataframe(
            df.head(10),
            use_container_width=True
        )
    
    # Show summary for export
    st.markdown("### Export Summary")
    
    # Create summary information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Cluster Statistics:**")
        
        total_clusters = len(cluster_insights)
        total_keywords = len(df)
        
        st.markdown(f"- Total clusters: {total_clusters}")
        st.markdown(f"- Total keywords: {total_keywords}")
        st.markdown(f"- Average keywords per cluster: {total_keywords / max(1, total_clusters):.1f}")
        
        # Intent distribution if available
        if any('primary_intent' in insights for insights in cluster_insights.values()):
            st.markdown("**Intent Distribution:**")
            intent_counts = {}
            
            for insights in cluster_insights.values():
                intent = insights.get('primary_intent', 'Unknown')
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
            for intent, count in intent_counts.items():
                st.markdown(f"- {intent}: {count} clusters ({(count / total_clusters) * 100:.1f}%)")
    
    with col2:
        st.markdown("**Top Clusters by Size:**")
        
        # Sort clusters by size
        sorted_clusters = sorted(
            cluster_insights.items(),
            key=lambda x: x[1].get('keyword_count', 0),
            reverse=True
        )
        
        # Show top 5
        for i, (cluster_id, insights) in enumerate(sorted_clusters[:5]):
            name = insights.get('cluster_name', f"Cluster {cluster_id}")
            count = insights.get('keyword_count', 0)
            st.markdown(f"{i+1}. **{name}**: {count} keywords")
        
        # Show top by search volume if available
        has_volume = any('total_volume' in insights for insights in cluster_insights.values())
        if has_volume:
            st.markdown("**Top Clusters by Search Volume:**")
            
            # Sort by volume
            volume_sorted = sorted(
                cluster_insights.items(),
                key=lambda x: x[1].get('total_volume', 0),
                reverse=True
            )
            
            # Show top 5
            for i, (cluster_id, insights) in enumerate(volume_sorted[:5]):
                if 'total_volume' in insights:
                    name = insights.get('cluster_name', f"Cluster {cluster_id}")
                    volume = insights.get('total_volume', 0)
                    st.markdown(f"{i+1}. **{name}**: {volume:,} monthly searches")
