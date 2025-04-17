# ui/components.py
"""
UI components for the keyword clustering application.

This module provides reusable Streamlit UI components for displaying
cluster cards, filter controls, metrics, and other UI elements.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
import plotly.express as px
import plotly.graph_objects as go

# Import utility functions
from utils.constants import INTENT_COLORS

def display_cluster_card(cluster_id: int, 
                       insights: Dict[str, Any], 
                       df: pd.DataFrame,
                       on_click_callback: Optional[Callable] = None,
                       card_index: int = 0) -> None:  # Add card_index parameter
    """
    Display a single cluster card with all relevant information.
    
    Args:
        cluster_id: The ID of the cluster to display
        insights: Dictionary containing cluster insights
        df: The full dataframe with all clusters
        on_click_callback: Optional callback function when card is clicked
        card_index: Index of the card in the display (used for unique keys)
    """
    # Get cluster data
    cluster_name = insights.get('cluster_name', f"Cluster {cluster_id}")
    keyword_count = insights.get('keyword_count', 0)
    rep_keywords = insights.get('representative_keywords', [])[:5]  # Show top 5
    primary_intent = insights.get('primary_intent', 'Unknown')
    
    # Get search volume if available
    total_volume = insights.get('total_volume', None)
    
    # Get quality score
    quality_score = insights.get('quality_score', None)
    if quality_score is not None:
        quality_score = round(quality_score, 1)
    
    # Get color based on intent
    intent_color = INTENT_COLORS.get(primary_intent, "#757575")
    intent_class = primary_intent.lower().replace(" ", "-")
    
    # Create card HTML
    card_html = f"""
    <div class="cluster-card" style="border-left: 4px solid {intent_color};">
        <h3 style="margin-top: 0; margin-bottom: 10px;">{cluster_name}</h3>
        
        <div class="metric-container">
            <div class="metric-item">
                Keywords: <span class="metric-value">{keyword_count}</span>
            </div>
            <div class="metric-item">
                <span class="intent-badge intent-{intent_class}" 
                      style="background-color: {intent_color}20; color: {intent_color};">
                    {primary_intent}
                </span>
            </div>
    """
    
    # Add search volume if available
    if total_volume is not None:
        card_html += f"""
            <div class="metric-item">
                Volume: <span class="metric-value">{total_volume:,}</span>
            </div>
        """
    
    # Add quality score if available
    if quality_score is not None:
        quality_color = "#4CAF50" if quality_score >= 7 else "#FF9800" if quality_score >= 5 else "#F44336"
        card_html += f"""
            <div class="metric-item">
                Quality: <span class="metric-value" style="color: {quality_color};">{quality_score}</span>
            </div>
        """
    
    # Close metric container and add description
    description = insights.get('cluster_description', '')
    if description:
        card_html += f"""
        </div>
        <div style="font-size: 14px; margin-bottom: 10px; color: #555;">
            {description}
        </div>
        """
    else:
        card_html += "</div>"
    
    # Add representative keywords
    if rep_keywords:
        card_html += '<div class="keywords-container">'
        for kw in rep_keywords:
            card_html += f'<span class="keyword-pill">{kw}</span>'
        
        if keyword_count > len(rep_keywords):
            card_html += f'<span class="keyword-pill">+{keyword_count - len(rep_keywords)} more</span>'
        
        card_html += '</div>'
    
    # Close card
    card_html += "</div>"
    
    # Display the card
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Add a button within the same container for viewing details
    if on_click_callback:
        # Create a truly unique key for each button that won't cause conflicts
        # Use both cluster_id and card_index to guarantee uniqueness
        unique_key = f"view_cluster_{cluster_id}_{card_index}_{hash(str(insights))}"
        if st.button(f"View Details", key=unique_key, use_container_width=True):
            on_click_callback(cluster_id)

def show_filter_controls(df: pd.DataFrame, cluster_insights: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Display filter controls for clusters and return the filter settings.
    
    Args:
        df: Dataframe with all keyword data
        cluster_insights: Dictionary of cluster insights
        
    Returns:
        Dictionary of filter settings
    """
    st.markdown("### Filter Clusters")
    
    # Create columns for filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Intent filter
        intents = ["All Intents"]
        if cluster_insights:
            # Get unique intents from insights
            unique_intents = set()
            for insights in cluster_insights.values():
                intent = insights.get('primary_intent')
                if intent:
                    unique_intents.add(intent)
            
            intents.extend(sorted(list(unique_intents)))
        
        intent_filter = st.selectbox(
            "Search Intent:",
            options=intents
        )
    
    with col2:
        # Size filter
        min_size = st.number_input(
            "Min. Keywords:",
            min_value=1,
            value=5,
            step=5
        )
    
    with col3:
        # Sort options
        sort_options = ["Size (Largest First)", "Quality Score", "Search Volume (if available)", "Alphabetical"]
        sort_by = st.selectbox(
            "Sort By:",
            options=sort_options
        )
    
    # Advanced filters in an expander
    with st.expander("Advanced Filters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            # Journey phase filter
            journey_phases = ["All Journey Phases"]
            if cluster_insights:
                # Get unique journey phases
                unique_phases = set()
                for insights in cluster_insights.values():
                    phase = insights.get('journey_phase')
                    if phase:
                        unique_phases.add(phase)
                
                journey_phases.extend(sorted(list(unique_phases)))
            
            journey_filter = st.selectbox(
                "Customer Journey Phase:",
                options=journey_phases
            )
        
        with col2:
            # Quality filter
            min_quality = st.slider(
                "Min. Quality Score:",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.5
            )
    
    # Return all filter settings
    return {
        "intent_filter": intent_filter,
        "min_size": min_size,
        "sort_by": sort_by,
        "journey_filter": journey_filter,
        "min_quality": min_quality
    }

def apply_filters(cluster_insights: Dict[int, Dict[str, Any]], 
                filters: Dict[str, Any]) -> List[int]:
    """
    Apply filters to cluster insights and return filtered cluster IDs.
    
    Args:
        cluster_insights: Dictionary of cluster insights
        filters: Dictionary of filter settings
        
    Returns:
        List of filtered cluster IDs
    """
    filtered_ids = []
    
    for cluster_id, insights in cluster_insights.items():
        # Check size filter
        if insights.get('keyword_count', 0) < filters['min_size']:
            continue
        
        # Check intent filter
        if filters['intent_filter'] != "All Intents" and insights.get('primary_intent') != filters['intent_filter']:
            continue
        
        # Check journey filter
        if filters['journey_filter'] != "All Journey Phases" and insights.get('journey_phase') != filters['journey_filter']:
            continue
        
        # Check quality filter
        if insights.get('quality_score', 0) < filters['min_quality']:
            continue
        
        # If we get here, all filters passed
        filtered_ids.append(cluster_id)
    
    # Sort the clusters
    sort_by = filters['sort_by']
    if sort_by == "Size (Largest First)":
        filtered_ids.sort(key=lambda cid: cluster_insights[cid].get('keyword_count', 0), reverse=True)
    elif sort_by == "Quality Score":
        filtered_ids.sort(key=lambda cid: cluster_insights[cid].get('quality_score', 0), reverse=True)
    elif sort_by == "Search Volume (if available)":
        filtered_ids.sort(key=lambda cid: cluster_insights[cid].get('total_volume', 0), reverse=True)
    elif sort_by == "Alphabetical":
        filtered_ids.sort(key=lambda cid: cluster_insights[cid].get('cluster_name', f"Cluster {cid}"))
    
    return filtered_ids

def show_metrics_summary(cluster_insights: Dict[int, Dict[str, Any]]) -> None:
    """
    Display summary metrics for all clusters.
    
    Args:
        cluster_insights: Dictionary of cluster insights
    """
    if not cluster_insights:
        return
    
    # Calculate metrics
    total_keywords = sum(insights.get('keyword_count', 0) for insights in cluster_insights.values())
    total_clusters = len(cluster_insights)
    
    # Calculate total search volume if available
    total_volume = None
    clusters_with_volume = 0
    for insights in cluster_insights.values():
        if 'total_volume' in insights and insights['total_volume'] is not None:
            if total_volume is None:
                total_volume = 0
            total_volume += insights['total_volume']
            clusters_with_volume += 1
    
    # Calculate intent distribution
    intent_counts = {}
    for insights in cluster_insights.values():
        intent = insights.get('primary_intent', 'Unknown')
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Keywords",
            f"{total_keywords:,}",
            help="Total number of keywords across all clusters"
        )
    
    with col2:
        st.metric(
            "Total Clusters",
            f"{total_clusters}",
            help="Number of semantic clusters found"
        )
    
    with col3:
        if total_volume is not None:
            st.metric(
                "Total Search Volume",
                f"{total_volume:,}",
                help="Combined search volume across all keywords"
            )
        else:
            st.metric(
                "Avg. Cluster Size",
                f"{int(total_keywords / max(1, total_clusters)):,}",
                help="Average number of keywords per cluster"
            )

def show_keyword_table(df: pd.DataFrame, cluster_id: Optional[int] = None) -> None:
    """
    Display a table of keywords with optional filtering by cluster.
    
    Args:
        df: Dataframe with all keyword data
        cluster_id: Optional cluster ID to filter by
    """
    # Copy the dataframe to avoid modifying the original
    display_df = df.copy()
    
    # Filter by cluster if specified
    if cluster_id is not None:
        display_df = display_df[display_df['cluster_id'] == cluster_id]
    
    # Select columns to display
    columns_to_show = ['keyword', 'cluster_id', 'cluster_name']
    
    # Add search volume if available
    if 'search_volume' in display_df.columns:
        columns_to_show.append('search_volume')
    
    # Add other metrics if available
    for col in ['coherence_score', 'primary_intent']:
        if col in display_df.columns:
            columns_to_show.append(col)
    
    # Keep only selected columns that exist
    display_df = display_df[[col for col in columns_to_show if col in display_df.columns]]
    
    # Display the table
    st.dataframe(
        display_df,
        use_container_width=True,
        height=min(400, 35 * len(display_df) + 38)  # Dynamic height based on row count
    )

def show_cluster_detail_view(cluster_id: int, 
                           df: pd.DataFrame, 
                           insights: Dict[str, Any]) -> None:
    """
    Display detailed information about a single cluster.
    
    Args:
        cluster_id: ID of the cluster to display
        df: Dataframe with all keyword data
        insights: Dictionary containing cluster insights
    """
    # Get cluster data
    cluster_df = df[df['cluster_id'] == cluster_id]
    cluster_name = insights.get('cluster_name', f"Cluster {cluster_id}")
    
    # Create header
    st.markdown(f"## {cluster_name}")
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Keywords", 
            f"{len(cluster_df)}",
            help="Number of keywords in this cluster"
        )
    
    with col2:
        intent = insights.get('primary_intent', 'Unknown')
        st.metric(
            "Primary Intent", 
            intent,
            help="Dominant search intent for this cluster"
        )
    
    with col3:
        if 'quality_score' in insights:
            quality = insights['quality_score']
            quality_color = "#4CAF50" if quality >= 7 else "#FF9800" if quality >= 5 else "#F44336"
            st.metric(
                "Coherence Score", 
                f"{quality:.1f}/10",
                help="Semantic coherence of the cluster (higher is better)",
                delta_color=quality_color
            )
        elif 'total_volume' in insights:
            st.metric(
                "Search Volume", 
                f"{insights['total_volume']:,}",
                help="Total monthly search volume"
            )
    
    # Journey and intent breakdown
    st.subheader("Search Intent Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Show intent scores if available
        intent_scores = insights.get('intent_scores', {})
        if intent_scores:
            # Prepare data for visualization
            intents = list(intent_scores.keys())
            scores = list(intent_scores.values())
            colors = [INTENT_COLORS.get(intent, "#757575") for intent in intents]
            
            # Create bar chart
            fig = px.bar(
                x=intents, 
                y=scores,
                labels={'x': 'Intent', 'y': 'Score'},
                color=intents,
                color_discrete_map={intent: color for intent, color in zip(intents, colors)},
                title="Intent Distribution",
                height=300
            )
            
            fig.update_layout(
                margin=dict(l=20, r=20, t=30, b=40),
                xaxis_title=None
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Show journey phase
        journey_phase = insights.get('journey_phase', 'Unknown')
        
        # Create a visual representation of customer journey
        phases = ["Research Phase (Early)", 
                 "Research-to-Consideration Transition", 
                 "Consideration Phase (Middle)",
                 "Consideration-to-Decision Transition",
                 "Decision Phase (Late)"]
        
        # Simplify phase names for display
        display_phases = ["Research", "Research → Consider", "Consideration", "Consider → Decision", "Decision"]
        
        # Determine active phase
        active_idx = -1
        for i, phase in enumerate(phases):
            if journey_phase.startswith(phase.split(" ")[0]):
                active_idx = i
                break
        
        # Create journey visualization
        st.markdown("##### Customer Journey Phase")
        
        # Create a simple visual representation
        journey_html = '<div style="display: flex; width: 100%; margin-top: 10px;">'
        
        for i, phase in enumerate(display_phases):
            # Determine styling based on active phase
            if i == active_idx:
                bg_color = "#4CAF50"
                text_color = "white"
                weight = "bold"
            else:
                bg_color = "#f1f1f1"
                text_color = "#666"
                weight = "normal"
            
            journey_html += f'''
            <div style="flex: 1; text-align: center; padding: 8px 2px; background-color: {bg_color}; 
                        color: {text_color}; font-weight: {weight}; font-size: 12px; 
                        border-radius: 4px; margin: 0 2px;">
                {phase}
            </div>
            '''
        
        journey_html += '</div>'
        
        st.markdown(journey_html, unsafe_allow_html=True)
        
        # Show content recommendations
        st.markdown("##### Recommended Content Types")
        content_formats = insights.get('suggested_content_formats', [])
        if content_formats:
            format_html = '<div style="margin-top: 10px;">'
            for format_type in content_formats[:4]:  # Show top 4
                format_html += f'<div style="background-color: #f1f1f1; padding: 8px; margin-bottom: 5px; border-radius: 4px;">{format_type}</div>'
            format_html += '</div>'
            st.markdown(format_html, unsafe_allow_html=True)
    
    # Show keyword list
    st.subheader("Keywords in this Cluster")
    show_keyword_table(df, cluster_id)
    
    # Add export option
    st.download_button(
        "Export Cluster Data (CSV)",
        cluster_df.to_csv(index=False),
        file_name=f"cluster_{cluster_id}_{cluster_name.replace(' ', '_')}.csv",
        mime="text/csv",
        use_container_width=True
    )

def show_export_options(df: pd.DataFrame, cluster_insights: Dict[int, Dict[str, Any]]) -> None:
    """
    Display export options for the clustering results.
    
    Args:
        df: Dataframe with all keyword data
        cluster_insights: Dictionary of cluster insights
    """
    st.subheader("Export Options")
    
    # Create columns for export options
    col1, col2 = st.columns(2)
    
    with col1:
        format_options = ["CSV", "Excel"]
        export_format = st.radio(
            "Export Format:",
            options=format_options,
            horizontal=True
        )
    
    with col2:
        include_options = st.multiselect(
            "Include in Export:",
            options=["Cluster Names", "Intent Analysis", "Quality Scores", "Journey Phases"],
            default=["Cluster Names", "Intent Analysis"]
        )
    
    # Prepare export dataframe
    export_df = df.copy()
    
    # Add selected columns based on user choices
    if "Cluster Names" in include_options and 'cluster_name' not in export_df.columns:
        export_df['cluster_name'] = export_df['cluster_id'].apply(
            lambda cid: cluster_insights.get(cid, {}).get('cluster_name', f"Cluster {cid}")
        )
    
    if "Intent Analysis" in include_options and 'primary_intent' not in export_df.columns:
        export_df['primary_intent'] = export_df['cluster_id'].apply(
            lambda cid: cluster_insights.get(cid, {}).get('primary_intent', "Unknown")
        )
    
    if "Quality Scores" in include_options and 'quality_score' not in export_df.columns:
        export_df['quality_score'] = export_df['cluster_id'].apply(
            lambda cid: cluster_insights.get(cid, {}).get('quality_score', None)
        )
    
    if "Journey Phases" in include_options and 'journey_phase' not in export_df.columns:
        export_df['journey_phase'] = export_df['cluster_id'].apply(
            lambda cid: cluster_insights.get(cid, {}).get('journey_phase', "Unknown")
        )
    
    # Create export button
    if export_format == "CSV":
        st.download_button(
            "Download Clustered Keywords",
            export_df.to_csv(index=False),
            file_name="keyword_clusters.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:  # Excel
        try:
            from io import BytesIO
            
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                export_df.to_excel(writer, sheet_name='All Keywords', index=False)
                
                # Create a summary sheet
                if cluster_insights:
                    summary_data = []
                    for cluster_id, insights in cluster_insights.items():
                        row = {
                            'Cluster ID': cluster_id,
                            'Cluster Name': insights.get('cluster_name', f"Cluster {cluster_id}"),
                            'Keywords': insights.get('keyword_count', 0),
                            'Primary Intent': insights.get('primary_intent', 'Unknown')
                        }
                        
                        if "Quality Scores" in include_options:
                            row['Quality Score'] = insights.get('quality_score', None)
                        
                        if "Journey Phases" in include_options:
                            row['Journey Phase'] = insights.get('journey_phase', 'Unknown')
                        
                        if 'total_volume' in insights:
                            row['Search Volume'] = insights.get('total_volume', 0)
                        
                        summary_data.append(row)
                    
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Cluster Summary', index=False)
                
                writer.save()
            
            st.download_button(
                "Download Excel Report",
                output.getvalue(),
                file_name="keyword_clusters.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error creating Excel file: {str(e)}")
            st.info("Please try CSV format instead, or install additional dependencies with 'pip install xlsxwriter openpyxl'")

def show_opportunity_clusters(opportunities: List[int], 
                            cluster_insights: Dict[int, Dict[str, Any]],
                            df: pd.DataFrame,
                            on_click_callback: Optional[Callable] = None) -> None:
    """
    Display a special section highlighting opportunity clusters.
    
    Args:
        opportunities: List of cluster IDs identified as opportunities
        cluster_insights: Dictionary of cluster insights
        df: Dataframe with all keyword data
        on_click_callback: Optional callback function for when a cluster is clicked
    """
    if not opportunities:
        return
    
    st.markdown("### 🌟 Top Opportunities")
    st.markdown("These clusters represent your best opportunities based on search volume, competition, and relevance.")
    
    # Display opportunity clusters in a 2-column grid
    cols = st.columns(2)
    for i, cluster_id in enumerate(opportunities):
        with cols[i % 2]:
            if cluster_id in cluster_insights:
                display_cluster_card(
                    cluster_id, 
                    cluster_insights[cluster_id], 
                    df,
                    on_click_callback
                )
