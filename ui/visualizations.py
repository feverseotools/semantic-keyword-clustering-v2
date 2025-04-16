# ui/visualizations.py
"""
Visualization components for the keyword clustering application.

This module provides functions for creating and displaying visualizations
of keyword clustering results, including charts, graphs, and other visual elements.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

# Import constants
from utils.constants import INTENT_COLORS

def show_intent_distribution(cluster_insights: Dict[int, Dict[str, Any]]) -> None:
    """
    Display a pie chart showing the distribution of search intents.
    
    Args:
        cluster_insights: Dictionary of cluster insights
    """
    st.markdown("### Search Intent Distribution")
    
    # Count clusters by primary intent
    intent_counts = {}
    
    for cluster_id, insights in cluster_insights.items():
        intent = insights.get('primary_intent', 'Unknown')
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    if not intent_counts:
        st.info("No intent data available.")
        return
    
    # Create pie chart
    labels = list(intent_counts.keys())
    values = list(intent_counts.values())
    colors = [INTENT_COLORS.get(intent, "#757575") for intent in labels]
    
    fig = px.pie(
        names=labels,
        values=values,
        color=labels,
        color_discrete_map={intent: color for intent, color in zip(labels, colors)},
        hole=0.4
    )
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    # Add percentage labels
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        insidetextfont=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_cluster_sizes(cluster_insights: Dict[int, Dict[str, Any]], 
                      top_n: int = 15) -> None:
    """
    Display a bar chart showing the sizes of clusters.
    
    Args:
        cluster_insights: Dictionary of cluster insights
        top_n: Number of top clusters to display
    """
    st.markdown("### Cluster Sizes")
    
    # Extract cluster sizes
    cluster_data = []
    
    for cluster_id, insights in cluster_insights.items():
        name = insights.get('cluster_name', f"Cluster {cluster_id}")
        count = insights.get('keyword_count', 0)
        intent = insights.get('primary_intent', 'Unknown')
        cluster_data.append({
            'cluster_id': cluster_id,
            'name': name,
            'count': count,
            'intent': intent
        })
    
    if not cluster_data:
        st.info("No cluster data available.")
        return
    
    # Sort by size and take top N
    sorted_clusters = sorted(cluster_data, key=lambda x: x['count'], reverse=True)
    top_clusters = sorted_clusters[:min(top_n, len(sorted_clusters))]
    
    # Create dataframe for plotting
    df = pd.DataFrame(top_clusters)
    
    # Truncate long names
    df['display_name'] = df['name'].apply(lambda x: (x[:25] + '...') if len(x) > 25 else x)
    
    # Create bar chart
    fig = px.bar(
        df,
        x='count',
        y='display_name',
        color='intent',
        color_discrete_map=INTENT_COLORS,
        orientation='h',
        labels={
            'count': 'Number of Keywords',
            'display_name': 'Cluster',
            'intent': 'Search Intent'
        }
    )
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis={'categoryorder': 'total ascending'},
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        height=min(100 + 30 * len(top_clusters), 600)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_search_volume_by_intent(cluster_insights: Dict[int, Dict[str, Any]]) -> None:
    """
    Display a bar chart showing search volume by intent.
    
    Args:
        cluster_insights: Dictionary of cluster insights
    """
    st.markdown("### Search Volume by Intent")
    
    # Calculate volume by intent
    intent_volumes = {}
    
    for cluster_id, insights in cluster_insights.items():
        if 'total_volume' in insights and insights['total_volume'] is not None:
            intent = insights.get('primary_intent', 'Unknown')
            volume = insights.get('total_volume', 0)
            intent_volumes[intent] = intent_volumes.get(intent, 0) + volume
    
    if not intent_volumes:
        st.info("No search volume data available.")
        return
    
    # Create dataframe for plotting
    df = pd.DataFrame({
        'intent': list(intent_volumes.keys()),
        'volume': list(intent_volumes.values())
    })
    
    # Sort by volume
    df.sort_values('volume', ascending=False, inplace=True)
    
    # Create bar chart
    fig = px.bar(
        df,
        x='intent',
        y='volume',
        color='intent',
        color_discrete_map=INTENT_COLORS,
        labels={
            'intent': 'Search Intent',
            'volume': 'Monthly Search Volume'
        }
    )
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title=None
    )
    
    # Format y-axis with commas for thousands
    fig.update_layout(yaxis=dict(tickformat=","))
    
    st.plotly_chart(fig, use_container_width=True)

def show_journey_distribution(cluster_insights: Dict[int, Dict[str, Any]]) -> None:
    """
    Display a visualization of the customer journey distribution.
    
    Args:
        cluster_insights: Dictionary of cluster insights
    """
    st.markdown("### Customer Journey Distribution")
    
    # Count clusters by journey phase
    journey_counts = {}
    
    for cluster_id, insights in cluster_insights.items():
        journey = insights.get('journey_phase', 'Unknown')
        journey_counts[journey] = journey_counts.get(journey, 0) + 1
    
    if not journey_counts:
        st.info("No customer journey data available.")
        return
    
    # Define phase order for consistent display
    phase_order = [
        "Research Phase (Early)",
        "Research-to-Consideration Transition",
        "Consideration Phase (Middle)",
        "Consideration-to-Decision Transition",
        "Decision Phase (Late)",
        "Mixed Journey Stages",
        "Unknown"
    ]
    
    # Colors for journey phases
    journey_colors = {
        "Research Phase (Early)": "#2196F3",  # Blue
        "Research-to-Consideration Transition": "#26A69A",  # Teal
        "Consideration Phase (Middle)": "#4CAF50",  # Green
        "Consideration-to-Decision Transition": "#9C27B0",  # Purple
        "Decision Phase (Late)": "#FF9800",  # Orange
        "Mixed Journey Stages": "#757575",  # Gray
        "Unknown": "#BDBDBD"  # Light Gray
    }
    
    # Create dataframe with ordered phases
    df_journey = pd.DataFrame({
        'phase': list(journey_counts.keys()),
        'count': list(journey_counts.values())
    })
    
    # Add order column for sorting
    df_journey['order'] = df_journey['phase'].apply(
        lambda x: phase_order.index(x) if x in phase_order else len(phase_order)
    )
    
    # Sort by the predefined order
    df_journey.sort_values('order', inplace=True)
    
    # Create bar chart
    fig = px.bar(
        df_journey,
        x='phase',
        y='count',
        color='phase',
        color_discrete_map=journey_colors,
        labels={
            'phase': 'Customer Journey Phase',
            'count': 'Number of Clusters'
        }
    )
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title=None,
        xaxis={'categoryorder': 'array', 'categoryarray': list(df_journey['phase'])}
    )
    
    # Add a funnel shape to indicate progression
    funnel_shape = [
        {"type": "line", "x0": -0.5, "y0": 0, "x1": len(df_journey)-0.5, "y1": 0, 
         "line": {"color": "rgba(0,0,0,0.1)", "width": 70}},
        {"type": "line", "x0": -0.5, "y0": 0, "x1": len(df_journey)-0.5, "y1": 0, 
         "line": {"color": "rgba(0,0,0,0.05)", "width": 120}}
    ]
    
    fig.update_layout(shapes=funnel_shape)
    
    st.plotly_chart(fig, use_container_width=True)

def show_cluster_coherence(cluster_insights: Dict[int, Dict[str, Any]], 
                         top_n: int = 15) -> None:
    """
    Display a visualization of cluster coherence scores.
    
    Args:
        cluster_insights: Dictionary of cluster insights
        top_n: Number of clusters to display
    """
    st.markdown("### Cluster Coherence")
    
    # Extract coherence scores
    coherence_data = []
    
    for cluster_id, insights in cluster_insights.items():
        if 'quality_score' in insights and insights['quality_score'] is not None:
            name = insights.get('cluster_name', f"Cluster {cluster_id}")
            score = insights.get('quality_score', 0)
            count = insights.get('keyword_count', 0)
            coherence_data.append({
                'cluster_id': cluster_id,
                'name': name,
                'quality_score': score,
                'keyword_count': count
            })
    
    if not coherence_data:
        st.info("No coherence data available.")
        return
    
    # Sort by coherence and take top N
    sorted_coherence = sorted(coherence_data, key=lambda x: x['quality_score'], reverse=True)
    top_clusters = sorted_coherence[:min(top_n, len(sorted_coherence))]
    
    # Create dataframe for plotting
    df = pd.DataFrame(top_clusters)
    
    # Truncate long names
    df['display_name'] = df['name'].apply(lambda x: (x[:25] + '...') if len(x) > 25 else x)
    
    # Create color scale
    color_scale = [
        [0, "#F44336"],    # Red for low scores
        [0.5, "#FF9800"],  # Orange for medium scores
        [1, "#4CAF50"]     # Green for high scores
    ]
    
    # Create bar chart
    fig = px.bar(
        df,
        x='quality_score',
        y='display_name',
        color='quality_score',
        color_continuous_scale=color_scale,
        orientation='h',
        labels={
            'quality_score': 'Coherence Score (0-10)',
            'display_name': 'Cluster'
        }
    )
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis={'categoryorder': 'total ascending'},
        height=min(100 + 30 * len(top_clusters), 600)
    )
    
    # Set x-axis range to 0-10
    fig.update_xaxes(range=[0, 10])
    
    st.plotly_chart(fig, use_container_width=True)

def show_word_count_distribution(df: pd.DataFrame) -> None:
    """
    Display a histogram of keyword word counts.
    
    Args:
        df: Dataframe with 'keyword' column
    """
    st.markdown("### Keyword Length Distribution")
    
    if 'keyword' not in df.columns:
        st.info("No keyword data available.")
        return
    
    # Calculate word counts
    df_wc = df.copy()
    df_wc['word_count'] = df_wc['keyword'].apply(lambda x: len(str(x).split()))
    
    # Create histogram
    fig = px.histogram(
        df_wc,
        x='word_count',
        color_discrete_sequence=["#2196F3"],
        labels={
            'word_count': 'Number of Words',
            'count': 'Number of Keywords'
        },
        nbins=20
    )
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        bargap=0.1
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_network_graph(cluster_insights: Dict[int, Dict[str, Any]],
                      max_clusters: int = 10,
                      max_keywords_per_cluster: int = 5) -> None:
    """
    Display a network graph of clusters and their representative keywords.
    
    Args:
        cluster_insights: Dictionary of cluster insights
        max_clusters: Maximum number of clusters to display
        max_keywords_per_cluster: Maximum keywords per cluster to display
    """
    st.markdown("### Cluster Network Graph")
    
    try:
        import networkx as nx
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Sort clusters by size
        sorted_clusters = sorted(
            cluster_insights.items(),
            key=lambda x: x[1].get('keyword_count', 0),
            reverse=True
        )
        top_clusters = sorted_clusters[:max_clusters]
        
        # Add nodes and edges
        for cluster_id, insights in top_clusters:
            cluster_name = insights.get('cluster_name', f"Cluster {cluster_id}")
            intent = insights.get('primary_intent', 'Unknown')
            G.add_node(cluster_name, node_type='cluster', intent=intent)
            
            # Add representative keywords as nodes
            rep_keywords = insights.get('representative_keywords', [])[:max_keywords_per_cluster]
            for kw in rep_keywords:
                G.add_node(kw, node_type='keyword')
                G.add_edge(cluster_name, kw)
        
        # Create positions
        pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=50)
        
        # Create node lists and attributes
        nodes = list(G.nodes())
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in nodes:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            if G.nodes[node].get('node_type') == 'cluster':
                node_size.append(20)
                intent = G.nodes[node].get('intent', 'Unknown')
                color = INTENT_COLORS.get(intent, "#757575")
                node_color.append(color)
            else:
                node_size.append(10)
                node_color.append("#B0BEC5")  # Light gray for keywords
        
        # Create edges
        edge_x = []
        edge_y = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=1, color='#888')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0, l=0, r=0, t=0),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))
        
        st.plotly_chart(fig, use_container_width=True)
    
    except ImportError:
        st.info("Network visualization requires the networkx library. Install with 'pip install networkx'.")

def show_heatmap(df: pd.DataFrame, cluster_insights: Dict[int, Dict[str, Any]]) -> None:
    """
    Display a heatmap showing relationships between intents and journey phases.
    
    Args:
        df: Dataframe with clustering results
        cluster_insights: Dictionary of cluster insights
    """
    st.markdown("### Intent vs. Journey Phase Heatmap")
    
    # Check if we have both intent and journey data
    has_intent = any('primary_intent' in insights for insights in cluster_insights.values())
    has_journey = any('journey_phase' in insights for insights in cluster_insights.values())
    
    if not (has_intent and has_journey):
        st.info("Intent or journey phase data is missing.")
        return
    
    # Create a matrix of intent vs journey phase
    intent_journey_matrix = {}
    
    for cluster_id, insights in cluster_insights.items():
        intent = insights.get('primary_intent', 'Unknown')
        journey = insights.get('journey_phase', 'Unknown')
        
        if intent not in intent_journey_matrix:
            intent_journey_matrix[intent] = {}
        
        intent_journey_matrix[intent][journey] = intent_journey_matrix[intent].get(journey, 0) + 1
    
    # Define phase order for consistent display
    phase_order = [
        "Research Phase (Early)",
        "Research-to-Consideration Transition",
        "Consideration Phase (Middle)",
        "Consideration-to-Decision Transition",
        "Decision Phase (Late)",
        "Mixed Journey Stages",
        "Unknown"
    ]
    
    # Define intent order
    intent_order = ["Informational", "Commercial", "Transactional", "Navigational", "Mixed Intent", "Unknown"]
    
    # Create matrix data
    matrix_data = []
    intents = set()
    phases = set()
    
    for intent, journey_counts in intent_journey_matrix.items():
        intents.add(intent)
        for journey, count in journey_counts.items():
            phases.add(journey)
            matrix_data.append({
                'intent': intent,
                'journey_phase': journey,
                'count': count
            })
    
    # Create dataframe for heatmap
    df_matrix = pd.DataFrame(matrix_data)
    
    # Create pivot table
    intents_present = [i for i in intent_order if i in intents]
    phases_present = [p for p in phase_order if p in phases]
    
    # Create heatmap
    fig = px.density_heatmap(
        df_matrix,
        x='journey_phase',
        y='intent',
        z='count',
        category_orders={
            'journey_phase': phases_present,
            'intent': intents_present
        },
        color_continuous_scale=px.colors.sequential.Blues
    )
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Customer Journey Phase",
        yaxis_title="Search Intent"
    )
    
    st.plotly_chart(fig, use_container_width=True)
