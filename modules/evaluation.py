# modules/evaluation.py
"""
Cluster evaluation and quality assessment module.

This module provides functions for evaluating the quality of clusters,
analyzing insights, and generating recommendations based on clusters.
"""

import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Dict, Optional, Any, Union, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

def evaluate_cluster_quality(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """
    Evaluate the quality of clusters based on semantic coherence.
    
    Args:
        df: Dataframe with 'cluster_id' column
        embeddings: Numpy array of keyword embeddings
        
    Returns:
        Dataframe with added quality metrics
    """
    if 'cluster_id' not in df.columns:
        st.error("Dataframe must have a 'cluster_id' column")
        return df
    
    result_df = df.copy()
    
    # Initialize quality metrics
    result_df['coherence_score'] = 0.0
    result_df['centroid_distance'] = 0.0
    
    # Get unique cluster IDs
    cluster_ids = result_df['cluster_id'].unique()
    
    # Set up progress tracking
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    # For each cluster, calculate quality metrics
    for i, cluster_id in enumerate(cluster_ids):
        progress_text.text(f"Evaluating cluster {i+1}/{len(cluster_ids)}")
        
        # Get indices for this cluster
        cluster_indices = result_df[result_df['cluster_id'] == cluster_id].index.tolist()
        
        if len(cluster_indices) <= 1:
            # If only one keyword in cluster, coherence is perfect and distance is zero
            result_df.loc[cluster_indices, 'coherence_score'] = 1.0
            result_df.loc[cluster_indices, 'centroid_distance'] = 0.0
            continue
        
        # Get embeddings for this cluster
        cluster_embeddings = embeddings[cluster_indices]
        
        # Calculate centroid
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # Normalize for cosine similarity
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid_normalized = centroid / centroid_norm
        else:
            centroid_normalized = centroid
        
        # Calculate pairwise similarities within cluster
        similarities = []
        distances = []
        
        for idx, embedding in zip(cluster_indices, cluster_embeddings):
            # Normalize embedding
            emb_norm = np.linalg.norm(embedding)
            if emb_norm > 0:
                embedding_normalized = embedding / emb_norm
            else:
                embedding_normalized = embedding
            
            # Calculate cosine similarity with centroid
            similarity = np.dot(embedding_normalized, centroid_normalized)
            similarities.append(similarity)
            
            # Calculate Euclidean distance to centroid
            distance = np.linalg.norm(embedding - centroid)
            distances.append(distance)
            
            # Store individual scores
            result_df.loc[idx, 'coherence_score'] = similarity
            result_df.loc[idx, 'centroid_distance'] = distance
        
        # Update progress
        progress_bar.progress((i + 1) / len(cluster_ids))
    
    # Calculate cluster-level metrics
    cluster_metrics = result_df.groupby('cluster_id').agg({
        'coherence_score': 'mean',
        'centroid_distance': 'mean'
    }).reset_index()
    
    # Add a quality score based on coherence (higher is better)
    cluster_metrics['quality_score'] = cluster_metrics['coherence_score'] * 10  # Scale to 0-10
    
    # Merge back with original dataframe
    result_df = pd.merge(
        result_df,
        cluster_metrics[['cluster_id', 'quality_score']],
        on='cluster_id',
        how='left'
    )
    
    # Clear progress indicators
    progress_text.empty()
    progress_bar.empty()
    
    st.success(f"✅ Evaluated quality for {len(cluster_ids)} clusters")
    return result_df

def identify_outliers(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Identify potential outliers in clusters based on distance from centroid.
    
    Args:
        df: Dataframe with 'centroid_distance' column
        threshold: Z-score threshold for outlier detection
        
    Returns:
        Dataframe with 'is_outlier' column added
    """
    if 'centroid_distance' not in df.columns or 'cluster_id' not in df.columns:
        return df
    
    result_df = df.copy()
    result_df['is_outlier'] = False
    
    # For each cluster, identify outliers
    for cluster_id in result_df['cluster_id'].unique():
        # Get distances for this cluster
        cluster_mask = result_df['cluster_id'] == cluster_id
        distances = result_df.loc[cluster_mask, 'centroid_distance'].values
        
        if len(distances) <= 2:
            continue  # Skip small clusters
        
        # Calculate z-scores
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        if std_dist > 0:
            z_scores = (distances - mean_dist) / std_dist
            
            # Mark outliers
            outlier_indices = result_df.loc[cluster_mask].index[z_scores > threshold]
            result_df.loc[outlier_indices, 'is_outlier'] = True
    
    return result_df

def analyze_cluster_insights(df: pd.DataFrame, 
                            cluster_intents: Dict[int, Dict[str, Any]],
                            cluster_names: Dict[int, str]) -> Dict[int, Dict[str, Any]]:
    """
    Generate comprehensive insights for each cluster.
    
    Args:
        df: Dataframe with clustering results
        cluster_intents: Dictionary mapping cluster IDs to intent data
        cluster_names: Dictionary mapping cluster IDs to cluster names
        
    Returns:
        Dictionary of cluster insights
    """
    if 'cluster_id' not in df.columns:
        st.error("Dataframe must have a 'cluster_id' column")
        return {}
    
    # Prepare results dictionary
    insights = {}
    
    # For each cluster, generate insights
    for cluster_id in df['cluster_id'].unique():
        # Get cluster data
        cluster_df = df[df['cluster_id'] == cluster_id]
        
        # Basic metrics
        keyword_count = len(cluster_df)
        
        # Get representative keywords
        if 'is_representative' in cluster_df.columns:
            rep_keywords = cluster_df[cluster_df['is_representative']]['keyword'].tolist()
        else:
            rep_keywords = cluster_df['keyword'].tolist()[:min(10, keyword_count)]
        
        # Calculate search volume if available
        if 'search_volume' in cluster_df.columns:
            try:
                total_volume = cluster_df['search_volume'].sum()
                avg_volume = cluster_df['search_volume'].mean()
                min_volume = cluster_df['search_volume'].min()
                max_volume = cluster_df['search_volume'].max()
            except:
                total_volume = None
                avg_volume = None
                min_volume = None
                max_volume = None
        else:
            total_volume = None
            avg_volume = None
            min_volume = None
            max_volume = None
        
        # Get quality metrics if available
        if 'quality_score' in cluster_df.columns:
            quality_score = cluster_df['quality_score'].mean()
        else:
            quality_score = None
        
        # Get coherence score if available
        if 'coherence_score' in cluster_df.columns:
            coherence_score = cluster_df['coherence_score'].mean()
        else:
            coherence_score = None
        
        # Get search intent data
        intent_data = cluster_intents.get(cluster_id, {})
        primary_intent = intent_data.get('primary_intent', 'Unknown')
        intent_scores = intent_data.get('scores', {})
        
        # Determine customer journey phase
        from modules.search_intent import get_customer_journey_phase
        journey_phase = get_customer_journey_phase(intent_data)
        
        # Get suggested content formats
        from modules.search_intent import suggest_content_formats
        content_formats = suggest_content_formats(primary_intent)
        
        # Create cluster insight dictionary
        cluster_insight = {
            'cluster_id': cluster_id,
            'cluster_name': cluster_names.get(cluster_id, f"Cluster {cluster_id}"),
            'keyword_count': keyword_count,
            'representative_keywords': rep_keywords,
            'primary_intent': primary_intent,
            'intent_scores': intent_scores,
            'journey_phase': journey_phase,
            'suggested_content_formats': content_formats,
            'quality_score': quality_score,
            'coherence_score': coherence_score,
            'keyword_sample': cluster_df['keyword'].tolist()[:20]  # Sample of keywords
        }
        
        # Add search volume metrics if available
        if total_volume is not None:
            cluster_insight['total_volume'] = total_volume
            cluster_insight['avg_volume'] = avg_volume
            cluster_insight['min_volume'] = min_volume
            cluster_insight['max_volume'] = max_volume
        
        # Add to insights dictionary
        insights[cluster_id] = cluster_insight
    
    return insights

def generate_cluster_recommendations(cluster_insights: Dict[int, Dict[str, Any]]) -> Dict[int, List[str]]:
    """
    Generate actionable recommendations for each cluster.
    
    Args:
        cluster_insights: Dictionary of cluster insights
        
    Returns:
        Dictionary mapping cluster IDs to lists of recommendations
    """
    recommendations = {}
    
    for cluster_id, insight in cluster_insights.items():
        cluster_recs = []
        
        # Get key data
        intent = insight.get('primary_intent', 'Unknown')
        journey_phase = insight.get('journey_phase', 'Unknown')
        keyword_count = insight.get('keyword_count', 0)
        quality_score = insight.get('quality_score')
        
        # Intent-based recommendations
        if intent == "Informational":
            cluster_recs.append("Create educational content that answers questions and provides valuable information")
            cluster_recs.append("Target featured snippets with clear, concise answers to common questions")
        elif intent == "Navigational":
            cluster_recs.append("Ensure your website is easy to find and navigate")
            cluster_recs.append("Optimize for brand and product name searches")
        elif intent == "Transactional":
            cluster_recs.append("Create clear calls-to-action and streamlined purchase paths")
            cluster_recs.append("Highlight pricing, deals, and payment options")
        elif intent == "Commercial":
            cluster_recs.append("Provide detailed product comparisons and reviews")
            cluster_recs.append("Include pros and cons, features, and use cases")
        
        # Journey phase recommendations
        if "Research Phase" in journey_phase:
            cluster_recs.append("Focus on building awareness and establishing expertise")
        elif "Consideration Phase" in journey_phase:
            cluster_recs.append("Help users evaluate options and narrow down choices")
        elif "Decision Phase" in journey_phase:
            cluster_recs.append("Remove purchase barriers and build confidence to convert")
        
        # Quality-based recommendations
        if quality_score is not None:
            if quality_score < 6:
                cluster_recs.append("Consider splitting this cluster into more cohesive groups")
            elif quality_score > 8:
                cluster_recs.append("This is a highly cohesive topic - consider creating comprehensive content")
        
        # Size-based recommendations
        if keyword_count > 50:
            cluster_recs.append("This is a large topic - consider creating a pillar page with subtopics")
        elif keyword_count < 10:
            cluster_recs.append("This is a niche topic - consider bundling with related content")
        
        # Add to recommendations dictionary
        recommendations[cluster_id] = cluster_recs
    
    return recommendations

def calculate_difficulty_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a difficulty score for ranking for keywords in each cluster.
    
    Args:
        df: Dataframe with clustering results
        
    Returns:
        Dataframe with 'difficulty_score' column added
    """
    result_df = df.copy()
    
    # Initialize difficulty score
    result_df['difficulty_score'] = 5.0  # Default mid-range score
    
    # If we have competition or keyword difficulty data, use it
    if 'competition' in result_df.columns:
        result_df['difficulty_score'] = result_df['competition'] * 10
    elif 'keyword_difficulty' in result_df.columns:
        result_df['difficulty_score'] = result_df['keyword_difficulty']
    else:
        # Estimate difficulty based on other factors if available
        
        # 1. Word count (longer queries tend to be easier)
        result_df['word_count'] = result_df['keyword'].apply(lambda x: len(str(x).split()))
        
        # More words = easier (generally)
        result_df.loc[result_df['word_count'] >= 4, 'difficulty_score'] -= 1
        result_df.loc[result_df['word_count'] <= 1, 'difficulty_score'] += 1.5
        
        # 2. Search volume if available (higher volume = harder)
        if 'search_volume' in result_df.columns:
            # Normalize search volume to 0-5 scale for adjustment
            volume_max = result_df['search_volume'].max()
            if volume_max > 0:
                volume_factor = result_df['search_volume'] / volume_max * 5
                result_df['difficulty_score'] += volume_factor
            
        # 3. Intent-based adjustment
        if 'primary_intent' in result_df.columns:
            # Informational tends to be easier than transactional
            result_df.loc[result_df['primary_intent'] == 'Informational', 'difficulty_score'] -= 0.5
            result_df.loc[result_df['primary_intent'] == 'Transactional', 'difficulty_score'] += 1.0
        
        # Keep within 0-10 range
        result_df['difficulty_score'] = result_df['difficulty_score'].clip(0, 10)
    
    # Calculate average difficulty for each cluster
    cluster_difficulty = result_df.groupby('cluster_id')['difficulty_score'].mean().reset_index()
    cluster_difficulty.rename(columns={'difficulty_score': 'cluster_difficulty'}, inplace=True)
    
    # Merge back with original dataframe
    result_df = pd.merge(
        result_df,
        cluster_difficulty,
        on='cluster_id',
        how='left'
    )
    
    return result_df

def identify_opportunity_clusters(cluster_insights: Dict[int, Dict[str, Any]]) -> List[int]:
    """
    Identify clusters that represent the best opportunities based on multiple factors.
    
    Args:
        cluster_insights: Dictionary of cluster insights
        
    Returns:
        List of cluster IDs representing the best opportunities
    """
    # Criteria to consider:
    # 1. Higher search volume (if available)
    # 2. Lower difficulty (if available)
    # 3. Higher quality score (more coherent clusters)
    # 4. Right balance of keyword count (not too small, not too big)
    
    opportunities = []
    
    # Score each cluster
    cluster_scores = {}
    for cluster_id, insight in cluster_insights.items():
        score = 0
        
        # Search volume factor (higher is better)
        if 'total_volume' in insight:
            # We'll use a log scale to prevent very high volume clusters from dominating
            volume_score = np.log10(max(insight['total_volume'], 1)) / 5  # Scale factor
            score += volume_score
        
        # Quality factor (higher is better)
        if 'quality_score' in insight and insight['quality_score'] is not None:
            quality_score = insight['quality_score'] / 10  # 0-1 scale
            score += quality_score * 2  # Weight quality more heavily
        
        # Difficulty factor (lower is better)
        if 'difficulty_score' in insight and insight['difficulty_score'] is not None:
            difficulty_score = (10 - insight['difficulty_score']) / 10  # Invert and scale to 0-1
            score += difficulty_score * 1.5
        
        # Size factor (medium is best)
        keyword_count = insight.get('keyword_count', 0)
        if 10 <= keyword_count <= 50:
            score += 0.5  # Bonus for ideal size
        elif keyword_count < 5 or keyword_count > 100:
            score -= 0.2  # Penalty for too small or too large
        
        # Store the score
        cluster_scores[cluster_id] = score
    
    # Sort clusters by score and take top 5
    sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)
    opportunities = [cluster_id for cluster_id, _ in sorted_clusters[:5]]
    
    return opportunities

def analyze_keyword_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the distribution of keywords across various dimensions.
    
    Args:
        df: Dataframe with clustering results
        
    Returns:
        Dictionary of distribution metrics
    """
    distributions = {}
    
    # 1. Cluster size distribution
    cluster_sizes = df['cluster_id'].value_counts().to_dict()
    distributions['cluster_sizes'] = cluster_sizes
    
    # 2. Intent distribution (if available)
    if 'primary_intent' in df.columns:
        intent_counts = df['primary_intent'].value_counts().to_dict()
        distributions['intent_distribution'] = intent_counts
        
        # Also calculate percentage
        total = sum(intent_counts.values())
        intent_pct = {intent: (count / total) * 100 for intent, count in intent_counts.items()}
        distributions['intent_percentage'] = intent_pct
    
    # 3. Journey phase distribution (if available)
    if 'journey_phase' in df.columns:
        journey_counts = df['journey_phase'].value_counts().to_dict()
        distributions['journey_distribution'] = journey_counts
    
    # 4. Word count distribution
    df['word_count'] = df['keyword'].apply(lambda x: len(str(x).split()))
    word_count_dist = df['word_count'].value_counts().sort_index().to_dict()
    distributions['word_count_distribution'] = word_count_dist
    
    # 5. Search volume distribution (if available)
    if 'search_volume' in df.columns:
        # Create bins for volume ranges
        try:
            df['volume_range'] = pd.cut(
                df['search_volume'],
                bins=[0, 10, 100, 1000, 10000, float('inf')],
                labels=['1-10', '11-100', '101-1000', '1001-10000', '10000+']
            )
            volume_dist = df['volume_range'].value_counts().to_dict()
            distributions['volume_distribution'] = volume_dist
        except:
            pass
    
    return distributions
