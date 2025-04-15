# modules/clustering.py
"""
Clustering module for semantic keyword grouping.

This module provides functions to cluster keywords based on their semantic embeddings,
identify representative keywords for each cluster, and refine clusters for better coherence.
"""

import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Dict, Optional, Any, Union, Tuple
from sklearn.metrics.pairwise import cosine_similarity

# Try to import optional clustering methods
try:
    import hdbscan
    hdbscan_available = True
except ImportError:
    hdbscan_available = False

@st.cache_data
def run_clustering(df: pd.DataFrame, embeddings: np.ndarray, num_clusters: int = 10) -> pd.DataFrame:
    """
    Perform clustering on keyword embeddings.
    
    Args:
        df: Dataframe containing keywords
        embeddings: Numpy array of keyword embeddings
        num_clusters: Number of clusters to create
        
    Returns:
        Dataframe with added 'cluster_id' column
    """
    from sklearn.cluster import KMeans
    
    st.info(f"Clustering keywords into {num_clusters} groups...")
    
    # Validate inputs
    if len(df) != embeddings.shape[0]:
        st.error(f"Number of rows in dataframe ({len(df)}) doesn't match number of embeddings ({embeddings.shape[0]})")
        return df.copy()
    
    try:
        # Apply KMeans clustering
        kmeans = KMeans(
            n_clusters=num_clusters,
            random_state=42,
            n_init=10
        )
        
        # Fit and predict
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Copy the dataframe to avoid modifying the original
        result_df = df.copy()
        
        # Add cluster IDs to dataframe (adding 1 so clusters start at 1 instead of 0)
        result_df['cluster_id'] = cluster_labels + 1
        
        # Add cluster sizes for reference
        cluster_sizes = result_df['cluster_id'].value_counts().to_dict()
        result_df['cluster_size'] = result_df['cluster_id'].map(cluster_sizes)
        
        st.success(f"✅ Successfully created {num_clusters} clusters")
        return result_df
    
    except Exception as e:
        st.error(f"Error during clustering: {str(e)}")
        
        # Return original dataframe with a dummy cluster_id column as fallback
        result_df = df.copy()
        result_df['cluster_id'] = 1  # All in one cluster as fallback
        return result_df

def try_dbscan_clustering(df: pd.DataFrame, embeddings: np.ndarray, eps: float = 0.5, 
                         min_samples: int = 5) -> pd.DataFrame:
    """
    Attempt DBSCAN clustering, which automatically determines the number of clusters.
    
    Args:
        df: Dataframe containing keywords
        embeddings: Numpy array of keyword embeddings
        eps: Maximum distance between samples in a cluster
        min_samples: Minimum number of samples in a cluster
        
    Returns:
        Dataframe with added 'cluster_id' column
    """
    from sklearn.cluster import DBSCAN
    
    st.info("Attempting DBSCAN clustering (automatically determines number of clusters)...")
    
    try:
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(embeddings)
        
        # Check if clustering was successful
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        if n_clusters < 2:
            st.warning(f"DBSCAN found only {n_clusters} clusters. Consider using KMeans instead.")
            return None
        
        # Copy the dataframe
        result_df = df.copy()
        
        # Handle noise points (assigned to cluster -1)
        # For visualization purposes, we'll assign them to a separate "outliers" cluster
        cluster_labels = [label if label >= 0 else max(cluster_labels) + 1 for label in cluster_labels]
        
        # Add cluster IDs to dataframe (adding 1 so clusters start at 1)
        result_df['cluster_id'] = [label + 1 for label in cluster_labels]
        
        # Add cluster sizes for reference
        cluster_sizes = result_df['cluster_id'].value_counts().to_dict()
        result_df['cluster_size'] = result_df['cluster_id'].map(cluster_sizes)
        
        st.success(f"✅ DBSCAN created {n_clusters} clusters and identified {n_noise} noise points")
        return result_df
    
    except Exception as e:
        st.warning(f"DBSCAN clustering failed: {str(e)}")
        return None

def try_hdbscan_clustering(df: pd.DataFrame, embeddings: np.ndarray, 
                          min_cluster_size: int = 10) -> Optional[pd.DataFrame]:
    """
    Attempt HDBSCAN clustering if available, which can find clusters of varying density.
    
    Args:
        df: Dataframe containing keywords
        embeddings: Numpy array of keyword embeddings
        min_cluster_size: Minimum number of samples in a cluster
        
    Returns:
        Dataframe with added 'cluster_id' column or None if HDBSCAN failed
    """
    if not hdbscan_available:
        return None
    
    st.info("Attempting HDBSCAN clustering (good for finding natural clusters)...")
    
    try:
        # Apply HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=5,
            metric='euclidean',
            prediction_data=True
        )
        cluster_labels = clusterer.fit_predict(embeddings)
        
        # Check if clustering was successful
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        if n_clusters < 2:
            st.warning(f"HDBSCAN found only {n_clusters} clusters. Consider using KMeans instead.")
            return None
        
        # Copy the dataframe
        result_df = df.copy()
        
        # Handle noise points (assigned to cluster -1)
        # For visualization purposes, we'll assign them to a separate "outliers" cluster
        noise_cluster_id = max(cluster_labels) + 2 if n_noise > 0 else 0
        cluster_labels = [label + 1 if label >= 0 else noise_cluster_id for label in cluster_labels]
        
        # Add cluster IDs to dataframe
        result_df['cluster_id'] = cluster_labels
        
        # Add HDBSCAN probabilities if available
        if hasattr(clusterer, 'probabilities_'):
            result_df['cluster_probability'] = clusterer.probabilities_
        
        # Add cluster sizes for reference
        cluster_sizes = result_df['cluster_id'].value_counts().to_dict()
        result_df['cluster_size'] = result_df['cluster_id'].map(cluster_sizes)
        
        st.success(f"✅ HDBSCAN created {n_clusters} clusters and identified {n_noise} noise points")
        return result_df
    
    except Exception as e:
        st.warning(f"HDBSCAN clustering failed: {str(e)}")
        return None

def find_representative_keywords(df: pd.DataFrame, embeddings: np.ndarray, 
                                max_representatives: int = 10) -> pd.DataFrame:
    """
    Find representative keywords for each cluster.
    
    Args:
        df: Dataframe with 'cluster_id' column
        embeddings: Numpy array of keyword embeddings
        max_representatives: Maximum number of representative keywords per cluster
        
    Returns:
        Dataframe with 'is_representative' column added
    """
    if 'cluster_id' not in df.columns:
        st.error("Dataframe does not have a 'cluster_id' column")
        return df
    
    result_df = df.copy()
    result_df['is_representative'] = False
    
    # For each cluster, find the most representative keywords
    for cluster_id in result_df['cluster_id'].unique():
        # Get indices of keywords in this cluster
        cluster_indices = result_df[result_df['cluster_id'] == cluster_id].index.tolist()
        
        if len(cluster_indices) <= max_representatives:
            # If cluster is small, all keywords are representative
            result_df.loc[cluster_indices, 'is_representative'] = True
            continue
        
        # Get embeddings for this cluster
        cluster_embeddings = embeddings[cluster_indices]
        
        # Calculate cluster centroid
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # Calculate distance to centroid for each keyword
        distances = []
        for idx, embedding in zip(cluster_indices, cluster_embeddings):
            # Use cosine similarity (higher is better)
            similarity = cosine_similarity([embedding], [centroid])[0][0]
            distances.append((idx, similarity))
        
        # Sort by similarity (descending)
        distances.sort(key=lambda x: x[1], reverse=True)
        
        # Select top representatives
        representative_indices = [idx for idx, _ in distances[:max_representatives]]
        result_df.loc[representative_indices, 'is_representative'] = True
    
    return result_df

def refine_clusters(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """
    Refine clusters by handling outliers and small clusters.
    
    Args:
        df: Dataframe with 'cluster_id' column
        embeddings: Numpy array of keyword embeddings
        
    Returns:
        Dataframe with refined clusters
    """
    if 'cluster_id' not in df.columns:
        st.error("Dataframe does not have a 'cluster_id' column")
        return df
    
    result_df = df.copy()
    
    # Calculate cluster sizes
    cluster_sizes = result_df['cluster_id'].value_counts()
    
    # Find small clusters (less than 3 keywords)
    small_clusters = cluster_sizes[cluster_sizes < 3].index.tolist()
    
    # If there are small clusters, reassign their keywords
    if small_clusters:
        st.info(f"Refining {len(small_clusters)} small clusters with fewer than 3 keywords")
        
        # Get keywords in small clusters
        small_cluster_mask = result_df['cluster_id'].isin(small_clusters)
        small_cluster_indices = result_df[small_cluster_mask].index.tolist()
        
        # Get keywords in normal clusters
        normal_cluster_mask = ~result_df['cluster_id'].isin(small_clusters)
        normal_cluster_indices = result_df[normal_cluster_mask].index.tolist()
        
        if normal_cluster_indices:  # Make sure there are normal clusters
            # For each keyword in a small cluster
            for idx in small_cluster_indices:
                # Get its embedding
                keyword_embedding = embeddings[idx].reshape(1, -1)
                
                # Get embeddings for normal clusters
                normal_embeddings = embeddings[normal_cluster_indices]
                
                # Find most similar keyword in normal clusters
                similarities = cosine_similarity(keyword_embedding, normal_embeddings)[0]
                most_similar_idx = normal_cluster_indices[np.argmax(similarities)]
                
                # Assign to same cluster as most similar keyword
                result_df.loc[idx, 'cluster_id'] = result_df.loc[most_similar_idx, 'cluster_id']
    
    # Find representative keywords for each cluster
    result_df = find_representative_keywords(result_df, embeddings)
    
    return result_df

def get_cluster_keywords(df: pd.DataFrame, cluster_id: int, 
                         representative_only: bool = False) -> List[str]:
    """
    Get keywords belonging to a specific cluster.
    
    Args:
        df: Dataframe with 'cluster_id' and 'keyword' columns
        cluster_id: ID of the cluster to get keywords for
        representative_only: Whether to return only representative keywords
        
    Returns:
        List of keywords in the cluster
    """
    if 'cluster_id' not in df.columns or 'keyword' not in df.columns:
        return []
    
    # Filter by cluster ID
    cluster_df = df[df['cluster_id'] == cluster_id]
    
    # If representative_only and the column exists, filter further
    if representative_only and 'is_representative' in df.columns:
        cluster_df = cluster_df[cluster_df['is_representative'] == True]
    
    # Return the keywords
    return cluster_df['keyword'].tolist()

def calculate_cluster_coherence(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """
    Calculate coherence scores for each cluster based on embedding similarity.
    
    Args:
        df: Dataframe with 'cluster_id' column
        embeddings: Numpy array of keyword embeddings
        
    Returns:
        Dataframe with 'coherence_score' column added
    """
    if 'cluster_id' not in df.columns:
        st.error("Dataframe does not have a 'cluster_id' column")
        return df
    
    result_df = df.copy()
    
    # Initialize coherence scores
    result_df['coherence_score'] = 0.0
    
    # For each cluster, calculate coherence
    for cluster_id in result_df['cluster_id'].unique():
        # Get indices of keywords in this cluster
        cluster_indices = result_df[result_df['cluster_id'] == cluster_id].index.tolist()
        
        if len(cluster_indices) <= 1:
            # If only one keyword, coherence is perfect
            result_df.loc[cluster_indices, 'coherence_score'] = 1.0
            continue
        
        # Get embeddings for this cluster
        cluster_embeddings = embeddings[cluster_indices]
        
        # Calculate centroid
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # Normalize centroid
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm
        
        # Calculate cosine similarity of each keyword to centroid
        coherence_scores = []
        for embedding in cluster_embeddings:
            # Normalize embedding
            emb_norm = np.linalg.norm(embedding)
            if emb_norm > 0:
                embedding = embedding / emb_norm
            
            # Calculate similarity
            similarity = np.dot(embedding, centroid)
            coherence_scores.append(similarity)
        
        # Average coherence for the cluster
        avg_coherence = sum(coherence_scores) / len(coherence_scores)
        
        # Assign to all keywords in the cluster
        result_df.loc[cluster_indices, 'coherence_score'] = avg_coherence
    
    return result_df

def merge_similar_clusters(df: pd.DataFrame, embeddings: np.ndarray, 
                          similarity_threshold: float = 0.8) -> pd.DataFrame:
    """
    Merge clusters that are very similar to each other.
    
    Args:
        df: Dataframe with 'cluster_id' column
        embeddings: Numpy array of keyword embeddings
        similarity_threshold: Threshold for merging clusters (0-1)
        
    Returns:
        Dataframe with updated 'cluster_id' column
    """
    if 'cluster_id' not in df.columns:
        st.error("Dataframe does not have a 'cluster_id' column")
        return df
    
    result_df = df.copy()
    
    # Get unique cluster IDs
    cluster_ids = sorted(result_df['cluster_id'].unique())
    
    if len(cluster_ids) <= 1:
        return result_df
    
    # Calculate centroids for each cluster
    centroids = {}
    for cluster_id in cluster_ids:
        indices = result_df[result_df['cluster_id'] == cluster_id].index.tolist()
        cluster_embeddings = embeddings[indices]
        centroids[cluster_id] = np.mean(cluster_embeddings, axis=0)
    
    # Calculate similarities between centroids
    to_merge = {}
    for i, cluster1 in enumerate(cluster_ids):
        for cluster2 in cluster_ids[i+1:]:
            # Calculate cosine similarity
            similarity = cosine_similarity([centroids[cluster1]], [centroids[cluster2]])[0][0]
            
            if similarity >= similarity_threshold:
                # Clusters are similar, mark for merging
                if cluster1 not in to_merge:
                    to_merge[cluster1] = []
                to_merge[cluster1].append(cluster2)
    
    # If no clusters to merge, return original dataframe
    if not to_merge:
        return result_df
    
    # Perform merging
    st.info(f"Merging {sum(len(v) for v in to_merge.values())} similar clusters")
    
    # Create a mapping from old cluster IDs to new ones
    cluster_map = {cluster_id: cluster_id for cluster_id in cluster_ids}
    
    for cluster1, similar_clusters in to_merge.items():
        for cluster2 in similar_clusters:
            # Map all occurrences of cluster2 to cluster1
            cluster_map[cluster2] = cluster_map[cluster1]
    
    # Apply the mapping
    result_df['cluster_id'] = result_df['cluster_id'].map(cluster_map)
    
    # Renumber clusters to be sequential
    old_to_new = {old: i+1 for i, old in enumerate(sorted(set(cluster_map.values())))}
    result_df['cluster_id'] = result_df['cluster_id'].map(old_to_new)
    
    return result_df
