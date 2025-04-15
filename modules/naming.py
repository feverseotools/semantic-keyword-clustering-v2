# modules/naming.py
"""
Cluster naming module.

This module provides functions for generating descriptive names and summaries
for keyword clusters, using AI models when available or heuristic approaches
as a fallback.
"""

import re
import json
import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Dict, Optional, Any, Union, Tuple
from collections import Counter

# Try to import optional dependencies
try:
    from openai import OpenAI
    openai_available = True
except ImportError:
    openai_available = False

def extract_key_terms(keywords: List[str], top_n: int = 5) -> List[str]:
    """
    Extract the most common key terms from a list of keywords.
    
    Args:
        keywords: List of keywords to analyze
        top_n: Number of top terms to extract
        
    Returns:
        List of top key terms
    """
    # Tokenize keywords and count term frequencies
    all_words = []
    for keyword in keywords:
        words = str(keyword).lower().split()
        all_words.extend(words)
    
    # Filter out very common words and single characters
    stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'than', 'so', 'as',
                'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
                'did', 'can', 'could', 'will', 'would', 'should', 'may', 'might'}
    
    filtered_words = [word for word in all_words if word not in stopwords and len(word) > 1]
    
    # Count word frequencies
    word_counts = Counter(filtered_words)
    
    # Get top terms
    top_terms = [term for term, count in word_counts.most_common(top_n)]
    
    return top_terms

def generate_basic_name(keywords: List[str]) -> str:
    """
    Generate a basic name for a cluster using frequency analysis.
    
    Args:
        keywords: List of keywords in the cluster
        
    Returns:
        Generated cluster name
    """
    if not keywords:
        return "Empty Cluster"
    
    # Get top terms
    top_terms = extract_key_terms(keywords, top_n=3)
    
    if not top_terms:
        # If no good terms found, use the first keyword
        return keywords[0][:30] + "..."
    
    # Combine top terms into a name
    name = " ".join(top_terms).title()
    
    # Keep it reasonably short
    if len(name) > 50:
        name = name[:47] + "..."
    
    return name

@st.cache_data
def generate_openai_cluster_names(cluster_keywords: Dict[int, List[str]], 
                                api_key: str, 
                                model: str = "gpt-3.5-turbo") -> Dict[int, Tuple[str, str]]:
    """
    Generate cluster names and descriptions using OpenAI.
    
    Args:
        cluster_keywords: Dictionary mapping cluster IDs to lists of keywords
        api_key: OpenAI API key
        model: OpenAI model to use
        
    Returns:
        Dictionary mapping cluster IDs to tuples of (name, description)
    """
    if not openai_available:
        st.error("OpenAI package is not installed. Try: pip install openai")
        return {}
    
    if not api_key:
        st.error("OpenAI API key is required for generating cluster names")
        return {}
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Process in batches to avoid context length issues
        batch_size = 5  # Process 5 clusters at a time
        all_results = {}
        
        # Get cluster IDs and prepare batches
        cluster_ids = list(cluster_keywords.keys())
        total_batches = (len(cluster_ids) + batch_size - 1) // batch_size
        
        # Set up progress tracking
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(cluster_ids))
            batch_cluster_ids = cluster_ids[batch_start:batch_end]
            
            progress_text.text(f"Generating names for clusters {batch_start+1}-{batch_end} (batch {batch_idx+1}/{total_batches})")
            
            # Create prompt for this batch
            prompt = """
            You are an expert in SEO and content marketing. Below you'll see several clusters 
            with lists of related keywords. Your task is to give each cluster:
            
            1. A short, clear name (3-6 words)
            2. A brief description (1-2 sentences) explaining the topic and likely search intent
            
            The name should be descriptive and optimized for SEO. The description should 
            help content creators understand what this cluster represents.
            
            FORMAT YOUR RESPONSE AS JSON:
            
            {
              "clusters": [
                {
                  "cluster_id": 1,
                  "cluster_name": "Example Cluster Name",
                  "cluster_description": "Example description of what this cluster represents."
                },
                ...
              ]
            }
            
            Here are the clusters:
            """
            
            for cluster_id in batch_cluster_ids:
                keywords_sample = cluster_keywords[cluster_id][:20]  # Limit to 20 keywords
                prompt += f"\n- Cluster {cluster_id}: {', '.join(keywords_sample)}"
            
            # Try to generate names with OpenAI
            try:
                # Try with response_format for newer models
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.4,
                        response_format={"type": "json_object"},
                        max_tokens=1000
                    )
                except:
                    # Fallback without response_format for older models
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt + "\n\nRespond only with the JSON."}],
                        temperature=0.4,
                        max_tokens=1000
                    )
                
                content = response.choices[0].message.content.strip()
                
                # Extract JSON from markdown code blocks if present
                json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
                json_matches = re.findall(json_pattern, content)
                
                if json_matches:
                    content = json_matches[0]  # Take the first JSON code block
                
                # Parse JSON
                try:
                    data = json.loads(content)
                    
                    if "clusters" in data and isinstance(data["clusters"], list):
                        for item in data["clusters"]:
                            c_id = item.get("cluster_id")
                            if c_id is not None:
                                try:
                                    c_id = int(c_id)
                                    c_name = item.get("cluster_name", f"Cluster {c_id}")
                                    c_desc = item.get("cluster_description", "No description provided")
                                    all_results[c_id] = (c_name, c_desc)
                                except (ValueError, TypeError):
                                    pass
                except json.JSONDecodeError:
                    # Try regex extraction as fallback
                    for cluster_id in batch_cluster_ids:
                        name_pattern = rf'cluster_id["\s:]+{cluster_id}["\s,}}]+\s*cluster_name["\s:]+([^"]+)["\s,}}]+'
                        desc_pattern = rf'cluster_id["\s:]+{cluster_id}["\s,}}]+.*?cluster_description["\s:]+([^"]+)["\s,}}]+'
                        
                        name_matches = re.findall(name_pattern, content)
                        desc_matches = re.findall(desc_pattern, content, re.DOTALL)
                        
                        if name_matches:
                            c_name = name_matches[0].strip()
                            c_desc = desc_matches[0].strip() if desc_matches else f"Group of related keywords (cluster {cluster_id})"
                            all_results[cluster_id] = (c_name, c_desc)
            
            except Exception as e:
                st.warning(f"Error in batch {batch_idx+1}: {str(e)}")
                # Continue with next batch
            
            # Update progress
            progress_bar.progress((batch_idx + 1) / total_batches)
        
        # Clear progress indicators
        progress_text.empty()
        progress_bar.empty()
        
        # Check if we got results for all clusters
        missing_clusters = set(cluster_ids) - set(all_results.keys())
        if missing_clusters:
            st.warning(f"Could not generate names for {len(missing_clusters)} clusters. Using fallback method.")
            
            # Use fallback for missing clusters
            for cluster_id in missing_clusters:
                name = generate_basic_name(cluster_keywords[cluster_id])
                all_results[cluster_id] = (name, f"Group of related keywords about {name.lower()}")
        
        st.success(f"✅ Generated names for {len(all_results)} clusters")
        return all_results
    
    except Exception as e:
        st.error(f"Error generating names with OpenAI: {str(e)}")
        return {}

def generate_cluster_names(df: pd.DataFrame, api_key: Optional[str] = None, 
                         model: str = "gpt-3.5-turbo") -> Dict[int, str]:
    """
    Generate descriptive names for each cluster.
    
    Args:
        df: Dataframe with 'cluster_id' and 'keyword' columns
        api_key: Optional OpenAI API key for AI-powered naming
        model: OpenAI model to use (if api_key provided)
        
    Returns:
        Dictionary mapping cluster IDs to cluster names
    """
    if 'cluster_id' not in df.columns or 'keyword' not in df.columns:
        st.error("Dataframe must have 'cluster_id' and 'keyword' columns")
        return {}
    
    # Group keywords by cluster
    cluster_keywords = {}
    
    # Get representative keywords if available
    if 'is_representative' in df.columns:
        for cluster_id in df['cluster_id'].unique():
            rep_df = df[(df['cluster_id'] == cluster_id) & (df['is_representative'] == True)]
            if len(rep_df) > 0:
                cluster_keywords[cluster_id] = rep_df['keyword'].tolist()
            else:
                # Fallback to all keywords if no representatives found
                cluster_keywords[cluster_id] = df[df['cluster_id'] == cluster_id]['keyword'].tolist()
    else:
        # Just use all keywords
        for cluster_id in df['cluster_id'].unique():
            cluster_keywords[cluster_id] = df[df['cluster_id'] == cluster_id]['keyword'].tolist()
    
    # Try AI-powered naming if API key provided
    if api_key and openai_available:
        st.info("Generating AI-powered cluster names...")
        name_results = generate_openai_cluster_names(cluster_keywords, api_key, model)
        
        # Extract just the names (not descriptions) for the return value
        cluster_names = {cluster_id: name for cluster_id, (name, _) in name_results.items()}
        
        # Add cluster descriptions to the dataframe if available
        for cluster_id, (_, description) in name_results.items():
            if 'cluster_description' not in df.columns:
                df['cluster_description'] = ""
            df.loc[df['cluster_id'] == cluster_id, 'cluster_description'] = description
        
        return cluster_names
    
    # Fallback to basic naming
    st.info("Using basic naming method (no AI)...")
    return {cluster_id: generate_basic_name(keywords) for cluster_id, keywords in cluster_keywords.items()}

def extract_keyword_themes(keywords: List[str]) -> Dict[str, int]:
    """
    Extract common themes/topics from a list of keywords.
    
    Args:
        keywords: List of keywords to analyze
        
    Returns:
        Dictionary mapping themes to their frequency
    """
    all_words = []
    for keyword in keywords:
        words = str(keyword).lower().split()
        all_words.extend(words)
    
    # Filter out common words
    stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of'}
    filtered_words = [word for word in all_words if word not in stopwords and len(word) > 2]
    
    # Count word frequencies
    word_counts = Counter(filtered_words)
    
    # Return top 10 themes
    return dict(word_counts.most_common(10))

def detect_name_consistency(df: pd.DataFrame) -> Dict[int, float]:
    """
    Detect if cluster names are consistent with their keywords.
    
    Args:
        df: Dataframe with 'cluster_id', 'cluster_name', and 'keyword' columns
        
    Returns:
        Dictionary mapping cluster IDs to consistency scores (0-1)
    """
    if not all(col in df.columns for col in ['cluster_id', 'cluster_name', 'keyword']):
        return {}
    
    consistency_scores = {}
    
    for cluster_id in df['cluster_id'].unique():
        cluster_df = df[df['cluster_id'] == cluster_id]
        
        # Get cluster name and keywords
        cluster_name = cluster_df['cluster_name'].iloc[0].lower()
        keywords = cluster_df['keyword'].tolist()
        
        # Extract themes from keywords
        themes = extract_keyword_themes(keywords)
        
        # Check if cluster name contains key themes
        name_words = cluster_name.split()
        theme_matches = 0
        
        for theme in themes:
            if theme in name_words:
                theme_matches += 1
        
        # Calculate consistency score
        if len(themes) > 0:
            consistency = min(1.0, theme_matches / min(3, len(themes)))
        else:
            consistency = 0.0
        
        consistency_scores[cluster_id] = consistency
    
    return consistency_scores
