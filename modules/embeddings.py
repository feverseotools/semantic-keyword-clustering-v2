# modules/embeddings.py
"""
Embedding generation module for keyword clustering.

This module provides functions to generate semantic embeddings for keywords
using various methods including OpenAI, SentenceTransformers, and TF-IDF.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Dict, Optional, Any, Union, Tuple

# Try to import optional dependencies
try:
    from openai import OpenAI
    openai_available = True
except ImportError:
    openai_available = False

try:
    from sentence_transformers import SentenceTransformer
    sentence_transformers_available = True
except ImportError:
    sentence_transformers_available = False

# Import scikit-learn for TF-IDF (always required)
from sklearn.feature_extraction.text import TfidfVectorizer

def check_embedding_models() -> Tuple[bool, bool, bool]:
    """
    Check which embedding models are available in the current environment.
    
    Returns:
        Tuple of booleans: (openai_available, sentence_transformers_available, spacy_available)
    """
    # Check spaCy
    try:
        import spacy
        spacy_available = True
    except ImportError:
        spacy_available = False
    
    return openai_available, sentence_transformers_available, spacy_available

@st.cache_data
def generate_openai_embeddings(texts: List[str], api_key: str, 
                               model: str = "text-embedding-3-small") -> np.ndarray:
    """
    Generate embeddings using OpenAI's API.
    
    Args:
        texts: List of texts to embed
        api_key: OpenAI API key
        model: OpenAI embedding model to use
        
    Returns:
        NumPy array of embeddings
    """
    if not openai_available:
        st.error("OpenAI package is not installed. Try: pip install openai")
        return np.array([])
    
    if not api_key:
        st.error("OpenAI API key is required for generating OpenAI embeddings")
        return np.array([])
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Process in batches to avoid rate limits
        batch_size = 100
        all_embeddings = []
        
        # Display progress
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        for i in range(0, len(texts), batch_size):
            batch_end = min(i + batch_size, len(texts))
            batch = texts[i:batch_end]
            
            progress_text.text(f"Generating OpenAI embeddings: batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            # Request embeddings
            response = client.embeddings.create(
                model=model,
                input=batch
            )
            
            # Extract embeddings from response
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            # Update progress
            progress_bar.progress(min(1.0, batch_end / len(texts)))
        
        # Clear progress indicators
        progress_text.empty()
        progress_bar.empty()
        
        # Convert list of embeddings to numpy array
        embeddings_array = np.array(all_embeddings)
        
        st.success(f"Generated OpenAI embeddings with dimensionality: {embeddings_array.shape[1]}")
        return embeddings_array
    
    except Exception as e:
        st.error(f"Error generating OpenAI embeddings: {str(e)}")
        return np.array([])

@st.cache_data
def generate_sentence_transformer_embeddings(texts: List[str], 
                                            model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> np.ndarray:
    """
    Generate embeddings using SentenceTransformers.
    
    Args:
        texts: List of texts to embed
        model_name: Name of the SentenceTransformer model to use
        
    Returns:
        NumPy array of embeddings
    """
    if not sentence_transformers_available:
        st.error("SentenceTransformers package is not installed. Try: pip install sentence-transformers")
        return np.array([])
    
    try:
        model = SentenceTransformer(model_name)
        
        # Display progress info
        progress_text = st.empty()
        progress_text.text(f"Generating embeddings with SentenceTransformer: {model_name}")
        
        # Generate embeddings
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        
        progress_text.empty()
        
        st.success(f"Generated SentenceTransformer embeddings with dimensionality: {embeddings.shape[1]}")
        return embeddings
    
    except Exception as e:
        st.error(f"Error generating SentenceTransformer embeddings: {str(e)}")
        return np.array([])

@st.cache_data
def generate_tfidf_embeddings(texts: List[str], max_features: int = 300, 
                             min_df: int = 1, max_df: float = 0.95) -> np.ndarray:
    """
    Generate embeddings using TF-IDF vectorization.
    
    Args:
        texts: List of texts to embed
        max_features: Maximum number of features to use
        min_df: Minimum document frequency
        max_df: Maximum document frequency
        
    Returns:
        NumPy array of embeddings
    """
    try:
        # Replace None or empty strings with a space to avoid errors
        clean_texts = [t if isinstance(t, str) and t.strip() else " " for t in texts]
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english'
        )
        
        tfidf_matrix = vectorizer.fit_transform(clean_texts)
        tfidf_array = tfidf_matrix.toarray()
        
        st.success(f"Generated TF-IDF embeddings with dimensionality: {tfidf_array.shape[1]}")
        return tfidf_array
    
    except Exception as e:
        st.error(f"Error generating TF-IDF embeddings: {str(e)}")
        
        # Last resort: random embeddings
        st.warning("Falling back to random embeddings")
        return np.random.rand(len(texts), 100)

@st.cache_data
def generate_embeddings(df: pd.DataFrame, use_openai: bool = False, 
                       api_key: Optional[str] = None, 
                       column: str = 'keyword_processed') -> np.ndarray:
    """
    Generate embeddings for the keywords in the dataframe using the best available method.
    
    Args:
        df: Dataframe containing the keywords
        use_openai: Whether to use OpenAI embeddings (requires API key)
        api_key: OpenAI API key (if use_openai is True)
        column: Column containing the preprocessed text to embed
        
    Returns:
        NumPy array of embeddings
    """
    # Validate input
    if column not in df.columns:
        st.error(f"Column '{column}' not found in dataframe")
        return np.array([])
    
    # Get texts to embed
    texts = df[column].fillna('').tolist()
    
    # Attempt OpenAI embeddings if requested
    if use_openai and openai_available and api_key:
        embeddings = generate_openai_embeddings(texts, api_key)
        if embeddings.size > 0:
            return embeddings
        # If OpenAI fails, fall through to next option
    
    # Attempt SentenceTransformers if available
    if sentence_transformers_available:
        embeddings = generate_sentence_transformer_embeddings(texts)
        if embeddings.size > 0:
            return embeddings
        # If SentenceTransformers fails, fall through to TF-IDF
    
    # Fallback to TF-IDF (always available)
    st.info("Using TF-IDF for embeddings (consider installing sentence-transformers for better results)")
    return generate_tfidf_embeddings(texts)

def reduce_dimensionality(embeddings: np.ndarray, 
                          target_dims: int = 100, 
                          method: str = 'pca') -> np.ndarray:
    """
    Reduce the dimensionality of embeddings.
    
    Args:
        embeddings: Input embeddings
        target_dims: Target dimensionality
        method: Dimensionality reduction method ('pca' or 'umap')
        
    Returns:
        Reduced embeddings
    """
    if embeddings.shape[1] <= target_dims:
        return embeddings
    
    try:
        if method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=target_dims)
            reduced = reducer.fit_transform(embeddings)
            st.info(f"Reduced dimensions from {embeddings.shape[1]} to {target_dims} using PCA")
            return reduced
        
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=target_dims)
                reduced = reducer.fit_transform(embeddings)
                st.info(f"Reduced dimensions from {embeddings.shape[1]} to {target_dims} using UMAP")
                return reduced
            except ImportError:
                st.warning("UMAP not installed. Falling back to PCA.")
                return reduce_dimensionality(embeddings, target_dims, 'pca')
        
        else:
            st.warning(f"Unknown dimensionality reduction method: {method}. Using PCA.")
            return reduce_dimensionality(embeddings, target_dims, 'pca')
            
    except Exception as e:
        st.error(f"Error in dimensionality reduction: {str(e)}")
        return embeddings
