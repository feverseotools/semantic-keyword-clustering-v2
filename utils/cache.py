# utils/cache.py
"""
Caching utilities for the keyword clustering application.

This module provides functions for caching expensive computations
to improve performance during interactive use.
"""

import os
import pickle
import hashlib
import streamlit as st
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from functools import wraps
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def get_cache_key(func_name: str, args: Tuple, kwargs: Dict) -> str:
    """
    Generate a unique cache key for a function call.
    
    Args:
        func_name: Name of the function being cached
        args: Positional arguments to the function
        kwargs: Keyword arguments to the function
        
    Returns:
        A unique string key for caching
    """
    # Convert args and kwargs to a string representation
    arg_string = str(args) + str(sorted(kwargs.items()))
    
    # Create hash
    return f"{func_name}_{hashlib.md5(arg_string.encode()).hexdigest()}"

def cache_dataframe(ttl_minutes: int = 60) -> Callable:
    """
    Custom cache decorator for functions returning pandas DataFrames.
    
    Args:
        ttl_minutes: Time-to-live in minutes for cached results
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> pd.DataFrame:
            # Initialize cache if it doesn't exist
            if "df_cache" not in st.session_state:
                st.session_state.df_cache = {}
            
            # Generate cache key
            cache_key = get_cache_key(func.__name__, args, kwargs)
            
            # Check if result is in cache and not expired
            if cache_key in st.session_state.df_cache:
                timestamp, result = st.session_state.df_cache[cache_key]
                age = datetime.now() - timestamp
                
                if age < timedelta(minutes=ttl_minutes):
                    return result
            
            # If not in cache or expired, compute result
            result = func(*args, **kwargs)
            
            # Cache the result with timestamp
            st.session_state.df_cache[cache_key] = (datetime.now(), result)
            
            return result
        return wrapper
    return decorator

def cache_embeddings(ttl_minutes: int = 120) -> Callable:
    """
    Custom cache decorator for functions returning embeddings (numpy arrays).
    
    Args:
        ttl_minutes: Time-to-live in minutes for cached results
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> np.ndarray:
            # Initialize cache if it doesn't exist
            if "embedding_cache" not in st.session_state:
                st.session_state.embedding_cache = {}
            
            # Generate cache key
            cache_key = get_cache_key(func.__name__, args, kwargs)
            
            # Check if result is in cache and not expired
            if cache_key in st.session_state.embedding_cache:
                timestamp, result = st.session_state.embedding_cache[cache_key]
                age = datetime.now() - timestamp
                
                if age < timedelta(minutes=ttl_minutes):
                    return result
            
            # If not in cache or expired, compute result
            result = func(*args, **kwargs)
            
            # Cache the result with timestamp
            st.session_state.embedding_cache[cache_key] = (datetime.now(), result)
            
            return result
        return wrapper
    return decorator

def cache_api_response(ttl_minutes: int = 240) -> Callable:
    """
    Custom cache decorator for functions making API calls.
    
    Args:
        ttl_minutes: Time-to-live in minutes for cached results
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Initialize cache if it doesn't exist
            if "api_cache" not in st.session_state:
                st.session_state.api_cache = {}
            
            # Generate cache key
            cache_key = get_cache_key(func.__name__, args, kwargs)
            
            # Check if result is in cache and not expired
            if cache_key in st.session_state.api_cache:
                timestamp, result = st.session_state.api_cache[cache_key]
                age = datetime.now() - timestamp
                
                if age < timedelta(minutes=ttl_minutes):
                    return result
            
            # If not in cache or expired, compute result
            result = func(*args, **kwargs)
            
            # Cache the result with timestamp
            st.session_state.api_cache[cache_key] = (datetime.now(), result)
            
            return result
        return wrapper
    return decorator

def clear_cache(cache_type: Optional[str] = None) -> None:
    """
    Clear the specified cache type or all caches.
    
    Args:
        cache_type: Type of cache to clear ('df', 'embedding', 'api', or None for all)
    """
    if cache_type == 'df' and 'df_cache' in st.session_state:
        st.session_state.df_cache = {}
        st.success("DataFrame cache cleared")
    
    elif cache_type == 'embedding' and 'embedding_cache' in st.session_state:
        st.session_state.embedding_cache = {}
        st.success("Embedding cache cleared")
    
    elif cache_type == 'api' and 'api_cache' in st.session_state:
        st.session_state.api_cache = {}
        st.success("API cache cleared")
    
    elif cache_type is None:
        # Clear all caches
        for cache in ['df_cache', 'embedding_cache', 'api_cache']:
            if cache in st.session_state:
                st.session_state[cache] = {}
        st.success("All caches cleared")
    
    else:
        st.warning(f"Unknown cache type: {cache_type}")

def get_cache_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about the current caches.
    
    Returns:
        Dictionary with cache statistics
    """
    cache_info = {}
    
    # DataFrame cache
    if 'df_cache' in st.session_state:
        df_cache = st.session_state.df_cache
        cache_info['df_cache'] = {
            'size': len(df_cache),
            'oldest': min([timestamp for timestamp, _ in df_cache.values()]) if df_cache else None,
            'newest': max([timestamp for timestamp, _ in df_cache.values()]) if df_cache else None
        }
    
    # Embedding cache
    if 'embedding_cache' in st.session_state:
        embedding_cache = st.session_state.embedding_cache
        cache_info['embedding_cache'] = {
            'size': len(embedding_cache),
            'oldest': min([timestamp for timestamp, _ in embedding_cache.values()]) if embedding_cache else None,
            'newest': max([timestamp for timestamp, _ in embedding_cache.values()]) if embedding_cache else None
        }
    
    # API cache
    if 'api_cache' in st.session_state:
        api_cache = st.session_state.api_cache
        cache_info['api_cache'] = {
            'size': len(api_cache),
            'oldest': min([timestamp for timestamp, _ in api_cache.values()]) if api_cache else None,
            'newest': max([timestamp for timestamp, _ in api_cache.values()]) if api_cache else None
        }
    
    return cache_info

def invalidate_cache_by_pattern(pattern: str) -> int:
    """
    Invalidate cache entries matching a pattern.
    
    Args:
        pattern: String pattern to match against cache keys
        
    Returns:
        Number of cache entries invalidated
    """
    invalidated = 0
    
    for cache_name in ['df_cache', 'embedding_cache', 'api_cache']:
        if cache_name in st.session_state:
            cache = st.session_state[cache_name]
            keys_to_remove = [key for key in cache if pattern in key]
            
            for key in keys_to_remove:
                del cache[key]
                invalidated += 1
    
    return invalidated

# Apply cache_data decorator equivalents to common expensive functions
def apply_cache_decorators() -> None:
    """
    Apply cache decorators to common expensive functions.
    This function should be called at application startup.
    """
    # This function would be used if we were dynamically applying decorators
    # In practice, we'll use the decorators directly on the function definitions
    pass
