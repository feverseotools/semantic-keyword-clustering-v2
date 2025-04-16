# utils/helpers.py
"""
Utility helper functions for the keyword clustering application.

This module provides miscellaneous helper functions that are used
across different parts of the application.
"""

import os
import re
import time
import random
import streamlit as st
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np

def generate_sample_csv() -> str:
    """
    Generate a sample CSV file with keywords and metrics.
    
    Returns:
        CSV content as a string
    """
    # Create header row
    header = ["Keyword", "search_volume", "competition", "cpc"]
    months = [f"month{i}" for i in range(1, 13)]
    header += months
    
    # Create sample data
    sample_data = [
        ["running shoes", 5400, 0.75, 1.25] + [450 + i*10 for i in range(12)],
        ["nike shoes", 8900, 0.82, 1.78] + [700 + i*20 for i in range(12)],
        ["adidas sneakers", 3200, 0.65, 1.12] + [260 + i*10 for i in range(12)],
        ["hiking boots", 2800, 0.45, 0.89] + [230 + i*10 for i in range(12)],
        ["women's running shoes", 4100, 0.68, 1.35] + [340 + i*10 for i in range(12)],
        ["best running shoes 2025", 3100, 0.78, 1.52] + [280 + i*10 for i in range(12)],
        ["how to choose running shoes", 2500, 0.42, 0.95] + [220 + i*10 for i in range(12)],
        ["running shoes for flat feet", 1900, 0.56, 1.28] + [170 + i*10 for i in range(12)],
        ["trail running shoes reviews", 1700, 0.64, 1.42] + [150 + i*10 for i in range(12)],
        ["buy nike air zoom", 1500, 0.87, 1.95] + [130 + i*10 for i in range(12)],
        ["running shoe size chart", 2200, 0.35, 0.78] + [190 + i*10 for i in range(12)],
        ["cheap running shoes", 3600, 0.72, 1.15] + [310 + i*10 for i in range(12)],
        ["brooks ghost running shoes", 1300, 0.58, 1.32] + [110 + i*10 for i in range(12)],
        ["running shoes for beginners", 2000, 0.52, 1.05] + [180 + i*10 for i in range(12)],
        ["compare running shoes", 950, 0.61, 1.38] + [80 + i*10 for i in range(12)],
    ]
    
    # Convert to CSV string
    csv_lines = [",".join(map(str, header))]
    for row in sample_data:
        csv_lines.append(",".join(map(str, row)))
    
    return "\n".join(csv_lines)

def generate_random_keywords(count: int = 100, seed: int = 42) -> List[str]:
    """
    Generate random keywords for testing purposes.
    
    Args:
        count: Number of keywords to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of random keywords
    """
    random.seed(seed)
    
    # Lists of components to build keywords
    adjectives = ["best", "top", "cheap", "affordable", "premium", "quality", "new", 
                 "used", "professional", "beginner", "lightweight", "durable", "fast",
                 "comfortable", "stylish", "waterproof", "reliable", "portable"]
    
    nouns = ["shoes", "sneakers", "boots", "trainers", "runners", "footwear",
            "clothing", "jacket", "pants", "shorts", "socks", "hat", "gloves",
            "equipment", "gear", "accessories", "watch", "headphones", "bottle"]
    
    brands = ["nike", "adidas", "puma", "reebok", "asics", "brooks", "new balance",
             "under armour", "saucony", "hoka", "on running", "salomon", "mizuno"]
    
    categories = ["running", "training", "fitness", "gym", "workout", "trail", "hiking",
                 "walking", "tennis", "basketball", "soccer", "golf", "cycling", "swimming"]
    
    prefixes_info = ["how to", "what is", "why", "when to", "where to", "guide to", 
                    "tips for", "best way to", "how do"]
    
    prefixes_trans = ["buy", "purchase", "order", "shop", "get", "find", "price of", "cost of"]
    
    prefixes_comm = ["compare", "review", "vs", "versus", "or", "alternative to"]
    
    suffixes = ["review", "reviews", "for women", "for men", "for kids", "near me", 
               "online", "on sale", "discount", "cheap", "best", "2025"]
    
    # Generate keywords with different patterns
    keywords = []
    
    # Informational keywords (25%)
    for _ in range(count // 4):
        prefix = random.choice(prefixes_info)
        category = random.choice(categories)
        noun = random.choice(nouns)
        kw = f"{prefix} {random.choice([category, ''])} {noun}".strip()
        keywords.append(kw)
    
    # Transactional keywords (25%)
    for _ in range(count // 4):
        prefix = random.choice(prefixes_trans)
        adj = random.choice(["", random.choice(adjectives)])
        brand = random.choice(["", random.choice(brands)])
        category = random.choice(["", random.choice(categories)])
        noun = random.choice(nouns)
        suffix = random.choice(["", random.choice(suffixes)])
        
        components = [prefix, adj, brand, category, noun, suffix]
        random.shuffle(components[:4])  # Shuffle the first 4 components
        kw = " ".join(component for component in components if component)
        keywords.append(kw)
    
    # Commercial keywords (25%)
    for _ in range(count // 4):
        prefix = random.choice(prefixes_comm)
        brand1 = random.choice(brands)
        brand2 = random.choice([b for b in brands if b != brand1])
        category = random.choice(["", random.choice(categories)])
        noun = random.choice(nouns)
        
        if "vs" in prefix or "versus" in prefix or "or" in prefix:
            kw = f"{brand1} {random.choice(['vs', 'or', 'versus'])} {brand2} {category} {noun}".strip()
        else:
            kw = f"{prefix} {brand1} {category} {noun}".strip()
        
        keywords.append(kw)
    
    # Navigational + mixed keywords (25%)
    for _ in range(count - len(keywords)):
        # 50% navigational, 50% mixed
        if random.random() < 0.5:
            # Navigational
            brand = random.choice(brands)
            suffix = random.choice(["website", "official site", "login", "store", "shop", "customer service", "contact", "app"])
            kw = f"{brand} {suffix}"
        else:
            # Mixed (random combination)
            components = [
                random.choice([random.choice(adjectives), ""]),
                random.choice([random.choice(brands), ""]),
                random.choice([random.choice(categories), ""]),
                random.choice(nouns),
                random.choice([random.choice(suffixes), ""])
            ]
            kw = " ".join(component for component in components if component)
        
        keywords.append(kw)
    
    # Shuffle and return
    random.shuffle(keywords)
    return keywords[:count]

def format_number(num: Union[int, float], precision: int = 1) -> str:
    """
    Format a number with K, M, B suffixes.
    
    Args:
        num: Number to format
        precision: Decimal precision
        
    Returns:
        Formatted number string
    """
    if num is None:
        return "0"
    
    if num == 0:
        return "0"
        
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    
    # Add suffix
    suffix = ['', 'K', 'M', 'B', 'T'][min(magnitude, 4)]
    
    # Format with appropriate precision
    if magnitude > 0:
        return f"{num:.{precision}f}{suffix}"
    else:
        return f"{num:.0f}"

def clean_text_for_display(text: str, max_length: int = 100) -> str:
    """
    Clean and truncate text for display purposes.
    
    Args:
        text: Text to clean
        max_length: Maximum length before truncation
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Truncate if needed
    if len(text) > max_length:
        text = text[:max_length-3] + "..."
    
    return text

def measure_execution_time(func):
    """
    Decorator to measure execution time of a function.
    
    Args:
        func: Function to decorate
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"Function {func.__name__} executed in {execution_time:.2f} seconds")
        
        return result
    
    return wrapper

def get_device_info() -> Dict[str, Any]:
    """
    Get information about the user's device and capabilities.
    
    Returns:
        Dictionary with device information
    """
    device_info = {
        "streamlit_version": st.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__
    }
    
    # Check for optional dependencies
    try:
        import torch
        device_info["torch_version"] = torch.__version__
        device_info["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        device_info["torch_version"] = None
        device_info["cuda_available"] = False
    
    try:
        import tensorflow as tf
        device_info["tensorflow_version"] = tf.__version__
        device_info["gpu_available"] = len(tf.config.list_physical_devices('GPU')) > 0
    except ImportError:
        device_info["tensorflow_version"] = None
        device_info["gpu_available"] = False
    
    return device_info

def get_dataset_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate statistics about a keyword dataset.
    
    Args:
        df: Dataframe with 'keyword' column
        
    Returns:
        Dictionary with statistics
    """
    if 'keyword' not in df.columns:
        return {}
    
    # Calculate word count for each keyword
    df_stats = df.copy()
    df_stats['word_count'] = df_stats['keyword'].apply(lambda x: len(str(x).split()))
    
    stats = {
        "total_keywords": len(df),
        "unique_keywords": df['keyword'].nunique(),
        "avg_word_count": df_stats['word_count'].mean(),
        "min_word_count": df_stats['word_count'].min(),
        "max_word_count": df_stats['word_count'].max(),
        "common_words": []
    }
    
    # Find common words
    all_words = []
    for keyword in df['keyword']:
        words = str(keyword).lower().split()
        all_words.extend(words)
    
    # Count word frequencies
    from collections import Counter
    word_counts = Counter(all_words)
    
    # Get top 10 words
    stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of', 'is', 'are'}
    common_words = [(word, count) for word, count in word_counts.most_common(20) if word not in stopwords and len(word) > 1]
    stats["common_words"] = common_words[:10]
    
    # Search volume stats if available
    if 'search_volume' in df.columns:
        stats["total_search_volume"] = df['search_volume'].sum()
        stats["avg_search_volume"] = df['search_volume'].mean()
        stats["min_search_volume"] = df['search_volume'].min()
        stats["max_search_volume"] = df['search_volume'].max()
    
    return stats

def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.
    
    Args:
        items: List to split
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def flatten_nested_dict(d: Dict[Any, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flatten a nested dictionary into a flat dictionary with concatenated keys.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key string for recursion
        sep: Separator between key levels
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_nested_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)
