# __init__.py
"""
Semantic Keyword Clustering Tool
--------------------------------

A modular application for clustering keywords based on semantic similarity 
and search intent classification.

This package provides tools for:
- Semantic analysis of keywords
- Clustering based on embeddings
- Search intent classification
- Visualization of keyword clusters
- Content strategy recommendations
"""

__version__ = '1.0.0'
__author__ = 'Your Name'

# Make key functions available at package level
from modules.preprocessing import preprocess_keywords
from modules.embeddings import generate_embeddings
from modules.clustering import run_clustering
from modules.search_intent import classify_search_intent_ml

# Version info
VERSION_INFO = {
    'version': __version__,
    'name': 'Semantic Keyword Clustering',
    'description': 'Tool for clustering keywords by meaning and search intent',
    'required_python': '>=3.8.0'
}
