# data/__init__.py
"""
Sample data and resources for the keyword clustering application.

This package contains sample datasets, test files, and other resources
that can be used for testing or as examples in the application.
"""

import os
from pathlib import Path

# Define paths to sample data
DATA_DIR = Path(__file__).parent
SAMPLE_CSV_PATH = DATA_DIR / "sample_keywords.csv"

def get_sample_data_path():
    """Returns the path to the sample keywords CSV file."""
    return SAMPLE_CSV_PATH
