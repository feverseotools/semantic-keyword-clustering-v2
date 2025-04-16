# utils/constants.py
"""
Constants and configuration values for the keyword clustering application.

This module provides central storage for constants, configuration values,
and default settings used throughout the application.
"""

# Application information
APP_NAME = "Semantic Keyword Clustering"
APP_DESCRIPTION = "Group keywords by meaning and search intent for better content strategy"
APP_VERSION = "1.0.0"

# Default values
DEFAULT_NUM_CLUSTERS = 10
DEFAULT_PCA_VARIANCE = 95
DEFAULT_MAX_PCA_COMPONENTS = 100
DEFAULT_MIN_DF = 1
DEFAULT_MAX_DF = 0.95
DEFAULT_GPT_MODEL = "gpt-3.5-turbo"

# HTML/CSS color codes for each intent type
INTENT_COLORS = {
    "Informational": "#2196F3",  # Blue
    "Navigational": "#4CAF50",   # Green
    "Transactional": "#FF9800",  # Orange
    "Commercial": "#9C27B0",     # Purple
    "Mixed Intent": "#757575"    # Gray
}

# Journey phase colors
JOURNEY_COLORS = {
    "Research Phase (Early)": "#2196F3",            # Blue
    "Research-to-Consideration Transition": "#26A69A",  # Teal
    "Consideration Phase (Middle)": "#4CAF50",      # Green
    "Consideration-to-Decision Transition": "#9C27B0",  # Purple
    "Decision Phase (Late)": "#FF9800",             # Orange
    "Mixed Journey Stages": "#757575",              # Gray
    "Unknown": "#BDBDBD"                            # Light Gray
}

# Search intent patterns for classification
SEARCH_INTENT_PATTERNS = {
    "Informational": {
        "prefixes": [
            "how", "what", "why", "when", "where", "who", "which",
            "can", "does", "is", "are", "will", "should", "do", "did",
            "guide", "tutorial", "learn", "understand", "explain"
        ],
        "suffixes": ["definition", "meaning", "examples", "ideas", "guide", "tutorial"],
        "exact_matches": [
            "guide to", "how-to", "tutorial", "resources", "information", "knowledge",
            "examples of", "definition of", "explanation", "steps to", "learn about",
            "facts about", "history of", "benefits of", "causes of", "types of"
        ],
        "weight": 1.0
    },
    
    "Navigational": {
        "prefixes": ["go to", "visit", "website", "homepage", "home page", "sign in", "login"],
        "suffixes": ["login", "website", "homepage", "official", "online"],
        "exact_matches": [
            "login", "sign in", "register", "create account", "download", "official website",
            "official site", "homepage", "contact", "support", "customer service", "app"
        ],
        "brand_indicators": True,  # Presence of brand names indicates navigational intent
        "weight": 1.2  # Navigational intent is often more clear-cut
    },
    
    "Transactional": {
        "prefixes": ["buy", "purchase", "order", "shop", "get"],
        "suffixes": [
            "for sale", "discount", "deal", "coupon", "price", "cost", "cheap", "online", 
            "free", "download", "subscription", "trial"
        ],
        "exact_matches": [
            "buy", "purchase", "order", "shop", "subscribe", "download", "free trial",
            "coupon code", "discount", "deal", "sale", "cheap", "best price", "near me",
            "shipping", "delivery", "in stock", "available", "pay", "checkout"
        ],
        "weight": 1.5  # Strong transactional signals are highly valuable
    },
    
    "Commercial": {
        "prefixes": ["best", "top", "review", "compare", "vs", "versus"],
        "suffixes": [
            "review", "reviews", "comparison", "vs", "versus", "alternative", "alternatives", 
            "recommendation", "recommendations", "comparison", "guide"
        ],
        "exact_matches": [
            "best", "top", "vs", "versus", "comparison", "compare", "review", "reviews", 
            "rating", "ratings", "ranked", "recommended", "alternative", "alternatives",
            "pros and cons", "features", "worth it", "should i buy", "is it good"
        ],
        "weight": 1.2  # Commercial intent signals future transactions
    }
}

# Content format recommendations based on intent
CONTENT_RECOMMENDATIONS = {
    "Informational": [
        "How-to guides",
        "Tutorials",
        "Explanatory articles",
        "FAQ pages",
        "Educational videos",
        "Infographics",
        "Glossaries",
        "Step-by-step instructions"
    ],
    "Navigational": [
        "Landing pages",
        "About us/Contact pages",
        "Sitemaps",
        "Directory listings",
        "Location pages",
        "App download pages",
        "Login/account pages"
    ],
    "Transactional": [
        "Product pages",
        "Service pages",
        "Pricing tables",
        "Shopping carts",
        "Checkout pages",
        "Special offers/deals",
        "Free trial pages",
        "Call-to-action focused content"
    ],
    "Commercial": [
        "Product comparisons",
        "Buying guides",
        "Reviews",
        "Testimonials",
        "Case studies",
        "Product feature matrices",
        "Alternative/competitor analysis",
        "Best-of lists"
    ],
    "Mixed Intent": [
        "Comprehensive guides",
        "Mixed format content",
        "Interactive tools",
        "Conversion-focused educational content",
        "Product guides with purchasing options"
    ]
}

# Embedding model configurations
EMBEDDING_MODELS = {
    "OpenAI": {
        "default": "text-embedding-3-small",
        "dimension": 1536,
        "requires_api_key": True
    },
    "SentenceTransformers": {
        "default": "paraphrase-multilingual-MiniLM-L12-v2",
        "dimension": 384,
        "requires_api_key": False
    },
    "TF-IDF": {
        "default": None,
        "max_features": 300,
        "requires_api_key": False
    }
}

# Language model configurations for cluster naming
LLM_MODELS = {
    "gpt-3.5-turbo": {
        "provider": "OpenAI",
        "temperature": 0.4,
        "max_tokens": 1000,
        "requires_api_key": True
    },
    "gpt-4": {
        "provider": "OpenAI",
        "temperature": 0.3,
        "max_tokens": 1000,
        "requires_api_key": True
    }
}

# Supported languages for text processing
SUPPORTED_LANGUAGES = [
    "English",
    "Spanish",
    "French",
    "German",
    "Dutch",
    "Italian",
    "Portuguese",
    "Swedish",
    "Norwegian",
    "Danish",
    "Greek",
    "Romanian"
]

# Default prompts for OpenAI cluster naming
DEFAULT_NAMING_PROMPT = """
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
"""

# Export format configurations
EXPORT_FORMATS = {
    "CSV": {
        "mime": "text/csv",
        "extension": ".csv"
    },
    "Excel": {
        "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "extension": ".xlsx"
    },
    "JSON": {
        "mime": "application/json",
        "extension": ".json"
    }
}
