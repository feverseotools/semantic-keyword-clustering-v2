# modules/search_intent.py
"""
Search intent classification module.

This module provides functions for classifying keywords by search intent
(informational, navigational, transactional, commercial) based on
linguistic patterns and keyword features.
"""

import re
import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Dict, Optional, Any, Union, Tuple
from collections import Counter

# Search intent classification patterns
# Comprehensive patterns based on SEO industry standards
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
        "keyword_patterns": [
            r'\bhow\s+to\b', r'\bwhat\s+is\b', r'\bwhy\s+is\b', r'\bwhen\s+to\b', 
            r'\bwhere\s+to\b', r'\bwho\s+is\b', r'\bwhich\b.*\bbest\b',
            r'\bdefinition\b', r'\bmeaning\b', r'\bexamples?\b', r'\btips\b',
            r'\btutorials?\b', r'\bguide\b', r'\blearn\b', r'\bsteps?\b',
            r'\bversus\b', r'\bvs\b', r'\bcompared?\b', r'\bdifference\b'
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
        "keyword_patterns": [
            r'\blogin\b', r'\bsign\s+in\b', r'\bwebsite\b', r'\bhomepage\b', r'\bportal\b',
            r'\baccount\b', r'\bofficial\b', r'\bdashboard\b', r'\bdownload\b.*\bfrom\b',
            r'\bcontact\b', r'\baddress\b', r'\blocation\b', r'\bdirections?\b',
            r'\bmap\b', r'\btrack\b.*\border\b', r'\bmy\s+\w+\s+account\b'
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
        "keyword_patterns": [
            r'\bbuy\b', r'\bpurchase\b', r'\border\b', r'\bshop\b', r'\bstores?\b',
            r'\bprice\b', r'\bcost\b', r'\bcheap\b', r'\bdiscount\b', r'\bdeal\b',
            r'\bsale\b', r'\bcoupon\b', r'\bpromo\b', r'\bfree\s+shipping\b',
            r'\bnear\s+me\b', r'\bshipping\b', r'\bdelivery\b', r'\bcheck\s*out\b',
            r'\bin\s+stock\b', r'\bavailable\b', r'\bsubscribe\b', r'\bdownload\b',
            r'\binstall\b', r'\bfor\s+sale\b', r'\bhire\b', r'\brent\b'
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
        "keyword_patterns": [
            r'\bbest\b', r'\btop\b', r'\breview\b', r'\bcompare\b', r'\bcompari(son|ng)\b', 
            r'\bvs\b', r'\bversus\b', r'\balternatives?\b', r'\brated\b', r'\branking\b',
            r'\bworth\s+it\b', r'\bshould\s+I\s+buy\b', r'\bis\s+it\s+good\b',
            r'\bpros\s+and\s+cons\b', r'\badvantages?\b', r'\bdisadvantages?\b',
            r'\bfeatures\b', r'\bspecifications?\b', r'\bwhich\s+(is\s+)?(the\s+)?best\b'
        ],
        "weight": 1.2  # Commercial intent signals future transactions
    }
}

# HTML/CSS color codes for each intent type
INTENT_COLORS = {
    "Informational": "#2196F3",  # Blue
    "Navigational": "#4CAF50",   # Green
    "Transactional": "#FF9800",  # Orange
    "Commercial": "#9C27B0",     # Purple
    "Mixed Intent": "#757575"    # Gray
}

def extract_features_for_intent(keyword: str, search_intent_description: str = "") -> Dict[str, Any]:
    """
    Extract features from a keyword for intent classification.
    
    Args:
        keyword: The keyword to analyze
        search_intent_description: Optional context to enhance classification
        
    Returns:
        Dictionary of features
    """
    if not isinstance(keyword, str):
        return {}
    
    # Features to extract
    features = {
        "keyword_length": len(keyword.split()),
        "keyword_lower": keyword.lower(),
        "has_informational_prefix": False,
        "has_navigational_prefix": False,
        "has_transactional_prefix": False,
        "has_commercial_prefix": False,
        "has_informational_suffix": False,
        "has_navigational_suffix": False,
        "has_transactional_suffix": False,
        "has_commercial_suffix": False,
        "is_informational_exact_match": False,
        "is_navigational_exact_match": False,
        "is_transactional_exact_match": False,
        "is_commercial_exact_match": False,
        "informational_pattern_matches": 0,
        "navigational_pattern_matches": 0,
        "transactional_pattern_matches": 0,
        "commercial_pattern_matches": 0,
        "includes_brand": False,
        "includes_product_modifier": False,
        "includes_price_modifier": False,
        "local_intent": False,
        "modal_verbs": False  # signals a question typically
    }
    
    keyword_lower = keyword.lower()
    
    # Check prefixes
    words = keyword_lower.split()
    if words:
        first_word = words[0]
        for intent_type, patterns in SEARCH_INTENT_PATTERNS.items():
            if any(first_word == prefix.lower() for prefix in patterns["prefixes"]):
                features[f"has_{intent_type.lower()}_prefix"] = True
    
    # Check suffixes 
    if words and len(words) > 1:
        last_word = words[-1]
        for intent_type, patterns in SEARCH_INTENT_PATTERNS.items():
            if any(last_word == suffix.lower() for suffix in patterns["suffixes"]):
                features[f"has_{intent_type.lower()}_suffix"] = True
    
    # Check exact matches
    for intent_type, patterns in SEARCH_INTENT_PATTERNS.items():
        for exact_match in patterns["exact_matches"]:
            if exact_match.lower() in keyword_lower:
                features[f"is_{intent_type.lower()}_exact_match"] = True
                break
    
    # Check pattern matches
    for intent_type, patterns in SEARCH_INTENT_PATTERNS.items():
        match_count = 0
        for pattern in patterns["keyword_patterns"]:
            if re.search(pattern, keyword_lower):
                match_count += 1
        features[f"{intent_type.lower()}_pattern_matches"] = match_count
    
    # Additional features
    features["local_intent"] = any(term in keyword_lower for term in [
        "near me", "nearby", "in my area", "close to me", "closest", "local"
    ])
    
    features["modal_verbs"] = any(modal in keyword_lower.split() for modal in [
        "can", "could", "should", "would", "will", "may", "might"
    ])
    
    features["includes_price_modifier"] = any(term in keyword_lower for term in [
        "price", "cost", "cheap", "expensive", "affordable", "discount", 
        "offer", "deal", "coupon", "free"
    ])
    
    features["includes_product_modifier"] = any(term in keyword_lower for term in [
        "best", "top", "cheap", "premium", "quality", "new", "used", 
        "refurbished", "alternative", "recommended"
    ])
    
    return features

@st.cache_data
def classify_search_intent_ml(keywords: List[str], 
                             search_intent_description: str = "", 
                             cluster_name: str = "") -> Dict[str, Any]:
    """
    Classify the search intent of a list of keywords using a ML-inspired approach.
    
    Args:
        keywords: List of keywords to classify
        search_intent_description: Optional context description
        cluster_name: Optional cluster name for additional context
        
    Returns:
        Dictionary with primary intent, scores, and evidence
    """
    if not keywords:
        return {
            "primary_intent": "Unknown",
            "scores": {
                "Informational": 25,
                "Navigational": 25,
                "Transactional": 25,
                "Commercial": 25
            },
            "evidence": {}
        }
    
    # Extract features for all keywords (limit to first 20 for performance)
    all_features = []
    for keyword in keywords[:min(len(keywords), 20)]:
        features = extract_features_for_intent(keyword, search_intent_description)
        all_features.append(features)
    
    # Aggregate features
    informational_signals = []
    navigational_signals = []
    transactional_signals = []
    commercial_signals = []
    
    # Count pattern matches across all features
    for features in all_features:
        # Informational signals
        if features["has_informational_prefix"]:
            informational_signals.append("Has informational prefix")
        if features["has_informational_suffix"]:
            informational_signals.append("Has informational suffix")
        if features["is_informational_exact_match"]:
            informational_signals.append("Contains informational phrase")
        if features["informational_pattern_matches"] > 0:
            informational_signals.append(f"Matches {features['informational_pattern_matches']} informational patterns")
        if features["modal_verbs"]:
            informational_signals.append("Contains question-like modal verb")
            
        # Navigational signals
        if features["has_navigational_prefix"]:
            navigational_signals.append("Has navigational prefix")
        if features["has_navigational_suffix"]:
            navigational_signals.append("Has navigational suffix")
        if features["is_navigational_exact_match"]:
            navigational_signals.append("Contains navigational phrase")
        if features["navigational_pattern_matches"] > 0:
            navigational_signals.append(f"Matches {features['navigational_pattern_matches']} navigational patterns")
        if features["includes_brand"]:
            navigational_signals.append("Includes brand name")
            
        # Transactional signals
        if features["has_transactional_prefix"]:
            transactional_signals.append("Has transactional prefix")
        if features["has_transactional_suffix"]:
            transactional_signals.append("Has transactional suffix")
        if features["is_transactional_exact_match"]:
            transactional_signals.append("Contains transactional phrase")
        if features["transactional_pattern_matches"] > 0:
            transactional_signals.append(f"Matches {features['transactional_pattern_matches']} transactional patterns")
        if features["includes_price_modifier"]:
            transactional_signals.append("Includes price-related term")
        if features["local_intent"]:
            transactional_signals.append("Shows local intent (near me, etc.)")
            
        # Commercial signals
        if features["has_commercial_prefix"]:
            commercial_signals.append("Has commercial prefix")
        if features["has_commercial_suffix"]:
            commercial_signals.append("Has commercial suffix")
        if features["is_commercial_exact_match"]:
            commercial_signals.append("Contains commercial phrase")
        if features["commercial_pattern_matches"] > 0:
            commercial_signals.append(f"Matches {features['commercial_pattern_matches']} commercial patterns")
        if features["includes_product_modifier"]:
            commercial_signals.append("Includes product comparison term")
    
    # Calculate scores based on unique signals
    info_signals = set(informational_signals)
    nav_signals = set(navigational_signals)
    trans_signals = set(transactional_signals)
    comm_signals = set(commercial_signals)
    
    # Calculate relative proportions (with weighting)
    info_weight = SEARCH_INTENT_PATTERNS["Informational"]["weight"]
    nav_weight = SEARCH_INTENT_PATTERNS["Navigational"]["weight"]
    trans_weight = SEARCH_INTENT_PATTERNS["Transactional"]["weight"]
    comm_weight = SEARCH_INTENT_PATTERNS["Commercial"]["weight"]
    
    info_score = len(info_signals) * info_weight
    nav_score = len(nav_signals) * nav_weight
    trans_score = len(trans_signals) * trans_weight
    comm_score = len(comm_signals) * comm_weight
    
    # Check description for explicit mentions
    if search_intent_description:
        desc_lower = search_intent_description.lower()
        if re.search(r'\binformational\b|\binformation\s+intent\b|\binformation\s+search\b|\bleaning\b|\bquestion\b', desc_lower):
            info_score += 5
        if re.search(r'\bnavigational\b|\bnavigate\b|\bfind\s+\w+\s+website\b|\bfind\s+\w+\s+page\b|\baccess\b', desc_lower):
            nav_score += 5
        if re.search(r'\btransactional\b|\bbuy\b|\bpurchase\b|\bshopping\b|\bsale\b|\btransaction\b', desc_lower):
            trans_score += 5
        if re.search(r'\bcommercial\b|\bcompar(e|ing|ison)\b|\breview\b|\balternative\b|\bbest\b', desc_lower):
            comm_score += 5
    
    # Check cluster name for signals
    if cluster_name:
        name_lower = cluster_name.lower()
        if re.search(r'\bhow\b|\bwhat\b|\bwhy\b|\bwhen\b|\bguide\b|\btutorial\b', name_lower):
            info_score += 3
        if re.search(r'\bwebsite\b|\bofficial\b|\blogin\b|\bportal\b|\bdownload\b', name_lower):
            nav_score += 3
        if re.search(r'\bbuy\b|\bshop\b|\bpurchase\b|\bsale\b|\bdiscount\b|\bcost\b|\bprice\b', name_lower):
            trans_score += 3
        if re.search(r'\bbest\b|\btop\b|\breview\b|\bcompare\b|\bvs\b|\balternative\b', name_lower):
            comm_score += 3
    
    # Normalize to percentages
    total_score = max(1, info_score + nav_score + trans_score + comm_score)
    info_pct = (info_score / total_score) * 100
    nav_pct = (nav_score / total_score) * 100
    trans_pct = (trans_score / total_score) * 100
    comm_pct = (comm_score / total_score) * 100
    
    # Prepare scores
    scores = {
        "Informational": info_pct,
        "Navigational": nav_pct,
        "Transactional": trans_pct,
        "Commercial": comm_pct
    }
    
    # Find primary intent (highest score)
    primary_intent = max(scores, key=scores.get)
    
    # If the highest score is less than 30%, consider it mixed intent
    max_score = max(scores.values())
    if max_score < 30:
        # Check if there's a close second
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and (sorted_scores[0] - sorted_scores[1] < 10):
            primary_intent = "Mixed Intent"
    
    # Collect evidence for the primary intent
    evidence = {
        "Informational": list(info_signals),
        "Navigational": list(nav_signals),
        "Transactional": list(trans_signals),
        "Commercial": list(comm_signals)
    }
    
    return {
        "primary_intent": primary_intent,
        "scores": scores,
        "evidence": evidence
    }

def analyze_keyword_intent(keyword: str) -> Dict[str, Any]:
    """
    Analyze the intent of a single keyword.
    
    Args:
        keyword: Keyword to analyze
        
    Returns:
        Dictionary with classification results
    """
    return classify_search_intent_ml([keyword])

def classify_cluster_intents(df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """
    Classify the search intent for each cluster in the dataframe.
    
    Args:
        df: Dataframe with 'cluster_id' and 'keyword' columns
        
    Returns:
        Dictionary mapping cluster IDs to their intent classification
    """
    if 'cluster_id' not in df.columns or 'keyword' not in df.columns:
        st.error("Dataframe must have 'cluster_id' and 'keyword' columns")
        return {}
    
    # Get unique cluster IDs
    cluster_ids = df['cluster_id'].unique()
    
    # Store results
    cluster_intents = {}
    
    # Create progress indicators
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    # Analyze each cluster
    for i, cluster_id in enumerate(cluster_ids):
        progress_text.text(f"Analyzing search intent for cluster {i+1}/{len(cluster_ids)}")
        
        # Get keywords for this cluster
        cluster_keywords = df[df['cluster_id'] == cluster_id]['keyword'].tolist()
        
        # Get cluster name if available
        cluster_name = ""
        if 'cluster_name' in df.columns:
            cluster_names = df[df['cluster_id'] == cluster_id]['cluster_name'].unique()
            if len(cluster_names) > 0:
                cluster_name = cluster_names[0]
        
        # Classify intent
        intent_data = classify_search_intent_ml(cluster_keywords, cluster_name=cluster_name)
        
        # Store results
        cluster_intents[cluster_id] = intent_data
        
        # Update progress
        progress_bar.progress((i + 1) / len(cluster_ids))
    
    # Clear progress indicators
    progress_text.empty()
    progress_bar.empty()
    
    st.success(f"✅ Analyzed search intent for {len(cluster_ids)} clusters")
    return cluster_intents

def get_customer_journey_phase(intent_data: Dict[str, Any]) -> str:
    """
    Determine the customer journey phase based on intent scores.
    
    Args:
        intent_data: Intent classification data
        
    Returns:
        Customer journey phase as a string
    """
    scores = intent_data.get("scores", {})
    if not scores:
        return "Unknown"
    
    # Extract scores
    info_score = scores.get("Informational", 0)
    comm_score = scores.get("Commercial", 0)
    trans_score = scores.get("Transactional", 0)
    
    # Determine phase based on dominant scores
    if info_score > 50:
        return "Research Phase (Early)"
    elif comm_score > 50:
        return "Consideration Phase (Middle)"
    elif trans_score > 50:
        return "Decision Phase (Late)"
    elif info_score > 25 and comm_score > 25:
        return "Research-to-Consideration Transition"
    elif comm_score > 25 and trans_score > 25:
        return "Consideration-to-Decision Transition"
    else:
        return "Mixed Journey Stages"

def get_intent_color(intent: str) -> str:
    """
    Get the HTML/CSS color code for a given intent.
    
    Args:
        intent: The search intent
        
    Returns:
        Color code as a string
    """
    return INTENT_COLORS.get(intent, "#757575")  # Default to gray

def suggest_content_formats(intent: str) -> List[str]:
    """
    Suggest content formats based on search intent.
    
    Args:
        intent: The search intent
        
    Returns:
        List of suggested content formats
    """
    if intent == "Informational":
        return [
            "How-to guides",
            "Tutorials",
            "Explanatory articles",
            "FAQs",
            "Educational videos",
            "Infographics",
            "Glossaries",
            "Step-by-step instructions"
        ]
    elif intent == "Navigational":
        return [
            "Landing pages",
            "About us/Contact pages",
            "Sitemaps",
            "Directory listings",
            "Location pages",
            "App download pages",
            "Login/account pages"
        ]
    elif intent == "Transactional":
        return [
            "Product pages",
            "Service pages",
            "Pricing tables",
            "Shopping carts",
            "Checkout pages",
            "Special offers/deals",
            "Free trial pages",
            "Call-to-action focused content"
        ]
    elif intent == "Commercial":
        return [
            "Product comparisons",
            "Buying guides",
            "Reviews",
            "Testimonials",
            "Case studies",
            "Product feature matrices",
            "Alternative/competitor analysis",
            "Best-of lists"
        ]
    else:  # Mixed Intent
        return [
            "Comprehensive guides",
            "Mixed format content",
            "Interactive tools",
            "Conversion-focused educational content",
            "Product guides with purchasing options"
        ]
