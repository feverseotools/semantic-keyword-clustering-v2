# modules/preprocessing.py
"""
Text preprocessing module for keyword clustering.

This module provides functions to preprocess keywords for semantic analysis,
including tokenization, stopword removal, lemmatization, and more advanced
NLP techniques when available.
"""

import re
import nltk
import pandas as pd
import streamlit as st
from typing import List, Dict, Optional, Any, Union

# Download NLTK resources at startup - with better error handling
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk_resources_available = True
except Exception as e:
    print(f"NLTK resource download error: {str(e)}")
    nltk_resources_available = False

# Try to import optional dependencies with better error handling
try:
    import spacy
    spacy_base_available = True
    # Don't try to load the model on import, just check if spaCy itself is available
    spacy_model_available = False  # Will check when actually trying to load
except ImportError:
    spacy_base_available = False
    spacy_model_available = False

try:
    from textblob import TextBlob
    textblob_available = True
    try:
        # Test TextBlob functionality
        _test_blob = TextBlob("Testing")
        _test_phrases = _test_blob.noun_phrases
        textblob_resources_available = True
    except:
        textblob_resources_available = False
except ImportError:
    textblob_available = False
    textblob_resources_available = False

# Map of language names to spaCy model names
SPACY_LANGUAGE_MODELS = {
    "English": "en_core_web_sm",
    "Spanish": "es_core_news_sm",
    "French": "fr_core_news_sm",
    "German": "de_core_news_sm",
    "Dutch": "nl_core_news_sm",
    "Italian": "it_core_news_sm",
    "Portuguese": "pt_core_news_sm",
    "Swedish": "sv_core_news_sm",
    "Norwegian": "nb_core_news_sm",
    "Danish": "da_core_news_sm",
    "Greek": "el_core_news_sm",
    "Romanian": "ro_core_news_sm"
}

@st.cache_data
def load_spacy_model(language: str) -> Optional[Any]:
    """
    Load a spaCy model for the specified language.
    
    Args:
        language: The language name (e.g., "English", "Spanish")
        
    Returns:
        A loaded spaCy model or None if not available
    """
    if not spacy_base_available:
        st.info("spaCy is not available. Using basic text processing instead.")
        return None
    
    model_name = SPACY_LANGUAGE_MODELS.get(language)
    if not model_name:
        st.info(f"No spaCy model defined for {language}. Using basic processing instead.")
        return None
    
    try:
        model = spacy.load(model_name)
        # If we got here, model loading worked
        global spacy_model_available
        spacy_model_available = True
        return model
    except Exception as e:
        st.info(f"Could not load spaCy model '{model_name}'. This is normal if you haven't installed it. Using basic processing instead.")
        st.info(f"To install the model, run: python -m spacy download {model_name}")
        return None

def basic_preprocessing(text: str) -> str:
    """
    Basic text preprocessing using NLTK if available, with multiple fallbacks.
    
    Args:
        text: Input text to preprocess
        
    Returns:
        Preprocessed text
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Basic tokenization and stopword removal approach
        if nltk_resources_available:
            try:
                # Use NLTK if available
                from nltk.tokenize import word_tokenize
                from nltk.corpus import stopwords
                
                tokens = word_tokenize(text)
                stop_words = set(stopwords.words('english'))
                tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
                
                # Lemmatization if available
                try:
                    from nltk.stem import WordNetLemmatizer
                    lemmatizer = WordNetLemmatizer()
                    tokens = [lemmatizer.lemmatize(t) for t in tokens]
                except Exception:
                    # Skip lemmatization if not available
                    pass
                
                return " ".join(tokens)
            except Exception as e:
                # Fall through to simple approach on error
                print(f"NLTK processing error: {str(e)}")
        
        # Very simple fallback tokenization
        tokens = text.split()
        
        # Simple stopword list
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 
                    'as', 'what', 'when', 'where', 'how', 'why', 'who',
                    'which', 'this', 'that', 'these', 'those', 'is', 'are',
                    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                    'do', 'does', 'did', 'to', 'at', 'in', 'on', 'for', 'with'}
        
        # Remove stopwords and non-alphabetic tokens
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
        
        return " ".join(tokens)
    
    except Exception as e:
        # Ultimate fallback - just return lowercase text
        print(f"Error in basic preprocessing: {str(e)}")
        return text.lower()

def advanced_preprocessing(text: str, spacy_nlp: Any = None) -> str:
    """
    Advanced text preprocessing using spaCy or TextBlob.
    
    Args:
        text: Input text to preprocess
        spacy_nlp: A loaded spaCy model (if available)
        
    Returns:
        Preprocessed text
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    try:
        if spacy_nlp is not None:
            # Use spaCy for advanced NLP
            doc = spacy_nlp(text.lower())
            
            # Extract base tokens, entities, and noun chunks
            tokens = []
            for token in doc:
                if not token.is_stop and token.is_alpha and len(token.text) > 1:
                    tokens.append(token.lemma_)
            
            # Add entities if present
            entities = [ent.text for ent in doc.ents]
            
            # Add noun chunks (for phrases)
            noun_chunks = [chunk.text for chunk in doc.noun_chunks]
            
            # Combine everything
            return " ".join(tokens + entities + noun_chunks)
        
        elif textblob_available and textblob_resources_available:
            # Use TextBlob as a fallback
            blob = TextBlob(text.lower())
            
            # Get words (excluding stopwords)
            common_stops = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 
                          'as', 'what', 'when', 'where', 'how', 'why', 'who',
                          'in', 'on', 'at', 'to', 'for', 'with'}
            words = [w for w in blob.words if len(w) > 1 and w.lower() not in common_stops]
            
            # Get noun phrases
            try:
                noun_phrases = list(blob.noun_phrases)
            except:
                noun_phrases = []
            
            # Combine
            return " ".join(words + noun_phrases)
        
        else:
            # Fall back to basic preprocessing
            return basic_preprocessing(text)
            
    except Exception as e:
        print(f"Error in advanced preprocessing: {str(e)}")
        return basic_preprocessing(text)

def preprocess_keywords(df: pd.DataFrame, language: str = "English") -> pd.DataFrame:
    """
    Preprocess all keywords in the dataframe.
    
    Args:
        df: Dataframe containing a 'keyword' column
        language: Language of the keywords for language-specific processing
        
    Returns:
        Dataframe with added 'keyword_processed' column
    """
    # Validate input
    if 'keyword' not in df.columns:
        st.error("No 'keyword' column found in dataframe")
        return df
    
    # Load spaCy model if possible
    spacy_nlp = load_spacy_model(language)
    
    # Determine preprocessing method
    use_advanced = (spacy_nlp is not None) or (textblob_available and textblob_resources_available)
    
    # Log the preprocessing method being used
    if spacy_nlp is not None:
        st.success(f"Using advanced preprocessing with spaCy for {language}")
    elif textblob_available and textblob_resources_available:
        st.success("Using advanced preprocessing with TextBlob")
    else:
        st.info("Using basic preprocessing with NLTK")
    
    # Process keywords with progress indicator
    total = len(df)
    progress_bar = st.progress(0)
    
    # Create a new column for processed keywords
    processed_keywords = []
    
    for i, keyword in enumerate(df['keyword']):
        try:
            if use_advanced:
                processed = advanced_preprocessing(keyword, spacy_nlp)
            else:
                processed = basic_preprocessing(keyword)
            
            processed_keywords.append(processed)
        except Exception as e:
            # Fallback to simple processing on error
            st.warning(f"Error processing keyword '{keyword}': {str(e)}")
            processed_keywords.append(str(keyword).lower())
        
        # Update progress bar every 20 items to avoid excessive rerenders
        if i % 20 == 0 or i == total - 1:
            progress_bar.progress(min((i + 1) / total, 1.0))
    
    # Add processed keywords to dataframe
    df['keyword_processed'] = processed_keywords
    
    # Clear progress bar
    progress_bar.empty()
    
    return df

def extract_keyword_features(keyword: str) -> Dict[str, Any]:
    """
    Extract useful features from a keyword for advanced analysis.
    
    Args:
        keyword: The keyword to analyze
        
    Returns:
        Dictionary of features
    """
    if not isinstance(keyword, str):
        return {}
    
    # Basic features
    features = {
        "length": len(keyword),
        "word_count": len(keyword.split()),
        "has_numbers": any(c.isdigit() for c in keyword),
        "has_special_chars": any(not c.isalnum() and not c.isspace() for c in keyword),
        "is_question": keyword.strip().endswith('?') or keyword.lower().startswith(('what', 'how', 'why', 'when', 'where', 'who', 'which'))
    }
    
    return features

# Print available resources - useful for debugging
print("NLP Resources Available:")
print(f"NLTK: {nltk_resources_available}")
print(f"spaCy base: {spacy_base_available}")
print(f"spaCy models: {spacy_model_available}")
print(f"TextBlob: {textblob_available}")
print(f"TextBlob resources: {textblob_resources_available}")
