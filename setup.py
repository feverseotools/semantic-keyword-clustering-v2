# setup.py
"""
Setup script to install all required resources for the keyword clustering application.
Run this once before using the application to ensure all models and data are available.
"""

import subprocess
import sys

def install_package(package):
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def install_spacy_model(model):
    print(f"Installing spaCy model {model}...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", model])

def download_nltk_data(resource):
    print(f"Downloading NLTK resource {resource}...")
    import nltk
    nltk.download(resource)

def download_textblob_corpora():
    print("Downloading TextBlob corpora...")
    import textblob.download_corpora
    textblob.download_corpora.download_all()

def main():
    # Install key packages if not already installed
    try:
        import nltk
    except ImportError:
        install_package("nltk")
        import nltk
    
    try:
        import textblob
    except ImportError:
        install_package("textblob")
        import textblob
    
    try:
        import spacy
    except ImportError:
        install_package("spacy")
        import spacy
    
    # Download NLTK resources
    download_nltk_data("punkt")
    download_nltk_data("stopwords")
    download_nltk_data("wordnet")
    
    # Download TextBlob corpora
    download_textblob_corpora()
    
    # Install spaCy model
    try:
        install_spacy_model("en_core_web_sm")
    except:
        print("Warning: Could not install spaCy model. Some features may be limited.")
    
    print("\nSetup complete! You can now run the application with 'streamlit run app.py'")

if __name__ == "__main__":
    main()
