# src/data/preprocess.py

"""
Text Preprocessing Module
Clean and normalize review text for ML models.
"""

import re
import pandas as pd
from typing import Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

# Optional: NLTK for stopwords (only if needed)
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
    NLTK_AVAILABLE = True
except (ImportError, LookupError):
    STOPWORDS = set()
    NLTK_AVAILABLE = False
    logger.warning("NLTK stopwords not available. Install with: nltk.download('stopwords')")


def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags from text (e.g., <br />, <p>, etc.).
    
    Args:
        text: Raw text with potential HTML tags
        
    Returns:
        Text with HTML tags removed
    """
    if not isinstance(text, str):
        return text
    
    # Remove HTML tags using regex
    clean_text = re.sub(r'<[^>]+>', '', text)
    return clean_text


def remove_special_characters(text: str, keep_spaces: bool = True) -> str:
    """
    Remove special characters, keeping only letters and optionally spaces.
    
    Args:
        text: Text to clean
        keep_spaces: Whether to keep whitespace characters
        
    Returns:
        Text with special characters removed
    """
    if not isinstance(text, str):
        return text
    
    if keep_spaces:
        # Keep letters and spaces
        pattern = r'[^a-zA-Z\s]'
    else:
        # Keep only letters
        pattern = r'[^a-zA-Z]'
    
    clean_text = re.sub(pattern, ' ', text)
    return clean_text


def normalize_whitespace(text: str) -> str:
    """
    Remove extra whitespace and trim.
    
    Args:
        text: Text with potential extra whitespace
        
    Returns:
        Text with normalized whitespace
    """
    if not isinstance(text, str):
        return text
    
    # Replace multiple spaces with single space
    clean_text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    clean_text = clean_text.strip()
    
    return clean_text


def remove_stopwords_from_text(text: str) -> str:
    """
    Remove common stopwords from text.
    
    Args:
        text: Text to process
        
    Returns:
        Text with stopwords removed
    """
    if not isinstance(text, str) or not NLTK_AVAILABLE:
        return text
    
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in STOPWORDS]
    return ' '.join(filtered_words)


def clean_text(text: str, 
               lowercase: bool = True, 
               remove_stopwords: bool = False) -> str:
    """
    Main cleaning pipeline - orchestrates all cleaning steps.
    
    Args:
        text: Raw review text
        lowercase: Convert to lowercase
        remove_stopwords: Remove common stopwords
        
    Returns:
        Cleaned and normalized text
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Step 1: Remove HTML tags
    text = remove_html_tags(text)
    
    # Step 2: Convert to lowercase
    if lowercase:
        text = text.lower()
    
    # Step 3: Remove special characters
    text = remove_special_characters(text, keep_spaces=True)
    
    # Step 4: Remove stopwords (optional)
    if remove_stopwords:
        text = remove_stopwords_from_text(text)
    
    # Step 5: Normalize whitespace
    text = normalize_whitespace(text)
    
    return text


def preprocess_dataframe(df: pd.DataFrame, 
                        text_column: str,
                        lowercase: bool = True,
                        remove_stopwords: bool = False,
                        show_progress: bool = True) -> pd.DataFrame:
    """
    Apply cleaning to entire DataFrame.
    
    Args:
        df: DataFrame with review data
        text_column: Name of column containing text
        lowercase: Convert to lowercase
        remove_stopwords: Remove stopwords
        show_progress: Show progress bar
        
    Returns:
        DataFrame with added 'cleaned_text' column
    """
    logger.info(f"Preprocessing {len(df)} texts from column '{text_column}'")
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Apply cleaning with progress bar
    if show_progress:
        tqdm.pandas(desc="Cleaning text")
        df_clean['cleaned_text'] = df_clean[text_column].progress_apply(
            lambda x: clean_text(x, lowercase, remove_stopwords)
        )
    else:
        df_clean['cleaned_text'] = df_clean[text_column].apply(
            lambda x: clean_text(x, lowercase, remove_stopwords)
        )
    
    # Remove empty texts
    original_len = len(df_clean)
    df_clean = df_clean[df_clean['cleaned_text'].str.len() > 0]
    removed = original_len - len(df_clean)
    
    if removed > 0:
        logger.warning(f"Removed {removed} empty texts after cleaning")
    
    logger.info(f"Preprocessing complete. {len(df_clean)} texts remaining")
    
    return df_clean