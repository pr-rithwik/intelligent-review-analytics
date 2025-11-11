"""
Data Loading Module
Handles loading data from various formats with proper error handling.
"""

import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_raw_data(filepath: Path) -> pd.DataFrame:
    """
    Load raw TSV data from file.
    
    Args:
        filepath: Path to TSV file
        
    Returns:
        DataFrame with raw review data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    logger.info(f"Loading raw data from: {filepath}")
    
    # Load TSV with proper encoding
    df = pd.read_csv(
        filepath,
        sep='\t',
        encoding='utf-8',
        header=None,  # No header row in this dataset
        on_bad_lines='skip'  # Skip malformed lines
    )
    
    # Basic validation
    if df.empty:
        raise ValueError("Loaded dataframe is empty")
    
    # Log data info
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        logger.warning(f"Found {missing_count} missing values in dataset")
    
    return df


def load_processed_data(train_path: Path, test_path: Path) -> tuple:
    """
    Load pre-processed train/test data.
    
    Args:
        train_path: Path to processed training CSV
        test_path: Path to processed test CSV
        
    Returns:
        Tuple of (train_df, test_df)
        
    Raises:
        FileNotFoundError: If either file doesn't exist
    """
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")
    
    logger.info("Loading processed train and test data")
    
    # Load CSVs
    train_df = pd.read_csv(train_path, encoding='utf-8')
    test_df = pd.read_csv(test_path, encoding='utf-8')
    
    logger.info(f"Train set: {len(train_df)} samples")
    logger.info(f"Test set: {len(test_df)} samples")
    
    return train_df, test_df


def extract_features_labels(df: pd.DataFrame, 
                           text_column: str, 
                           label_column: str) -> tuple:
    """
    Extract feature and label columns from dataframe.
    
    Args:
        df: DataFrame containing data
        text_column: Name of text feature column
        label_column: Name of label column
        
    Returns:
        Tuple of (X, y) where X is text series and y is labels
    """
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in dataframe")
    
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataframe")
    
    X = df[text_column]
    y = df[label_column]
    
    # Remove any NaN values
    mask = X.notna() & y.notna()
    X = X[mask]
    y = y[mask]
    
    logger.info(f"Extracted {len(X)} samples with features and labels")
    
    return X, y