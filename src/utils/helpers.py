"""
Utility Helper Functions for Intelligent Review Analytics Platform

Common utility functions used across the project including logging setup,
file operations, data validation, and general helper utilities.

"""

import os
import sys
import logging
import json
import pickle
import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration for the project.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (str, optional): Path to log file
        log_format (str, optional): Custom log format
    
    Returns:
        logging.Logger: Configured logger instance
    """
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    logger = logging.getLogger('intelligent_review_analytics')
    logger.info(f"Logging initialized at level {log_level}")
    
    return logger

def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save dictionary as JSON file with proper formatting.
    
    Args:
        data (Dict[str, Any]): Data to save
        filepath (str): Path to save the JSON file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load JSON file as dictionary.
    
    Args:
        filepath (str): Path to JSON file
    
    Returns:
        Dict[str, Any]: Loaded data
    
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_pickle(obj: Any, filepath: str) -> None:
    """
    Save object as pickle file.
    
    Args:
        obj (Any): Object to save
        filepath (str): Path to save the pickle file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filepath: str) -> Any:
    """
    Load object from pickle file.
    
    Args:
        filepath (str): Path to pickle file
    
    Returns:
        Any: Loaded object
    
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Pickle file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_model(model: Any, filepath: str, use_joblib: bool = True) -> None:
    """
    Save machine learning model to file.
    
    Args:
        model (Any): Trained model object
        filepath (str): Path to save the model
        use_joblib (bool): Use joblib instead of pickle for better performance
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if use_joblib:
        joblib.dump(model, filepath)
    else:
        save_pickle(model, filepath)

def load_model(filepath: str, use_joblib: bool = True) -> Any:
    """
    Load machine learning model from file.
    
    Args:
        filepath (str): Path to model file
        use_joblib (bool): Use joblib instead of pickle for better performance
    
    Returns:
        Any: Loaded model object
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    if use_joblib:
        return joblib.load(filepath)
    else:
        return load_pickle(filepath)

def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory (Union[str, Path]): Directory path
    
    Returns:
        Path: Directory path as Path object
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def get_file_size(filepath: Union[str, Path]) -> str:
    """
    Get human-readable file size.
    
    Args:
        filepath (Union[str, Path]): Path to file
    
    Returns:
        str: Human-readable file size
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return "File not found"
    
    size_bytes = filepath.stat().st_size
    
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"

def calculate_processing_time(start_time: datetime) -> Tuple[float, str]:
    """
    Calculate processing time from start time.
    
    Args:
        start_time (datetime): Start time
    
    Returns:
        Tuple[float, str]: Processing time in seconds and human-readable format
    """
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    if processing_time < 60:
        readable_time = f"{processing_time:.2f} seconds"
    elif processing_time < 3600:
        readable_time = f"{processing_time/60:.2f} minutes"
    else:
        readable_time = f"{processing_time/3600:.2f} hours"
    
    return processing_time, readable_time

def validate_dataframe_structure(
    df: pd.DataFrame, 
    required_columns: List[str],
    optional_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (List[str]): Required column names
        optional_columns (List[str], optional): Optional column names
    
    Returns:
        Dict[str, Any]: Validation report
    """
    validation_report = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'shape': df.shape,
        'columns': list(df.columns),
        'missing_values': {}
    }
    
    # Check required columns
    missing_required = set(required_columns) - set(df.columns)
    if missing_required:
        validation_report['is_valid'] = False
        validation_report['errors'].append(f"Missing required columns: {missing_required}")
    
    # Check for empty DataFrame
    if df.empty:
        validation_report['is_valid'] = False
        validation_report['errors'].append("DataFrame is empty")
        return validation_report
    
    # Check missing values
    for column in df.columns:
        missing_count = df[column].isnull().sum()
        missing_percentage = (missing_count / len(df)) * 100
        
        validation_report['missing_values'][column] = {
            'count': int(missing_count),
            'percentage': round(missing_percentage, 2)
        }
        
        if missing_percentage > 50:
            validation_report['warnings'].append(
                f"Column '{column}' has {missing_percentage:.1f}% missing values"
            )
    
    # Check data types
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    text_columns = df.select_dtypes(include=['object']).columns
    
    validation_report['column_types'] = {
        'numeric': list(numeric_columns),
        'text': list(text_columns)
    }
    
    return validation_report

def create_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create comprehensive summary statistics for DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
    
    Returns:
        Dict[str, Any]: Summary statistics
    """
    summary = {
        'shape': df.shape,
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        'column_info': {},
        'data_types': df.dtypes.to_dict(),
        'missing_summary': {
            'total_missing': int(df.isnull().sum().sum()),
            'columns_with_missing': int((df.isnull().sum() > 0).sum()),
            'percentage_missing': round((df.isnull().sum().sum() / df.size) * 100, 2)
        }
    }
    
    # Column-specific information
    for column in df.columns:
        column_info = {
            'dtype': str(df[column].dtype),
            'unique_values': int(df[column].nunique()),
            'missing_count': int(df[column].isnull().sum()),
            'missing_percentage': round((df[column].isnull().sum() / len(df)) * 100, 2)
        }
        
        if df[column].dtype in ['int64', 'float64']:
            column_info.update({
                'min': float(df[column].min()) if pd.notna(df[column].min()) else None,
                'max': float(df[column].max()) if pd.notna(df[column].max()) else None,
                'mean': float(df[column].mean()) if pd.notna(df[column].mean()) else None,
                'std': float(df[column].std()) if pd.notna(df[column].std()) else None
            })
        
        summary['column_info'][column] = column_info
    
    return summary

def format_large_number(number: Union[int, float]) -> str:
    """
    Format large numbers with appropriate suffixes.
    
    Args:
        number (Union[int, float]): Number to format
    
    Returns:
        str: Formatted number with suffix
    """
    if abs(number) >= 1e9:
        return f"{number/1e9:.1f}B"
    elif abs(number) >= 1e6:
        return f"{number/1e6:.1f}M"
    elif abs(number) >= 1e3:
        return f"{number/1e3:.1f}K"
    else:
        return str(int(number)) if isinstance(number, float) and number.is_integer() else f"{number:.1f}"

def generate_timestamp(format_string: str = "%Y%m%d_%H%M%S") -> str:
    """
    Generate timestamp string for file naming.
    
    Args:
        format_string (str): Datetime format string
    
    Returns:
        str: Formatted timestamp
    """
    return datetime.now().strftime(format_string)

def create_progress_tracker() -> Dict[str, Any]:
    """
    Create a progress tracking dictionary.
    
    Returns:
        Dict[str, Any]: Progress tracker with timestamp and status
    """
    return {
        'created_at': datetime.now().isoformat(),
        'last_updated': datetime.now().isoformat(),
        'status': 'initialized',
        'progress_percentage': 0.0,
        'completed_steps': [],
        'current_step': None,
        'errors': [],
        'warnings': []
    }

def update_progress_tracker(
    tracker: Dict[str, Any],
    step: str,
    status: str = 'completed',
    progress_percentage: Optional[float] = None
) -> Dict[str, Any]:
    """
    Update progress tracker with new step completion.
    
    Args:
        tracker (Dict[str, Any]): Progress tracker dictionary
        step (str): Completed step name
        status (str): Step status
        progress_percentage (float, optional): Overall progress percentage
    
    Returns:
        Dict[str, Any]: Updated progress tracker
    """
    tracker['last_updated'] = datetime.now().isoformat()
    tracker['completed_steps'].append({
        'step': step,
        'status': status,
        'completed_at': datetime.now().isoformat()
    })
    
    if progress_percentage is not None:
        tracker['progress_percentage'] = progress_percentage
    
    tracker['current_step'] = step
    
    return tracker