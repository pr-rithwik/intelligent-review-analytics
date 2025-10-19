"""
Configuration Management Module
Central hub for all project constants, paths, and hyperparameters.
"""
from pathlib import Path
from enum import Enum


class Phase(Enum):
    """Project development phases"""
    PHASE1 = "phase1"
    PHASE2 = "phase2"


ROOT_DIR = Path(__file__).parent.parent
CURRENT_PHASE = Phase.PHASE1

DATA_DIR = ROOT_DIR / "data" / CURRENT_PHASE.value
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT_DIR / "models" / CURRENT_PHASE.value

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# data file paths
RAW_DATA_FILE = RAW_DATA_DIR / "reviews_train.tsv"
TRAIN_FILE = PROCESSED_DATA_DIR / "train_clean.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test_clean.csv"

# Model artifacts
MODEL_FILE = MODELS_DIR / "baseline_logreg.pkl"
VECTORIZER_FILE = MODELS_DIR / "tfidf_vectorizer.pkl"

# Column names for raw data
SENTIMENT_COLUMN = 0 # Column index for sentiment labels
TEXT_COLUMN = 4 # Column index for review text

# Train-test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

LOWERCASE = True  # Convert text to lowercase
REMOVE_STOPWORDS = False # Keep stopwords for baseline
REMOVE_NUMBERS = False # Keep numbers in text
REMOVE_SPECIAL_CHARS = True # Remove special characters

# TF-IDF parameters
MAX_FEATURES = 5000   # Maximum vocabulary size
MIN_DF = 2          # Minimum document frequency
MAX_DF = 0.95      # Maximum document frequency (as proportion)

# Logistic Regression settings
MAX_ITER = 1000      # Maximum iterations for convergence
SOLVER = 'lbfgs'   # Optimization algorithm
C = 1.0           # Inverse regularization strength

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'