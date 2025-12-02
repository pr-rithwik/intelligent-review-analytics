"""
Phase 1 Training Pipeline
End-to-end script to train the baseline sentiment analysis model.

Usage:
    python scripts/train_phase1.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import pandas as pd
from sklearn.model_selection import train_test_split

# Import project modules
from src import config
from src.data import loader, preprocess
from src.features import tfidf
from src.models import baseline, evaluate
from src.utils import persist

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline."""
    
    logger.info("="*60)
    logger.info("PHASE 1 TRAINING PIPELINE - START")
    logger.info("="*60)
    
    # ========================================================================
    # STEP 1: Load Raw Data
    # ========================================================================
    logger.info("\n[STEP 1] Loading raw data...")
    
    try:
        df_raw = loader.load_raw_data(config.RAW_DATA_FILE)
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        logger.error(f"Please place 'reviews_train.tsv' in: {config.RAW_DATA_DIR}")
        sys.exit(1)
    
    # Extract text and labels from specific columns
    logger.info("Extracting text and sentiment labels...")
    df_raw.columns = df_raw.columns.astype(str)  # Ensure string column names
    
    # Get text (column 4) and sentiment (column 0)
    text_data = df_raw.iloc[:, config.TEXT_COLUMN]
    sentiment_labels = df_raw.iloc[:, config.SENTIMENT_COLUMN]
    
    # Create working dataframe
    df = pd.DataFrame({
        'text': text_data,
        'sentiment': sentiment_labels
    })
    
    # Remove neutral sentiment (0) - keep only positive (1) and negative (-1)
    df = df[df['sentiment'] != 0].copy()
    logger.info(f"Filtered to binary classification: {len(df)} samples")
    
    # Convert labels to binary (1 for positive, 0 for negative)
    df['sentiment'] = (df['sentiment'] == 1).astype(int)
    
    logger.info(f"Class distribution:\n{df['sentiment'].value_counts()}")
    
    # ========================================================================
    # STEP 2: Preprocess Text
    # ========================================================================
    logger.info("\n[STEP 2] Preprocessing text...")
    
    df_clean = preprocess.preprocess_dataframe(
        df,
        text_column='text',
        lowercase=config.LOWERCASE,
        remove_stopwords=config.REMOVE_STOPWORDS,
        show_progress=True
    )
    
    logger.info(f"Preprocessing complete. {len(df_clean)} samples ready.")
    
    # ========================================================================
    # STEP 3: Train-Test Split
    # ========================================================================
    logger.info("\n[STEP 3] Splitting into train and test sets...")
    
    X = df_clean['cleaned_text']
    y = df_clean['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y  # Maintain class distribution
    )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Train class distribution:\n{y_train.value_counts()}")
    
    # ========================================================================
    # STEP 4: Feature Engineering (TF-IDF)
    # ========================================================================
    logger.info("\n[STEP 4] Creating TF-IDF features...")
    
    # Create vectorizer
    vectorizer = tfidf.create_tfidf_vectorizer(
        max_features=config.MAX_FEATURES,
        min_df=config.MIN_DF,
        max_df=config.MAX_DF
    )
    
    # Fit and transform training data
    X_train_tfidf, vectorizer_fitted = tfidf.fit_transform_tfidf(X_train, vectorizer)
    
    # Transform test data (using fitted vectorizer)
    X_test_tfidf = tfidf.transform_tfidf(X_test, vectorizer_fitted)
    
    logger.info(f"Training features shape: {X_train_tfidf.shape}")
    logger.info(f"Test features shape: {X_test_tfidf.shape}")
    
    # ========================================================================
    # STEP 5: Train Baseline Model
    # ========================================================================
    logger.info("\n[STEP 5] Training Logistic Regression model...")
    
    # Create model
    model = baseline.create_baseline_model(
        random_state=config.RANDOM_STATE,
        max_iter=config.MAX_ITER,
        solver=config.SOLVER,
        C=config.C
    )
    
    # Train model
    model_trained = baseline.train_model(X_train_tfidf, y_train, model)
    
    logger.info("Model training complete!")
    
    # ========================================================================
    # STEP 6: Evaluate Model
    # ========================================================================
    logger.info("\n[STEP 6] Evaluating model on test set...")
    
    # Make predictions
    y_pred, y_proba = baseline.predict(model_trained, X_test_tfidf)
    
    # Calculate metrics
    metrics = evaluate.calculate_metrics(y_test, y_pred, y_proba)
    
    # Print detailed evaluation
    evaluate.print_metrics_summary(metrics)
    evaluate.print_classification_report(
        y_test, y_pred,
        target_names=['Negative', 'Positive']
    )
    
    # Create visualizations
    logger.info("\nGenerating visualizations...")
    evaluate.plot_confusion_matrix(
        y_test, y_pred,
        class_names=['Negative', 'Positive'],
        save_path=config.MODELS_DIR / 'confusion_matrix.png'
    )
    
    # Plot ROC curve if binary classification
    if y_proba.shape[1] == 2:
        evaluate.plot_roc_curve(
            y_test, y_proba[:, 1],
            save_path=config.MODELS_DIR / 'roc_curve.png'
        )
    
    # ========================================================================
    # STEP 7: Save Model and Vectorizer
    # ========================================================================
    logger.info("\n[STEP 7] Saving model artifacts...")
    
    persist.save_artifacts(
        model_trained,
        vectorizer_fitted,
        config.MODEL_FILE,
        config.VECTORIZER_FILE
    )
    
    logger.info(f"Model saved to: {config.MODEL_FILE}")
    logger.info(f"Vectorizer saved to: {config.VECTORIZER_FILE}")
    
    # ========================================================================
    # STEP 8: Display Top Features
    # ========================================================================
    logger.info("\n[STEP 8] Analyzing top influential features...")
    
    feature_names = tfidf.get_feature_names(vectorizer_fitted)
    top_coeffs = baseline.get_top_coefficients(
        model_trained,
        feature_names,
        top_n=15
    )
    
    print("\n" + "="*60)
    print("TOP 15 POSITIVE SENTIMENT INDICATORS")
    print("="*60)
    for word, coef in top_coeffs['positive']:
        print(f"{word:.<40} {coef:.4f}")
    
    print("\n" + "="*60)
    print("TOP 15 NEGATIVE SENTIMENT INDICATORS")
    print("="*60)
    for word, coef in top_coeffs['negative']:
        print(f"{word:.<40} {coef:.4f}")
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("PHASE 1 TRAINING PIPELINE - COMPLETE ✓")
    logger.info("="*60)
    logger.info(f"\nModel accuracy: {metrics['accuracy']:.2%}")
    logger.info(f"F1 score: {metrics['f1_score']:.4f}")
    logger.info(f"\nNext steps:")
    logger.info(f"1. Review metrics and visualizations in: {config.MODELS_DIR}")
    logger.info(f"2. Run the Streamlit app: streamlit run app/streamlit_app.py")
    logger.info(f"3. Test predictions with your own reviews!")


if __name__ == "__main__":
    main()