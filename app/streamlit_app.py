"""
Sentiment Analysis Web Application
Interactive Streamlit interface for real-time sentiment prediction.

Usage:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import numpy as np

from src import config
from src.data.preprocess import clean_text
from src.features.tfidf import transform_tfidf
from src.utils.persist import load_artifacts


# Configure page
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💭",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model_artifacts():
    """
    Load trained model and vectorizer (cached for performance).
    
    Returns:
        Tuple of (model, vectorizer)
    """
    try:
        model, vectorizer = load_artifacts(
            config.MODEL_FILE,
            config.VECTORIZER_FILE
        )
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.info("🔧 Please run: `python scripts/train_phase1.py` first")
        st.stop()


def predict_sentiment(text: str, model, vectorizer) -> dict:
    """
    Predict sentiment for input text.
    
    Args:
        text: Raw review text
        model: Trained model
        vectorizer: Fitted vectorizer
        
    Returns:
        Dictionary with prediction details
    """
    # Clean text
    cleaned = clean_text(
        text,
        lowercase=config.LOWERCASE,
        remove_stopwords=config.REMOVE_STOPWORDS
    )
    
    # Check if text is empty after cleaning
    if not cleaned or len(cleaned.strip()) == 0:
        return {
            'error': 'Text is empty after preprocessing. Please enter more substantive text.'
        }
    
    # Vectorize
    X = transform_tfidf([cleaned], vectorizer)
    
    # Predict
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    # Get confidence (max probability)
    confidence = np.max(probabilities)
    
    return {
        'prediction': int(prediction),
        'sentiment': 'Positive' if prediction == 1 else 'Negative',
        'confidence': confidence,
        'probabilities': {
            'Negative': probabilities[0],
            'Positive': probabilities[1]
        }
    }


def main():
    """Main application."""
    
    # Load model artifacts
    model, vectorizer = load_model_artifacts()
    
    # ========================================================================
    # HEADER
    # ========================================================================
    st.title("💭 Sentiment Analysis Platform")
    st.markdown("""
    Analyze the sentiment of product reviews using machine learning.
    Enter a review below and get instant sentiment predictions!
    """)
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **Model:** Logistic Regression  
        **Features:** TF-IDF (5000 terms)  
        **Accuracy:** ~87%  
        **Training Data:** Amazon Reviews
        """)
        
        st.markdown("---")
        
        st.header("📊 Model Info")
        vocab_size = len(vectorizer.vocabulary_)
        st.metric("Vocabulary Size", f"{vocab_size:,}")
        
        st.markdown("---")
        
        st.header("🎯 How It Works")
        st.markdown("""
        1. **Input:** Enter product review text
        2. **Preprocessing:** Clean and normalize text
        3. **Vectorization:** Convert to TF-IDF features
        4. **Prediction:** Classify sentiment
        5. **Output:** Sentiment + confidence score
        """)
    
    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    
    # Text input
    st.subheader("📝 Enter Your Review")
    
    # Create tabs for manual input and examples
    tab1, tab2 = st.tabs(["✍️ Write Review", "📋 Try Examples"])
    
    with tab1:
        user_input = st.text_area(
            "Type or paste a product review:",
            height=150,
            placeholder="e.g., This product exceeded my expectations! Great quality and fast shipping.",
            help="Enter any product review text to analyze its sentiment"
        )
    
    with tab2:
        st.markdown("**Click an example to analyze:**")
        
        examples = {
            "Positive - Excellent Product": "This is the best coffee I've ever tasted! Rich flavor, smooth finish, and arrives fresh. Highly recommend to any coffee lover. Will definitely buy again!",
            "Negative - Poor Quality": "Terrible product. Arrived damaged and tastes awful. Complete waste of money. Very disappointed and would not recommend to anyone.",
            "Positive - Great Value": "Amazing value for the price. Works exactly as described and shipping was super fast. My dog loves these treats!",
            "Negative - Not as Described": "Not what I expected at all. The description was misleading and the quality is poor. Returning this item.",
            "Positive - Exceeded Expectations": "Wow! This exceeded all my expectations. The quality is outstanding and it arrived earlier than expected. Five stars!"
        }
        
        selected_example = st.radio(
            "Choose an example:",
            options=list(examples.keys()),
            label_visibility="collapsed"
        )
        
        if st.button("📋 Use This Example", type="primary"):
            user_input = examples[selected_example]
            st.session_state['example_input'] = user_input
            st.rerun()
    
    # Use example input if available
    if 'example_input' in st.session_state:
        user_input = st.session_state['example_input']
        del st.session_state['example_input']
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    
    # ========================================================================
    # PREDICTION & RESULTS
    # ========================================================================
    
    if analyze_button and user_input:
        with st.spinner("Analyzing sentiment..."):
            result = predict_sentiment(user_input, model, vectorizer)
        
        # Check for errors
        if 'error' in result:
            st.error(result['error'])
            st.stop()
        
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        # Display sentiment with appropriate styling
        sentiment = result['sentiment']
        confidence = result['confidence']
        
        if sentiment == 'Positive':
            st.success(f"### ✅ {sentiment} Sentiment")
        else:
            st.error(f"### ❌ {sentiment} Sentiment")
        
        # Confidence metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Confidence",
                f"{confidence:.1%}",
                delta=None
            )
        
        with col2:
            st.metric(
                "Positive Probability",
                f"{result['probabilities']['Positive']:.1%}"
            )
        
        with col3:
            st.metric(
                "Negative Probability",
                f"{result['probabilities']['Negative']:.1%}"
            )
        
        # Progress bar visualization
        st.markdown("**Sentiment Distribution:**")
        st.progress(result['probabilities']['Positive'])
        
        # Detailed breakdown
        with st.expander("🔍 View Detailed Analysis"):
            st.markdown("**Raw Probabilities:**")
            st.json(result['probabilities'])
            
            st.markdown("**Confidence Interpretation:**")
            if confidence >= 0.9:
                st.info("🎯 **Very High Confidence** - The model is very certain about this prediction.")
            elif confidence >= 0.75:
                st.info("✅ **High Confidence** - The model is quite confident in this prediction.")
            elif confidence >= 0.6:
                st.warning("⚠️ **Moderate Confidence** - The model has some uncertainty.")
            else:
                st.warning("❓ **Low Confidence** - The sentiment is unclear or mixed.")
    
    elif analyze_button:
        st.warning("⚠️ Please enter some text to analyze.")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Built with Streamlit • Powered by Scikit-learn • TF-IDF Features</p>
        <p>Phase 1: Baseline Model | Trained on Amazon Product Reviews</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()