# Intelligent Review Analytics Platform 
**Status:** `Active Development`

## 1. Business Problem & Solution
**Challenge**: E-commerce platforms spend 8+ hours daily on manual review analysis, leading to delayed product issue detection and inefficient resource allocation.

**Solution**: This project aims to build an end-to-end sentiment analysis platform that automates review intelligence. This document tracks the development process, starting with the establishment of a robust data pipeline and a baseline machine learning model.

---

## 2. Project Status & Strategic Roadmap

### Roadmap
- [X] Phase 1: Baseline model (4K reviews)
- [ ] Phase 2: Production scale (140M reviews)
- [ ] Phase 3: Advanced models
- [ ] Phase 4: Cloud deployment

Phase 1: Completed
Phase 2: In Development

This project is being developed in a phased approach to ensure a robust and scalable final product.

---

## 3. Technology Stack 

-   **Language:** Python 3.8+
-   **Core Libraries:** Pandas, NumPy, Scikit-learn, NLTK
-   **Data Analysis:** Jupyter Notebook, Matplotlib, Seaborn
-   **Version Control:** Git, GitHub

---

## 4. Baseline Model Performance

The initial baseline was established using a Logistic Regression model with TF-IDF vectorization. The goal of this phase was to create a reliable benchmark against which all future, more complex models will be compared.

| Algorithm           | Accuracy         | F1-Score       |
| ------------------- | ---------------- | -------------- |
| Logistic Regression | `82%` | `0.77` |

---

## 5. Getting Started

### **Prerequisites**
- Python 3.8 or higher
- `pip` for package installation

### **Installation**
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/pr-rithwik/intelligent-review-analytics.git
    cd intelligent-review-analytics
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Explore the analysis**:
    -   Navigate to the `notebooks/` directory.
    -   Run the `phase1_data_intelligence.ipynb` and `phase2_ml_development.ipynb` notebooks to see the full analysis and baseline model training process.
