# Intelligent Review Analytics Platform 
**Status:** `Active Development`

## 1. Business Problem & Solution
**Challenge**: E-commerce platforms spend 8+ hours daily on manual review analysis, leading to delayed product issue detection and inefficient resource allocation.

**Solution**: This project aims to build an end-to-end sentiment analysis platform that automates review intelligence. This document tracks the development process, starting with the establishment of a robust data pipeline and a baseline machine learning model.

---

## 2. Project Status & Strategic Roadmap

This project is being developed in a phased approach to ensure a robust and scalable final product.

### **Current Status:**
✔️ **Data Pipeline:** Successfully built a data processing pipeline for 4000+ Amazon product reviews.

✔️ **Baseline Model:** Implemented and evaluated a Logistic Regression model to establish a baseline performance metric.

✔️ **Core Insights:** Conducted initial exploratory data analysis (EDA) to extract foundational business insights.

✔️ **Framework:** The core project structure is in place, ready for the implementation of more advanced models.

### **Development Roadmap**

-   **Phase 2: Advanced Model Implementation (In Progress)**
    -   [ ] Implement and evaluate `SVM` and `Random Forest` models.
    -   [ ] Implement and evaluate production-grade `XGBoost` model.
    -   [ ] Integrate `BERT` via the Transformers library for state-of-the-art accuracy.

-   **Phase 3: Application & Deployment**
    -   [ ] Develop an interactive `Streamlit` web application for business users.
    -   [ ] Build data visualizations with `Plotly` to communicate insights.
    -   [ ] Deploy the final application to the cloud for live access.

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
| Logistic Regression | `[YOUR_ACCURACY]%` | `[YOUR_F1_SCORE]` |

---

## 5. Repository Structure

```
intelligent-review-analytics/
├── data/ # Raw and processed datasets
├── notebooks/ # EDA and model development notebooks
├── src/ # Python source code modules (e.g., for data cleaning)
└── README.md # Project documentation
```


---

## 6. Getting Started

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
