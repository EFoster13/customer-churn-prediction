# Customer Churn Prediction & Retention Strategy

A comprehensive machine learning project that predicts customer churn and provides actionable business insights for retention strategies in the telecom industry.

---

## Project Overview

**Business Problem:** Customer acquisition costs are 5-25x higher than retention costs. This project aims to:
- Predict which customers are likely to churn
- Identify key drivers of customer churn
- Recommend data-driven retention strategies
- Quantify potential ROI of retention initiatives

**Key Result:** Achieved **84.6% ROC-AUC** with actionable insights projected to save **$467K-$700K annually**

---

## Dataset

- **Source:** [IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 21 features
- **Target:** Binary classification (Churn: Yes/No)
- **Features:** Contract type, tenure, services, monthly charges, payment method, etc.

---

## Key Findings

### 1. Overall Churn Rate: 26.5%
Approximately 1 in 4 customers leave, representing a significant business problem.

### 2. Contract Type is the Primary Driver
| Contract Type | Churn Rate |
|---------------|------------|
| Month-to-month | **42.7%** |
| One year | 11.3% |
| Two year | **2.8%** |

**Insight:** Customers on month-to-month contracts are 15x more likely to churn than those on two-year contracts.

### 3. Fiber Optic Service - Critical Issue
- **Churn Rate:** 41.9%
- **Average Monthly Cost:** $91
- **Problem:** Premium pricing without commitment = highest churn segment

### 4. New Customer Risk
- Most churn occurs in the first 6 months
- Average tenure of churners: ~18 months vs. ~38 months for loyal customers

---

## Model Performance Comparison

Evaluated three models on stratified train-test split (80/20):

| Model | ROC-AUC | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| Logistic Regression | 0.8461 | 0.67 | 0.53 | 0.59 |
| XGBoost | 0.8403 | 0.65 | 0.53 | 0.59 |
| Random Forest | 0.8412 | 0.65 | 0.50 | 0.56 |

**Selected Model:** Logistic Regression
- Highest ROC-AUC score
- Most interpretable for business stakeholders
- Fastest inference time

**Evaluation Focus:** Prioritized Recall over Precision - missing churners is more costly than false alarms.

---

## Business Recommendations

### 1. Contract Incentive Program (Highest Priority)
**Problem:** Month-to-month contracts have 42.7% churn rate

**Solution:**
- Offer 15-20% discount for switching to 1-year or 2-year contracts
- Target new customers within first 90 days
- **Projected Impact:** Reduce M2M churn from 42.7% → 25%

---

### 2️. Fiber Loyalty Bundle
**Problem:** Fiber optic customers churn at 41.9% despite paying $91/month

**Solution:**
- Launch tiered fiber pricing (Standard/Premium/Ultimate)
- Offer 20% discount for 12-month fiber commitment
- **Projected Impact:** Reduce fiber churn from 41.9% → 25%

---

### 3️. First 90 Days Onboarding Program
**Problem:** New customers (tenure < 6 months) have highest churn risk

**Solution:**
- Proactive check-ins at 30, 60, and 90 days
- Dedicated support line for new customers
- Early contract upgrade incentives
- **Projected Impact:** Reduce new customer churn by 15-20%

---

## Estimated Business Impact

**Assumptions:**
- Average Customer Lifetime Value: $2,500
- Current annual churners: ~1,870 customers
- Target churn reduction: 10-15%

**Projected Annual Savings:**
- **Conservative (10% reduction):** 187 customers × $2,500 = **$467,500**
- **Optimistic (15% reduction):** 280 customers × $2,500 = **$700,000**

**ROI of retention campaigns:** Estimated 3:1

---

## Tech Stack

**Languages:** Python 3.8+

**Libraries:**
- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn, XGBoost, imbalanced-learn
- **Visualization:** matplotlib, seaborn
- **Development:** Jupyter Notebook

---

## Project Structure
```
customer-churn-prediction/
│
├── data/
│   ├── Telco-Customer-Churn.csv       # Original dataset
│   └── DOWNLOAD_INSTRUCTIONS.md       # Dataset source info
│
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb         # Data cleaning & feature engineering
│   ├── 03_modeling.ipynb              # Model training & evaluation
│   └── 04_visualizations.ipynb        # Dashboard creation
│
├── models/
│   ├── churn_model_logistic.pkl       # Saved trained model
│   └── scaler.pkl                     # Feature scaler
│
├── visuals/
│   └── churn_analysis_dashboard.png   # Summary dashboard
│
├── X_train.csv, X_test.csv            # Processed datasets
├── y_train.csv, y_test.csv
│
├── requirements.txt                   # Project dependencies
├── PROJECT_SUMMARY.md                 # One-page project summary
└── README.md                          # This file
```

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the dataset**
- Visit [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Place in `data/` folder and rename to `Telco-Customer-Churn.csv`

4. **Run the notebooks in order**
```
01_eda.ipynb → 02_preprocessing.ipynb → 03_modeling.ipynb → 04_visualizations.ipynb
```

---

## Model Usage

### Load and Use the Trained Model
```python
import pickle
import pandas as pd

# Load model and scaler
with open('models/churn_model_logistic.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Make predictions on new data
# (Ensure data is preprocessed same as training)
predictions = model.predict(X_new)
churn_probabilities = model.predict_proba(X_new)[:, 1]
```

---

## Key Insights from Feature Importance

**Top Churn Drivers (Positive Coefficients):**
1. Fiber optic internet service (+1.30) **Strongest churn influence**
2. Multiple lines (+0.46)
3. Streaming services (+0.44)

**Top Retention Factors (Negative Coefficients):**
1. Two-year contract (-1.60) **Strongest churn deterrent**
2. No internet service (-1.06)
3. Higher tenure (-0.76)

---

## Learning Outcomes

This project demonstrates:
- End-to-end ML workflow (EDA → Preprocessing → Modeling → Deployment)
- Handling imbalanced datasets
- Feature engineering based on domain insights
- Model selection and evaluation with business metrics
- Translating technical results into business recommendations
- Quantifying ROI for data science initiatives

---

## Future Enhancements

- [ ] Deploy model via Streamlit web app
- [ ] Implement time-based churn prediction
- [ ] A/B test retention campaigns
- [ ] Add SHAP values for advanced explainability
- [ ] Automate monthly retraining pipeline
- [ ] Create customer segmentation for targeted campaigns

---

## Contact

**Your Name**
- Email: ewfoster337@gmail.com
- Portfolio: [Ethan's Portfolio Website](https://efoster13.github.io/)

---

## Acknowledgments

- Dataset: IBM Sample Data Sets
- Inspiration: Real-world telecom churn reduction strategies
