# Customer Churn Prediction & Retention Strategy

**Project Type:** End-to-End Machine Learning  
**Domain:** Telecom/Subscription Business  
**Tools:** Python, Pandas, Scikit-learn, Matplotlib, Seaborn

---

## Business Problem
Customer acquisition costs 5-25x more than retention. This project predicts which customers will churn and provides data-driven retention strategies.

---

## Dataset
- **Source:** IBM Telco Customer Churn Dataset
- **Size:** 7,043 customers, 21 features
- **Target:** Binary churn classification (Yes/No)

---

## Key Findings

### 1. Churn Rate: 26.5% (High)
- 1 in 4 customers leave
- Significant business problem requiring intervention

### 2. Contract Type - Primary Driver
- Month-to-month: **42.7% churn**
- Two-year: **2.8% churn**
- 15x difference in churn rate

### 3. Fiber Optic Service - Critical Issue
- Churn rate: **41.9%**
- Average cost: **$91/month**
- Highest-risk customer group

### 4. Tenure Pattern
- Most churn occurs in first 6 months
- Long-term customers (38+ months) rarely leave

---

## Model Performance Comparison

| Model | ROC-AUC | Recall | Precision |
|-------|---------|--------|-----------|
| Logistic Regression | 0.8461 | 53% | 67% |
| XGBoost | 0.8403 | 53% | 65% |
| Random Forest | 0.8412 | 50% | 65% |

**Selected Model:** Logistic Regression (highest ROC-AUC, most interpretable)

---

## Business Recommendations

### 1. Contract Incentive Program
- Offer 15-20% discount for switching to annual contracts
- **Impact:** Reduce churn from 42.7% → ~25%

### 2. Fiber Loyalty Bundle
- Introduce tiered pricing for fiber customers
- **Impact:** Reduce fiber churn from 41.9% → ~25%

### 3. First 90 Days Onboarding
- Proactive check-ins at 30/60/90 days for new customers
- **Impact:** Reduce early churn by 15-20%

---

## Projected Business Impact
- **Churn reduction target:** 10-15%
- **Estimated annual savings:** $467,500 - $700,000
- **ROI of retention programs:** 3:1

---

## Technical Implementation
- Data cleaning & feature engineering
- Handled class imbalance (26.5% minority class)
- Engineered features: `avg_monthly_cost`, `is_new_customer`
- Model explainability via coefficient analysis
- Saved model for deployment (`churn_model_logistic.pkl`)

---

**GitHub:**  https://github.com/EFoster13
**Contact:** ewfoster337@gmai.com