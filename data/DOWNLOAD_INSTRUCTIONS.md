# Dataset Download Instructions

## IBM Telco Customer Churn Dataset

This project uses the **IBM Telco Customer Churn Dataset**, a widely-used dataset in the data science community for binary classification problems.

---

## Download Options

### Option 1: Kaggle (Recommended)

1. **Visit Kaggle:**
   - Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

2. **Download the dataset:**
   - Click the **"Download"** button (you may need to create a free Kaggle account)
   - This will download a ZIP file

3. **Extract and place:**
   - Extract the ZIP file
   - Find the file: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
   - Rename it to: `Telco-Customer-Churn.csv`
   - Place it in this `data/` folder

---

### Option 2: Direct Download from GitHub

1. **Visit the IBM GitHub repository:**
   - https://github.com/IBM/telco-customer-churn-on-icp4d/tree/master/data

2. **Download:**
   - Click on `Telco-Customer-Churn.csv`
   - Click the **"Raw"** button
   - Right-click → "Save As..." → save to this `data/` folder

---

### Option 3: Kaggle API (For Advanced Users)

If you have the Kaggle API configured:
```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d blastchar/telco-customer-churn

# Unzip
unzip telco-customer-churn.zip

# Move to data folder
mv WA_Fn-UseC_-Telco-Customer-Churn.csv data/Telco-Customer-Churn.csv
```

---

## Dataset Overview

**File Name:** `Telco-Customer-Churn.csv`  
**Size:** ~950 KB  
**Rows:** 7,043 customers  
**Columns:** 21 features

### Column Descriptions

| Column | Description | Type |
|--------|-------------|------|
| `customerID` | Unique customer identifier | Text |
| `gender` | Customer gender (Male/Female) | Categorical |
| `SeniorCitizen` | Whether customer is senior (1/0) | Binary |
| `Partner` | Whether customer has partner (Yes/No) | Categorical |
| `Dependents` | Whether customer has dependents (Yes/No) | Categorical |
| `tenure` | Number of months with company | Numeric |
| `PhoneService` | Has phone service (Yes/No) | Categorical |
| `MultipleLines` | Has multiple lines (Yes/No/No phone service) | Categorical |
| `InternetService` | Type of internet (DSL/Fiber optic/No) | Categorical |
| `OnlineSecurity` | Has online security (Yes/No/No internet) | Categorical |
| `OnlineBackup` | Has online backup (Yes/No/No internet) | Categorical |
| `DeviceProtection` | Has device protection (Yes/No/No internet) | Categorical |
| `TechSupport` | Has tech support (Yes/No/No internet) | Categorical |
| `StreamingTV` | Has streaming TV (Yes/No/No internet) | Categorical |
| `StreamingMovies` | Has streaming movies (Yes/No/No internet) | Categorical |
| `Contract` | Contract type (Month-to-month/One year/Two year) | Categorical |
| `PaperlessBilling` | Uses paperless billing (Yes/No) | Categorical |
| `PaymentMethod` | Payment method (4 types) | Categorical |
| `MonthlyCharges` | Monthly charge amount | Numeric |
| `TotalCharges` | Total charges to date | Numeric |
| **`Churn`** | **Customer churned (Yes/No)** | **Target Variable** |

---

## Verify Your Download

After downloading, your `data/` folder should look like this:
```
data/
├── Telco-Customer-Churn.csv          ← Your downloaded file
└── DOWNLOAD_INSTRUCTIONS.md          ← This file
```

### Quick Verification

Run this in Python to confirm:
```python
import pandas as pd

df = pd.read_csv('data/Telco-Customer-Churn.csv')
print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Expected: (7043, 21)")
```

**Expected output:**
```
Dataset loaded successfully!
Shape: (7043, 21)
Expected: (7043, 21)
```

---

## Important Notes

1. **File Name:** Make sure the file is named exactly `Telco-Customer-Churn.csv` (case-sensitive on some systems)

2. **Location:** File must be in the `data/` folder relative to the project root

3. **Do NOT commit to Git:** This file is already in `.gitignore` to keep the repository lightweight

4. **License:** This dataset is publicly available for educational and research purposes

---

## Troubleshooting

### Problem: "File not found" error

**Solution:**
- Check the file name is exactly: `Telco-Customer-Churn.csv`
- Check the file is in the `data/` folder
- Check your working directory in Jupyter

### Problem: "UnicodeDecodeError" when loading

**Solution:**
```python
df = pd.read_csv('data/Telco-Customer-Churn.csv', encoding='utf-8')
```

### Problem: Dataset looks different than expected

**Solution:**
- Make sure you downloaded from the correct source (links above)
- The file should be ~7,043 rows × 21 columns
- Re-download if necessary

---

## Dataset Citation

If using this dataset for publications or presentations:
```
IBM Sample Data Sets for Customer Analytics
Source: https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113
```

---

**Once downloaded, you're ready to run the notebooks!** 

Start with: `notebooks/01_eda.ipynb`