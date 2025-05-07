# 🧠 Project: Loan Eligibility Prediction – EDA Report

## 1. Objective
Conduct Exploratory Data Analysis (EDA) on the loan eligibility dataset to:
- Understand the structure and quality of features.
- Detect any data issues.
- Plan preprocessing steps for modeling.

---

## 2. Target Variable Analysis: `Loan_Status`
**Distribution:**
- 68.7% applicants eligible (`Y`).
- 31.3% not eligible (`N`).
- Slight class imbalance.

**Action:**
- Monitor using F1-score, Precision/Recall during model evaluation.
- No resampling needed.

---

## 3. Categorical Features Analysis

| Feature         | Dominance?       | Rare Categories? | Missing Values | Interpretation                              |
|------------------|------------------|------------------|----------------|---------------------------------------------|
| **Gender**       | 79% Male         | No               | 2%             | Mild dominance. Keep. Fill missing logically. |
| **Married**      | 64% Married      | No               | 0.5%           | Normal balance. Keep. Fill missing logically. |
| **Dependents**   | 0/1/2/3+         | 3+ = 8%          | 2%             | Small rare group but acceptable. Keep.      |
| **Education**    | 78% Graduate     | No               | 0%             | Dominance expected. Keep.                   |
| **Self_Employed**| 81% No           | No               | 5.2%           | Strong dominance, but meaningful. Keep. Fill missing logically. |
| **Property_Area**| Balanced         | No               | 0%             | Very good distribution. Keep.               |

**Note:**
- `Loan_ID` will be dropped (identifier, not a feature).
- `Loan_Status` is the target (will be separated later).

---

## 4. Numerical Features Analysis

| Feature             | Distribution       | Outliers       | Missing Values | Interpretation                              |
|----------------------|--------------------|----------------|----------------|---------------------------------------------|
| **ApplicantIncome**  | Right-skewed      | Some           | 0%             | Log transformation recommended.             |
| **CoapplicantIncome**| Right-skewed      | Some           | 0%             | Log transformation recommended.             |
| **LoanAmount**       | Right-skewed      | Extreme        | 22 missing      | Log transform. Impute missing values.       |
| **Loan_Amount_Term** | Discrete peaks    | Minor rare plans| 14 missing      | Impute missing with mode (360).             |
| **Credit_History**   | Dominance at 1    | No             | 50 missing      | Important binary categorical feature. Handle missing carefully. |

---

## 5. Special Feature Interpretation

| Feature             | Logical Behavior               | Treatment                                   |
|----------------------|--------------------------------|--------------------------------------------|
| **Dependents**       | Categorical                   | One-Hot Encode                             |
| **Credit_History**   | Binary Categorical (`0/1`)    | Leave as is (`0/1`).                       |
| **Loan_Amount_Term** | Discrete Numerical            | Treat as numerical (consider bucketing rare terms later if needed). |

---

## 6. Missing Values Summary

| Feature             | % Missing | Handling Strategy                              |
|----------------------|-----------|-----------------------------------------------|
| **Gender**           | ~2%       | Fill with most frequent value or "Unknown".   |
| **Married**          | ~0.5%     | Fill with most frequent value.                |
| **Dependents**       | ~2%       | Fill with most frequent value or create "Unknown". |
| **Self_Employed**    | ~5.2%     | Fill with "No" (logical assumption) or "Unknown". |
| **LoanAmount**       | ~5.2%     | Fill with median value.                       |
| **Loan_Amount_Term** | ~2.5%     | Fill with mode (360).                         |
| **Credit_History**   | ~8%       | Fill with 1 (good) cautiously or treat missing separately as its own category. |

---