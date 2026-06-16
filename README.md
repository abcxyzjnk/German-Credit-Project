# German Credit Risk Predictive Modeling

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Machine_Learning-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Framework](https://img.shields.io/badge/Methodology-CRISP--DM-green.svg)](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)

## 1. Project Overview & Business Context
In the banking industry, credit underwriting is the core revenue driver but inherently introduces substantial credit default risk. This project leverages advanced machine learning workflows on the **German Credit Dataset** (1,000 historical customer loan applications) to automate credit scoring and classify applicants into **Good Risk (1)** or **Bad Risk (0)**. 

The primary business objective is to balance the trade-off between maximizing lending interest margins (lowering opportunity costs) and protecting the bank's core capital from non-performing loans (NPLs).

---

## 2. Key Insights from Exploratory Data Analysis (EDA)
* **Feature Schema Discovered:** Out of 21 raw features, metadata exploration verified that only **3 features are genuinely continuous** (`duration`, `credit_amount`, `age`), while the remaining 18 are encoded categorical attributes.
* **Primary Risk Indicators:** Correlation matrices (Pearson, Spearman, Kendall) highlight that `checking_account`, `credit_history`, and `savings_account` hold the strongest positive correlation with creditworthiness. Conversely, loan `duration` acts as the strongest negative predictor (prolonged duration increases default risk).
* **Class Imbalance:** The target class exhibits an implicit **70% "Good" vs. 30% "Bad"** base rate imbalance, requiring strategic sampling and cost-sensitive adjustments.

---

## 3. Methodology & Core Framework
The workflow follows a rigorous pipeline:
1. **Data Preprocessing:** Standard Scaling for continuous distributions, One-Hot Encoding for nominal categorical vectors, and handling original German code mappings.
2. **Imbalance Mitigation:** Applied SMOTE (Synthetic Minority Over-sampling Technique) to ensure the models learn high-risk behavior patterns adequately.
3. **Algorithms Evaluated:** Trained and fine-tuned three distinct modeling archetypes:
   * **Logistic Regression** (Linear Discriminative Baseline)
   * **Random Forest** (Non-parametric Tree Ensemble)
   * **Naive Bayes** (Generative Probabilistic Classifier)

---

## 4. Recalibrated Model Performance & Trade-Offs

Evaluated on a 300-sample test set ($210$ Good, $90$ Bad), the fine-tuned models revealed a stark strategic trade-off:

| Metric | Model 1: Logistic Regression | Model 2: Random Forest | Model 3: Naive Bayes |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **62.0%** | 52.7% | 60.0% |
| **Recall (Bad Class)** | 70.0% | **81.1%** | 63.3% |
| **Precision (Bad Class)** | 42.0% | 36.9% | 39.6% |
| **Recall (Good Class)** | 58.6% | 40.5% | 58.6% |

### Key Takeaways & Strategic Decision:
* **Model 2 (Random Forest)** serves as a **highly conservative "Risk-Averse Shield"**, capturing an elite **81.1% of high-risk defaults** (only 17 bad loans slipped through). However, it introduces a substantial opportunity cost by falsely rejecting 125 creditworthy customers.
* **Model 1 (Logistic Regression)** offers a more **commercially balanced alternative**, securing the highest overall accuracy (62.0%) while still capturing 70.0% of default behaviors.

---

## 5. Repository Structure
```text
├── data/
│   ├── raw/               # Original encoded source file
│   ├── processed/         # Cleaned, standardized, and scaled arrays
│   └── README.md          # Comprehensive Data Dictionary
├── notebooks/
│   └── credit_risk_modeling_eda_and_training.ipynb  # End-to-end Jupyter Notebook
├── src/                   # Python source utility scripts
├── requirements.txt       # Project dependencies
└── README.md              # Project executive summary (This file)
