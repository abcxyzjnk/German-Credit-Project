# German Credit Risk Predictive Modeling 🏦📊

An end-to-end Machine Learning project to automate credit scoring and optimize the trade-off between default risk and lending revenue.

---

## 1. Project Overview & Business Context
Credit provisioning is the primary revenue engine for financial institutions but inherently exposes them to substantial default risk. This project applies advanced predictive modeling workflows on the **German Credit Dataset** (1,000 applicant records) to classify individuals into **Good Risk (1)** or **Bad Risk (0)**. 

The primary task is to develop a data-driven system that mitigates capital losses from Non-Performing Loans (NPLs) while sustaining portfolio growth.

---

## 2. Comprehensive Project Report (PDF)
> 💡 **For full methodology, theoretical framework, mathematical formulation, and in-depth business conclusions, please refer to my complete academic report:**

---

## 3. Key Findings & Recalibrated Performance
Our Exploratory Data Analysis (EDA) successfully mapped the feature schema, identifying **3 actual continuous numerical attributes** (`duration`, `credit_amount`, `age`) and 18 masked categorical features. 

After hyperparameter tuning and mitigating class imbalance (70% Good vs. 30% Bad) via SMOTE, the models evaluated on a 300-sample test set ($210$ Good, $90$ Bad) demonstrated a critical strategic trade-off:

* **Model 1: Logistic Regression (Balanced Champion - Accuracy: 62.0%)**
    * Achieved a **70.0% Recall on the Bad class**, catching 63/90 high-risk defaults.
    * Maintained a more commercially sustainable profile by minimizing false rejections of good clients.
* **Model 2: Random Forest (Risk-Averse Shield - Accuracy: 52.7%)**
    * Delivered an elite **81.1% Recall on the Bad class**, letting only 17 defaults slip through.
    * Acts as the ultimate capital protection mechanism during economic downturns, despite higher opportunity costs.
* **Model 3: Naive Bayes (Baseline - Accuracy: 60.0%)**
    * Served as a probabilistic baseline to benchmark linear and non-linear interactions.

---

## 4. Quick Start & Repository Structure
### Repository Layout
```text
├── data/              # Raw source and comprehensive Data Dictionary
├── notebooks/         # Jupyter Notebook for EDA & Model Training pipelines
├── reports/           
│   └── project_report.pdf
