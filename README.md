# Loan Approval Prediction

A machine learning project that predicts whether a loan application will be approved or rejected, based on applicant financial and demographic details. Built as a solo end-to-end pipeline — from raw data to a deployed web app.

## Overview

Loan approval decisions depend on multiple applicant factors (income, credit history, employment, etc.). This project builds and compares several classification models to predict approval outcomes, then deploys the best-performing model as an interactive web app.

## Dataset

- 614 loan applications, 13 original features
- Target: `Loan_Status` (Approved / Rejected), imbalanced ~69% / 31%
- Source: classic Loan Prediction dataset (`train_u6lujuX_CVtuZ9i.csv`)

## Approach

1. **Data cleaning** — missing values imputed (mode for categorical, median for numeric)
2. **Encoding** — categorical variables converted to numeric form; `Property_Area` one-hot encoded
3. **Feature engineering** — combined applicant incomes into `TotalIncome`; log-transformed skewed variables
4. **Model comparison** — tested 6 approaches:
   - Logistic Regression (baseline)
   - Random Forest (default & tuned via GridSearchCV)
   - Gradient Boosting
   - Random Forest + SMOTE (oversampling)
   - Random Forest with balanced class weights
5. **Evaluation** — accuracy, precision, recall, F1-score, confusion matrix, feature importance

## Results

| Model | Accuracy | Recall (Rejected) |
|---|---|---|
| Logistic Regression | 0.854 | 0.58 |
| **Random Forest (default)** | **0.870** | **0.71** |
| Random Forest (tuned) | 0.862 | 0.71 |
| Gradient Boosting | 0.837 | 0.63 |
| Random Forest + SMOTE | 0.829 | 0.68 |
| Random Forest (balanced weight) | 0.837 | 0.63 |

**Best model: Random Forest (default settings)** — 87% accuracy, 0.71 recall on rejected loans.

`Credit_History` was by far the strongest predictor of approval, followed by income-related features. Demographic factors (gender, marital status, education) contributed minimally.

## Web App

A Streamlit app (`app.py`) lets users input applicant details and get a live prediction from the trained model.

### Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `Lone_Approval_Prediction.ipynb` — full analysis: EDA, cleaning, modeling, evaluation
- `app.py` — Streamlit web app
- `loan_model.pkl` — trained Random Forest model
- `scaler.pkl` — fitted StandardScaler for preprocessing new inputs
- `requirements.txt` — Python dependencies

## Author

Arpita Parida
