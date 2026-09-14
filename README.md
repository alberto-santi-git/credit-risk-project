# Credit Risk Prediction

Classification model estimating the probability of default on consumer loans, based on the [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) (32,581 loan applications).

## The problem and the design of the solution

The goal is to predict `loan_status` (0 = repaid, 1 = default) from the applicant's and loan's characteristics.

Two columns of the dataset — `loan_grade` and `loan_int_rate` — are assigned by the bank **after** its own risk assessment of the customer. Including them in training risks letting the model "copy" a judgment already made by someone else, instead of learning it from the customer's raw data. For this reason the project compares two scenarios:

- **`full`**: all available features, including `loan_grade`/`loan_int_rate`
- **`clean`**: only the features known *before* the risk assessment — the true credit scoring exercise, and the model that gets saved and deployed in the demo app

## Pipeline

1. **Data cleaning**: removal of clear outliers (`person_age` and `person_emp_length` with clearly wrong values, e.g. age >100 years), median imputation of missing values on `person_emp_length` and `loan_int_rate`
2. **Feature engineering**: credit history/age ratio, income left over after the loan, income per year of employment
3. **Modeling**: comparison between Logistic Regression (interpretable baseline), Random Forest and XGBoost, each tuned with `GridSearchCV` and 5-fold stratified cross-validation, optimizing ROC-AUC
4. **Evaluation**: ROC-AUC and PR-AUC on a stratified test set (accuracy alone is not very informative on an imbalanced target, ~22% default), confusion matrix, feature importance
5. **Interpretable scorecard**: as an alternative to black-box models, features are transformed into Weight of Evidence (WoE) via binning, filtered by Information Value (IV), and used to fit a logistic regression converted into a points-based scorecard (scale 300–850, PDO scaling) — the standard approach in regulated banking credit scoring, where each point of the final score is attributable to a specific variable
6. **Deploy**: Streamlit app that loads the saved `clean` model and returns a real-time default probability

## Repo structure

```
├── notebooks/
│   └── credit_risk_analysis.ipynb   # EDA, cleaning, training, assessment, comparison of ‘full’ versus ‘clean’
├── src/
│   ├── data_prep.py                  # data cleaning and feature engineering (reused from notebooks and apps)
│   ├── modeling.py                   # pre-processing, training, model evaluation
│   └── scorecard.py                  # WoE, Information Value and the creation of a points-based scorecard
├── models/                           # final saved model 
├── data/                             
├── app.py                            # demo Streamlit
└── requirements.txt
```

## How to run it

```bash
pip install -r requirements.txt

# 1. Download the dataset from Kaggle and place it in data/credit_risk_dataset.csv

# 2. Run the notebook (generates models/best_credit_risk_model.pkl)
jupyter notebook notebooks/credit_risk_analysis.ipynb

# 3. Launch the demo
streamlit run app.py
```

## Possible extensions

- Probability calibration for direct use as a PD in an IFRS 9 context
- Fairness analysis on variables such as `person_home_ownership`
- Optimised decision threshold based on the asymmetric cost of false negatives versus false positives, rather than the default threshold of 0.5
