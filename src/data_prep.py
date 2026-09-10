"""
data_prep.py
Data cleaning and feature engineering for the Credit Risk Dataset (laotse, Kaggle).
Reusable both in the analysis notebook and in the Streamlit deployment app.
"""

import pandas as pd
import numpy as np

# Thresholds used to identify clearly incorrect values (input errors),
# identified through visual inspection of the histograms during the EDA phase.
MAX_PLAUSIBLE_AGE = 95
MAX_PLAUSIBLE_EMP_LENGTH = 66


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Removes rows containing clearly incorrect ages or years of work experience."""
    before = len(df)
    df = df[df["person_age"] <= MAX_PLAUSIBLE_AGE].copy()
    df = df[df["person_emp_length"] <= MAX_PLAUSIBLE_EMP_LENGTH].copy()
    removed = before - len(df)
    print(f"Removed {removed} rows ({removed/previously: 0.2%}) for each outlier relating to age/length of service")
    return df.reset_index(drop=True)


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute the known missing values in the dataset using the median: person_emp_length and loan_int_rate."""
    df = df.copy()
    df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())
    df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """It adds simple derived features that are consistent with a real-world credit scoring scenario."""
    df = df.copy()

    # proportion of working life covered by a known credit history
    df["credit_hist_to_age_ratio"] = df["cb_person_cred_hist_length"] / df["person_age"]

    # disposable income after the loan, as a proxy for repayment capacity
    df["income_after_loan"] = df["person_income"] - df["loan_amnt"]

    # income per year of work experience (a proxy for income stability/growth)
    df["income_per_emp_year"] = df["person_income"] / (df["person_emp_length"] + 1)

    return df


def prepare_dataset(path: str) -> pd.DataFrame:
    """Pipeline completa: load -> clean -> impute -> feature engineering."""
    df = load_data(path)
    df = clean_outliers(df)
    df = impute_missing(df)
    df = engineer_features(df)
    return df


CATEGORICAL_COLS = ["person_home_ownership", "loan_intent", "cb_person_default_on_file"]
LEAKAGE_COLS = ["loan_grade", "loan_int_rate"]  # assigned by the bank AFTER the risk assessment
TARGET_COL = "loan_status"


def get_feature_sets(df: pd.DataFrame):
    """
    It returns two lists of features: “full” (including loan_grade and loan_int_rate) and “clean”
    (excluding them), to compare how much the post-valuation variables ‘help’ the model.
    """
    all_cols = [c for c in df.columns if c != TARGET_COL]
    full_features = all_cols
    clean_features = [c for c in all_cols if c not in LEAKAGE_COLS]
    return full_features, clean_features
