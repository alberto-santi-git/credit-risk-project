"""
data_prep.py
Cleaning and feature engineering functions for the Credit Risk Dataset (laotse, Kaggle).
Reusable by both the analysis notebook and the Streamlit deployment app.
"""

import pandas as pd
import numpy as np

# Thresholds used to identify clearly wrong values (data-entry errors),
# found through visual inspection of the histograms during EDA.
MAX_PLAUSIBLE_AGE = 95
MAX_PLAUSIBLE_EMP_LENGTH = 66


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Removes rows with clearly wrong age or years of employment."""
    before = len(df)
    df = df[df["person_age"] <= MAX_PLAUSIBLE_AGE].copy()
    df = df[df["person_emp_length"] <= MAX_PLAUSIBLE_EMP_LENGTH].copy()
    removed = before - len(df)
    print(f"Removed {removed} rows ({removed/before:.2%}) due to outliers on age/emp_length")
    return df.reset_index(drop=True)


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Imputes the known missing values in the dataset: person_emp_length and loan_int_rate."""
    df = df.copy()
    df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())
    df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds simple derived features, consistent with a real credit scoring use case."""
    df = df.copy()

    # share of lifetime covered by known credit history
    df["credit_hist_to_age_ratio"] = df["cb_person_cred_hist_length"] / df["person_age"]

    # income left over after the loan, as a proxy for repayment capacity
    df["income_after_loan"] = df["person_income"] - df["loan_amnt"]

    # income per year of employment (proxy for income stability/growth)
    df["income_per_emp_year"] = df["person_income"] / (df["person_emp_length"] + 1)

    return df


def prepare_dataset(path: str) -> pd.DataFrame:
    """Full pipeline: load -> clean -> impute -> feature engineering."""
    df = load_data(path)
    df = clean_outliers(df)
    df = impute_missing(df)
    df = engineer_features(df)
    return df


CATEGORICAL_COLS = ["person_home_ownership", "loan_intent", "cb_person_default_on_file"]
LEAKAGE_COLS = ["loan_grade", "loan_int_rate"]  # assigned by the bank AFTER its own risk assessment
TARGET_COL = "loan_status"


def get_feature_sets(df: pd.DataFrame):
    """
    Returns two feature lists: 'full' (with loan_grade/loan_int_rate) and 'clean'
    (without them), to compare how much the post-assessment variables "help" the model.
    """
    all_cols = [c for c in df.columns if c != TARGET_COL]
    full_features = all_cols
    clean_features = [c for c in all_cols if c not in LEAKAGE_COLS]
    return full_features, clean_features
