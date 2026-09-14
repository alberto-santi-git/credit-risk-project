"""
data_prep.py
Funzioni di pulizia e feature engineering per il Credit Risk Dataset (laotse, Kaggle).
Riutilizzabile sia dal notebook di analisi sia dall'app Streamlit di deploy.
"""

import pandas as pd
import numpy as np

# Soglie usate per identificare valori chiaramente errati (errori di inserimento),
# individuate tramite ispezione visiva degli istogrammi in fase di EDA.
MAX_PLAUSIBLE_AGE = 100
MAX_PLAUSIBLE_EMP_LENGTH = 60


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Rimuove righe con età o anni di esperienza lavorativa chiaramente errati."""
    before = len(df)
    df = df[df["person_age"] <= MAX_PLAUSIBLE_AGE].copy()
    df = df[df["person_emp_length"] <= MAX_PLAUSIBLE_EMP_LENGTH].copy()
    removed = before - len(df)
    print(f"Rimosse {removed} righe ({removed/before:.2%}) per outlier su age/emp_length")
    return df.reset_index(drop=True)


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa i missing noti del dataset: person_emp_length e loan_int_rate."""
    df = df.copy()
    df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())
    df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge feature derivate semplici, coerenti con un caso di credit scoring reale."""
    df = df.copy()

    # quota di vita lavorativa coperta da storico creditizio noto
    df["credit_hist_to_age_ratio"] = df["cb_person_cred_hist_length"] / df["person_age"]

    # reddito disponibile dopo il prestito, come proxy di capacità di rimborso
    df["income_after_loan"] = df["person_income"] - df["loan_amnt"]

    # reddito per anno di esperienza lavorativa (proxy di stabilità/crescita reddituale)
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
LEAKAGE_COLS = ["loan_grade", "loan_int_rate"]  # assegnati dalla banca DOPO la valutazione del rischio
TARGET_COL = "loan_status"


def get_feature_sets(df: pd.DataFrame):
    """
    Ritorna due liste di feature: 'full' (con loan_grade/loan_int_rate) e 'clean'
    (senza), per confrontare quanto le variabili post-valutazione "aiutano" il modello.
    """
    all_cols = [c for c in df.columns if c != TARGET_COL]
    full_features = all_cols
    clean_features = [c for c in all_cols if c not in LEAKAGE_COLS]
    return full_features, clean_features
