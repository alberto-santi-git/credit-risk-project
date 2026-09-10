"""
modeling.py
Pre-processing (encoding), training and evaluation of credit risk models.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, RocCurveDisplay
)
from xgboost import XGBClassifier

from data_prep import TARGET_COL


def build_preprocessor(feature_cols, X_ref: pd.DataFrame):
    """
    One-hot encoding for the categorical features in the selected feature set.
    Categorical columns are identified by the dtype of X_ref (e.g. X_train),
    rather than from a fixed list, to avoid the risk of overlooking columns such as “loan_grade”.
    """
    cat_cols = X_ref[feature_cols].select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
    )
    return preprocessor, num_cols, cat_cols


def split_data(df: pd.DataFrame, feature_cols, test_size=0.2, random_state=42):
    X = df[feature_cols]
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


MODELS = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        {"clf__C": [0.01, 0.1, 1, 10]},
    ),
    "random_forest": (
        RandomForestClassifier(class_weight="balanced", random_state=42),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [6, 10, None]},
    ),
    "xgboost": (
        XGBClassifier(eval_metric="logloss", random_state=42),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 5, 7], "clf__learning_rate": [0.05, 0.1]},
    ),
}


def train_and_evaluate(X_train, X_test, y_train, y_test, feature_cols, model_name, cv_folds=5):
    """Train a model using GridSearchCV (stratified CV) and evaluate it on the test set."""
    preprocessor, num_cols, cat_cols = build_preprocessor(feature_cols, X_train)
    base_model, param_grid = MODELS[model_name]

    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", base_model),
    ])

    # scale_pos_weight for XGBoost (it does not have a native class_weight as in sklearn) 
    if model_name == "xgboost":
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        pipe.set_params(clf__scale_pos_weight=neg / pos)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    search = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    results = {
        "model_name": model_name,
        "best_params": search.best_params_,
        "cv_best_roc_auc": search.best_score_,
        "test_roc_auc": roc_auc_score(y_test, y_proba),
        "test_pr_auc": average_precision_score(y_test, y_proba),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    return best_model, results


def print_results(results: dict):
    print(f"\n{'='*60}\nModel: {results['model_name']}\n{'='*60}")
    print(f"Best Hyperparameters: {results['best_params']}")
    print(f"ROC-AUC (CV):   {results['cv_best_roc_auc']:.4f}")
    print(f"ROC-AUC (test): {results['test_roc_auc']:.4f}")
    print(f"PR-AUC  (test): {results['test_pr_auc']:.4f}")
    print("\nClassification report (test):")
    print(results["classification_report"])
    print("Confusion matrix (test):")
    print(results["confusion_matrix"])
