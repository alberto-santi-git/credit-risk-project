"""
scorecard.py
Weight of Evidence (WoE), Information Value (IV) and the scorecard construction
in the style of a banking credit score (score range 300–850, PDO scaling), based
on a logistic regression fitted to features transformed using WoE..

"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from data_prep import TARGET_COL

# A small constant to avoid log(0) when a bin has neither ‘good’ nor ‘bad’
EPS = 0.5


#  Weight of Evidence e Information Value

def _woe_iv_from_counts(good_counts: pd.Series, bad_counts: pd.Series) -> pd.DataFrame:
    """
    Given the distribution of ‘good’ and ‘bad’ values per bin, calculate the WoE and the contribution
    of each bin to the total IV of the variable.

    Convention: ‘bad’ = 1 (default), ‘good’ = 0 (repaid) -> a positive WoE
    indicates a low-risk bin (proportionally more ‘good’ than ‘bad’).
    """
    total_good = good_counts.sum()
    total_bad = bad_counts.sum()

    # EPS evita divisioni per zero/log(0) nei bin senza osservazioni di una classe
    pct_good = (good_counts + EPS) / (total_good + EPS * len(good_counts))
    pct_bad = (bad_counts + EPS) / (total_bad + EPS * len(bad_counts))

    woe = np.log(pct_good / pct_bad)
    iv_contribution = (pct_good - pct_bad) * woe

    table = pd.DataFrame({
        "n_good": good_counts,
        "n_bad": bad_counts,
        "pct_good": pct_good,
        "pct_bad": pct_bad,
        "woe": woe,
        "iv_contribution": iv_contribution,
    })
    return table


def woe_iv_numeric(df: pd.DataFrame, feature: str, target: str = TARGET_COL,
                    max_bins: int = 5) -> pd.DataFrame:
    """
    Quantile binning (pd.qcut) of a numerical variable, followed by WoE/IV per bin.
    duplicates="drop" handles variables with many repeated values (e.g. long queues)
    which would otherwise cause qcut to fail due to non-unique bin boundaries.
    """
    binned = pd.qcut(df[feature], q=max_bins, duplicates="drop")
    good_counts = df.loc[df[target] == 0].groupby(binned, observed=True).size()
    bad_counts = df.loc[df[target] == 1].groupby(binned, observed=True).size()
    # align the index if a bin has no observations in a class
    good_counts, bad_counts = good_counts.align(bad_counts, fill_value=0)
    table = _woe_iv_from_counts(good_counts.sort_index(), bad_counts.sort_index())
    table.index.name = feature
    return table


def woe_iv_categorical(df: pd.DataFrame, feature: str, target: str = TARGET_COL) -> pd.DataFrame:
    """WoE/IV for a categorical variable, one bin for each observed category."""
    good_counts = df.loc[df[target] == 0].groupby(feature, observed=True).size()
    bad_counts = df.loc[df[target] == 1].groupby(feature, observed=True).size()
    good_counts, bad_counts = good_counts.align(bad_counts, fill_value=0)
    table = _woe_iv_from_counts(good_counts.sort_index(), bad_counts.sort_index())
    table.index.name = feature
    return table


def woe_iv_table(df: pd.DataFrame, feature: str, target: str = TARGET_COL,
                  max_bins: int = 5) -> pd.DataFrame:
    """Dispatcher: selects numerical or categorical binning based on the dtype."""
    if pd.api.types.is_numeric_dtype(df[feature]):
        return woe_iv_numeric(df, feature, target, max_bins)
    return woe_iv_categorical(df, feature, target)


def _iv_bucket(iv: float) -> str:
    """Standard interpretation of IV thresholds used in credit scoring."""
    if iv < 0.02:
        return "non-predictive"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "suspicious (possible leakage/overfitting)"


def iv_summary(df: pd.DataFrame, features: list, target: str = TARGET_COL,
               max_bins: int = 5) -> pd.DataFrame:
    """
    Overall IV for a list of features, sorted in descending order.
    A starting point for deciding which variables to include in the scorecard.
    """
    rows = []
    for feat in features:
        table = woe_iv_table(df, feat, target, max_bins)
        iv = table["iv_contribution"].sum()
        rows.append({"feature": feat, "iv": iv, "interpretation": _iv_bucket(iv)})
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


# 2. WoE transformer 

class WOETransformer:
    """
    Learn the bins and their corresponding WoEs on the training set, then map any new
    data (validation, test, or an individual customer in the app) to the same bins.

    Numeric variables are binned by quantiles (bins saved in the fit);
    categorical variables are mapped category → WoE. A value or category never seen
    in the fit is mapped to WoE=0 (neutral contribution), rather than causing
    the transformation to fail.
    """

    def __init__(self, features: list, target: str = TARGET_COL, max_bins: int = 5):
        self.features = features
        self.target = target
        self.max_bins = max_bins
        self.bin_edges_ = {}      # numerical feature -> array of edges (from pd.qcut in fit)
        self.woe_maps_ = {}       # feature -> {bin/category: woe}
        self.iv_ = {}             # feature -> total IV (useful for post-fit inspection)
        self.is_numeric_ = {}     # feature -> bool

    def fit(self, df: pd.DataFrame):
        for feat in self.features:
            is_num = pd.api.types.is_numeric_dtype(df[feat])
            self.is_numeric_[feat] = is_num

            if is_num:
                # retbins=True to save the edges and reapply them identically in the transform
                binned, edges = pd.qcut(df[feat], q=self.max_bins, duplicates="drop", retbins=True)
                self.bin_edges_[feat] = edges
                table = self._woe_from_binned(df, binned, feat)
            else:
                table = woe_iv_categorical(df, feat, self.target)

            self.woe_maps_[feat] = table["woe"].to_dict()
            self.iv_[feat] = table["iv_contribution"].sum()
        return self

    def _woe_from_binned(self, df, binned, feat):
        good_counts = df.loc[df[self.target] == 0].groupby(binned, observed=True).size()
        bad_counts = df.loc[df[self.target] == 1].groupby(binned, observed=True).size()
        good_counts, bad_counts = good_counts.align(bad_counts, fill_value=0)
        return _woe_iv_from_counts(good_counts.sort_index(), bad_counts.sort_index())

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for feat in self.features:
            if self.is_numeric_[feat]:
                edges = self.bin_edges_[feat].copy()
                # extends the first/last edge to capture values outside the range seen in the fit
                edges[0], edges[-1] = -np.inf, np.inf
                binned = pd.cut(df[feat], bins=edges, include_lowest=True)
                out[f"{feat}_woe"] = binned.map(self.woe_maps_[feat]).astype(float).fillna(0.0)
            else:
                out[f"{feat}_woe"] = df[feat].map(self.woe_maps_[feat]).fillna(0.0)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# Scorecard: from logistics coefficients (on WoE) to a score of 300–850

def fit_scorecard_logit(woe_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """
    Logistics regarding features in WoE. There is no `class_weight=‘balanced’` here:
    unlike the ‘pure’ classification model in `modelling.py`, the
    traditional scorecard works on the portfolio’s natural proportions,
    because the `base_score` and `odds` are calibrated to the observed default rate.
    """
    logit = LogisticRegression(max_iter=1000)
    logit.fit(woe_train, y_train)
    return logit


def build_scorecard(logit: LogisticRegression, woe_transformer: WOETransformer,
                     base_score: int = 600, base_odds: float = 50.0,
                     pdo: int = 20) -> dict:
    """
    Converts intercepts and logistic coefficients (fitted to the WoE) into
    scorecard points, using the standard ‘Points to Double the Odds’ scaling:

        score = offset + factor * ln(odds)
        factor = PDO / ln(2)
        offset = base_score - factor * ln(base_odds)

    Each bin is assigned a partial score: -(WoE_bin * beta_j + intercept/n_feature) * factor.
    The customer’s score is the sum of the scores for each of their variables.
    """
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)

    features = woe_transformer.features
    coefs = dict(zip([f"{f}_woe" for f in features], logit.coef_[0]))
    intercept = logit.intercept_[0]
    n_features = len(features)

    points_table = {}
    for feat in features:
        beta = coefs[f"{feat}_woe"]
        bin_points = {}
        for bin_label, woe in woe_transformer.woe_maps_[feat].items():
            # The intercept is distributed equally amongst the variables,
            # a standard convention to avoid a ‘base score’ being hidden within a single feature
            pts = -(woe * beta + intercept / n_features) * factor + offset / n_features
            bin_points[bin_label] = round(pts, 1)
        points_table[feat] = bin_points

    return {
        "base_score": base_score,
        "base_odds": base_odds,
        "pdo": pdo,
        "points_table": points_table,
    }


def score_applicant(scorecard: dict, woe_transformer: WOETransformer,
                     applicant: pd.DataFrame) -> float:
    """
    Apply the scorecard to a new customer (a row of raw data, not in WoE):
    Create bins using the boundaries learnt during the fit, then sum the partial score for each bin.
    Returns the total score, which can be broken down by variable via `points_table`
    (useful for explainability: “how many points lost by income, by age, etc.”).
    """
    woe_row = woe_transformer.transform(applicant)
    total = 0.0
    breakdown = {}
    for feat in scorecard["points_table"]:
        col = f"{feat}_woe"
        woe_val = woe_row[col].iloc[0]
        # Find the bin corresponding to the calculated WoE (exact match in the fit dictionary)
        bin_points = scorecard["points_table"][feat]
        woe_to_points = {
            woe_transformer.woe_maps_[feat][b]: pts for b, pts in bin_points.items()
        }
        pts = woe_to_points.get(woe_val, 0.0)
        breakdown[feat] = pts
        total += pts
    return total, breakdown
