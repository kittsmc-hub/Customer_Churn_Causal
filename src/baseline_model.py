"""
Baseline churn classifier (XGBoost).

Purpose: establish the predictive performance ceiling and demonstrate, with
numbers, why a high-AUC churn classifier is not an intervention-targeting
tool. This model predicts P(churn | features, observed treatment) — it is
NOT a causal model and must never be used to decide who gets a discount.
That distinction is the entire point of this repo; see src/causal_model.py
for the model that actually answers an intervention question.

Run: python3 src/baseline_model.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

FEATURE_COLS = [
    "tenure_months",
    "monthly_charges",
    "support_calls",
    "has_dependents",
    "senior_citizen",
    "paperless_billing",
    "discount_offered",  # included as a feature here on purpose, see README
]
CATEGORICAL_COLS = ["contract", "internet_service", "payment_method"]
TARGET_COL = "churned"


@dataclass
class BaselineMetrics:
    auc: float
    average_precision: float
    brier_score: float
    n_train: int
    n_test: int
    churn_rate_test: float


def _build_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS].copy()
    dummies = pd.get_dummies(df[CATEGORICAL_COLS], drop_first=True)
    return pd.concat([X, dummies], axis=1)


def train_baseline(df: pd.DataFrame, seed: int = 42) -> tuple[XGBClassifier, BaselineMetrics, list[str]]:
    X = _build_design_matrix(df)
    y = df[TARGET_COL].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    metrics = BaselineMetrics(
        auc=float(roc_auc_score(y_test, proba)),
        average_precision=float(average_precision_score(y_test, proba)),
        brier_score=float(brier_score_loss(y_test, proba)),
        n_train=len(X_train),
        n_test=len(X_test),
        churn_rate_test=float(y_test.mean()),
    )
    return model, metrics, list(X.columns)


if __name__ == "__main__":
    df = pd.read_csv("data/churn_data.csv")
    model, metrics, feature_names = train_baseline(df)
    print(json.dumps(asdict(metrics), indent=2))

    importances = sorted(
        zip(feature_names, model.feature_importances_.tolist()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    print("\nTop feature importances:")
    for name, score in importances[:8]:
        print(f"  {name:25s} {score:.4f}")

    print(
        "\nNote: 'discount_offered' importance here reflects correlation, "
        "not the causal effect of offering a discount. A high-risk "
        "customer who already got a discount is still high-risk -- that's "
        "what this model has learned, and it is the trap the rest of this "
        "repo exists to avoid."
    )
