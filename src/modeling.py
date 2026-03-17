from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def select_modeling_strategy(target: Optional[pd.Series]) -> str:
    """Choose supervised or unsupervised strategy depending on labels."""
    if target is not None and target.nunique(dropna=True) > 1:
        return "supervised"
    return "isolation_forest"


def train_supervised_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
) -> Pipeline:
    """Train a Logistic Regression classifier with preprocessing."""
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def evaluate_supervised_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """Return simple metrics for binary classification."""
    y_pred = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
    }

    if hasattr(model, "predict_proba") and y_test.nunique() > 1:
        y_scores = model.predict_proba(X_test)[:, 1]
        results["roc_auc"] = roc_auc_score(y_test, y_scores)

    return results


def train_isolation_forest(
    X_train: pd.DataFrame,
    preprocessor,
    contamination: float = 0.2,
) -> Pipeline:
    """Train an Isolation Forest pipeline for anomaly detection."""
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "detector",
                IsolationForest(
                    n_estimators=300,
                    contamination=contamination,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train)
    return model


def predict_anomaly_with_isolation_forest(model: Pipeline, X_data: pd.DataFrame) -> np.ndarray:
    """Convert Isolation Forest output from {-1, 1} to {1, 0} for anomaly label."""
    raw_predictions = model.predict(X_data)
    return np.where(raw_predictions == -1, 1, 0)


def evaluate_unsupervised_predictions(
    y_true: Optional[pd.Series], y_pred: np.ndarray
) -> Dict:
    """Evaluate anomaly predictions when reference labels are available."""
    if y_true is None:
        return {"note": "No labels available: only predictions were generated."}

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }
