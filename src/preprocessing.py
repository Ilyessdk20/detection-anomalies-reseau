from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LABEL_MAP = {"normal": 0, "anomaly": 1}


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset into a pandas DataFrame."""
    return pd.read_csv(csv_path)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names: lowercase, trimmed, snake_case style."""
    cleaned = df.copy()
    cleaned.columns = (
        cleaned.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return cleaned


def clean_network_data(df: pd.DataFrame, target_column: str = "label") -> pd.DataFrame:
    """Apply light cleaning to network tabular data."""
    cleaned = clean_column_names(df)

    # Normalize text columns to avoid category duplication (e.g., "HTTP" vs "http").
    object_columns = cleaned.select_dtypes(include="object").columns
    for column in object_columns:
        cleaned[column] = cleaned[column].astype(str).str.strip().str.lower()
        cleaned[column] = cleaned[column].replace({"": pd.NA, "nan": pd.NA})

    if target_column in cleaned.columns:
        cleaned[target_column] = cleaned[target_column].replace(LABEL_MAP)
        cleaned[target_column] = pd.to_numeric(cleaned[target_column], errors="coerce")
        cleaned = cleaned.dropna(subset=[target_column])
        cleaned[target_column] = cleaned[target_column].astype(int)

    # Try numeric conversion for all non-categorical columns.
    categorical_like = {"protocol", "service", target_column}
    for column in cleaned.columns:
        if column not in categorical_like:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned


def missing_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing values count and ratio for each column."""
    missing_count = df.isna().sum()
    missing_ratio = (missing_count / len(df) * 100).round(2)

    summary = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_ratio_percent": missing_ratio,
        }
    )
    return summary[summary["missing_count"] > 0].sort_values(
        by="missing_count", ascending=False
    )


def split_features_target(
    df: pd.DataFrame, target_column: str = "label"
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Split DataFrame into features and optional target."""
    if target_column in df.columns:
        return df.drop(columns=[target_column]), df[target_column]
    return df.copy(), None


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create a preprocessing pipeline for numeric and categorical columns."""
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = features.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )
