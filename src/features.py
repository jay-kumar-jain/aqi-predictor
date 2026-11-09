"""Feature preparation and leakage-safe preprocessing pipeline."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:  # Supports package imports and direct script execution.
    from .preprocess import FEATURE_COLUMNS, TARGET_COLUMN
except ImportError:
    from preprocess import FEATURE_COLUMNS, TARGET_COLUMN


def build_preprocessor():
    numeric = [c for c in FEATURE_COLUMNS if c != "location"]
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, ["location"]),
    ])


def build_features(df):
    """Return raw (unimputed/unscaled) features and target."""
    return df[FEATURE_COLUMNS].copy(), df[TARGET_COLUMN].copy()


def prepare_data(data_path="data/processed/aqi_cleaned.csv", test_size=0.2, random_state=42):
    """Compatibility helper; split raw features before fitting preprocessing."""
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(data_path)
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    preprocessor = build_preprocessor()
    return preprocessor.fit_transform(X_train), preprocessor.transform(X_test), y_train, y_test, preprocessor
