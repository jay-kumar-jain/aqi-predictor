"""Feature engineering and leakage-safe preprocessing pipeline for AQI Prediction."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:  # Supports package imports and direct script execution.
    from .preprocess import BASE_FEATURE_COLUMNS, TARGET_COLUMN
except ImportError:
    try:
        from preprocess import BASE_FEATURE_COLUMNS, TARGET_COLUMN
    except ImportError:
        BASE_FEATURE_COLUMNS = [
            "location", "temp_c", "humidity", "pressure_mb", "windspeed_kph",
            "pm2_5", "pm10", "co", "no2", "hour", "day", "month", "day_of_week",
        ]
        TARGET_COLUMN = "aqi_index"

# Backwards compatibility alias
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS


class AQIFeatureEngineer(BaseEstimator, TransformerMixin):
    """Domain-grounded environmental and cyclical feature engineer for Delhi air quality."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Derives particulate ratios, stagnation index, dew point, cyclical time, and domain flags."""
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # Particulate dynamics
        df["pm_ratio"] = df["pm2_5"] / (df["pm10"] + 1e-4)
        df["pm_total"] = df["pm2_5"] + df["pm10"]
        df["pm_diff"] = np.maximum(df["pm10"] - df["pm2_5"], 0.0)

        # Atmospheric physics and dispersion proxies
        # Low wind speed causes atmospheric stagnation, heavily trapping pollutants in Delhi.
        df["stagnation_index"] = 1.0 / (df["windspeed_kph"] + 0.5)
        # Magnus-Tetens approximation for dew point
        df["dew_point"] = df["temp_c"] - ((100.0 - df["humidity"]) / 5.0)
        # Vehicular / fossil combustion interaction
        df["co_no2_prod"] = (df["co"] * df["no2"]) / 1000.0

        # Cyclical continuous time encoding (ensures 23:00 wraps smoothly to 00:00)
        df["sin_hour"] = np.sin(2.0 * np.pi * df["hour"] / 24.0)
        df["cos_hour"] = np.cos(2.0 * np.pi * df["hour"] / 24.0)
        df["sin_month"] = np.sin(2.0 * np.pi * (df["month"] - 1.0) / 12.0)
        df["cos_month"] = np.cos(2.0 * np.pi * (df["month"] - 1.0) / 12.0)
        df["sin_dow"] = np.sin(2.0 * np.pi * df["day_of_week"] / 7.0)
        df["cos_dow"] = np.cos(2.0 * np.pi * df["day_of_week"] / 7.0)

        # Delhi-specific seasonal and temporal behavioral flags
        # Winter months (Nov, Dec, Jan, Feb) experience severe thermal inversion and crop burning smog
        df["is_winter_smog"] = df["month"].isin([11, 12, 1, 2]).astype(float)
        # Peak traffic hours in Delhi: 08:00-11:00 and 17:00-21:00
        df["is_rush_hour"] = df["hour"].isin([8, 9, 10, 17, 18, 19, 20]).astype(float)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(float)

        return df


# List of engineered numeric columns after AQIFeatureEngineer is applied
ENGINEERED_NUMERIC_COLUMNS = [
    "temp_c", "humidity", "pressure_mb", "windspeed_kph",
    "pm2_5", "pm10", "co", "no2", "hour", "day", "month", "day_of_week",
    "pm_ratio", "pm_total", "pm_diff", "stagnation_index",
    "dew_point", "co_no2_prod", "sin_hour", "cos_hour",
    "sin_month", "cos_month", "sin_dow", "cos_dow",
    "is_winter_smog", "is_rush_hour", "is_weekend",
]


def build_preprocessor():
    """Builds a leakage-safe pipeline that engineers features, imputes, scales, and encodes."""
    column_transform = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]), ENGINEERED_NUMERIC_COLUMNS),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]), ["location"]),
    ])

    return Pipeline([
        ("feature_engineer", AQIFeatureEngineer()),
        ("col_transform", column_transform),
    ])


def build_features(df):
    """Return raw feature columns and target series."""
    return df[BASE_FEATURE_COLUMNS].copy(), df[TARGET_COLUMN].copy()


def prepare_data(data_path="data/processed/aqi_cleaned.csv", test_size=0.2, random_state=42):
    """Split raw features before fitting preprocessing to prevent data leakage."""
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(data_path)
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    preprocessor = build_preprocessor()
    return preprocessor.fit_transform(X_train), preprocessor.transform(X_test), y_train, y_test, preprocessor
