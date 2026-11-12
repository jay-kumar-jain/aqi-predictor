"""Lightweight, split-safe dataset cleanup for AQI modeling."""
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "aqi.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "aqi_cleaned.csv"

BASE_FEATURE_COLUMNS = [
    "location", "temp_c", "humidity", "pressure_mb", "windspeed_kph",
    "pm2_5", "pm10", "co", "no2", "hour", "day", "month", "day_of_week",
]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS
TARGET_COLUMN = "aqi_index"


def preprocess_data(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    """Clean the raw air quality records, extract time attributes, and prepare for modeling.

    Missing feature values remain intact so imputers can be fitted strictly on training folds.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    df = pd.read_csv(input_path)
    original_rows = len(df)
    df = df.drop_duplicates().copy()

    # Parse IST date and time
    datetime = pd.to_datetime(
        df["date_ist"].astype("string") + " " + df["time_ist"].astype("string"),
        errors="coerce", dayfirst=True,
    )
    df["hour"] = datetime.dt.hour
    df["day"] = datetime.dt.day
    df["month"] = datetime.dt.month
    df["day_of_week"] = datetime.dt.dayofweek

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN, "hour", "day", "month", "day_of_week"])

    for col in BASE_FEATURE_COLUMNS:
        if col != "location":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["location"] = df["location"].astype("string").str.strip().replace("", pd.NA)

    # Retain location coordinates if present for mapping
    save_cols = BASE_FEATURE_COLUMNS + [TARGET_COLUMN]
    if "lat" in df.columns and "lon" in df.columns:
        save_cols = ["lat", "lon"] + save_cols

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[save_cols].to_csv(output_path, index=False)
    print(f"Preprocessed: {original_rows:,} -> {len(df):,} rows.")
    print(f"Feature missing cells retained for fold-local imputation: {int(df[BASE_FEATURE_COLUMNS].isna().sum().sum()):,}")
    print(f"Saved cleaned dataset: {output_path}")
    return df


if __name__ == "__main__":
    preprocess_data()
