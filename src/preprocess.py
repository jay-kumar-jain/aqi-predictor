import pandas as pd


INPUT_PATH = "aqi-predictor/data/raw/aqi.csv"
OUTPUT_PATH = "aqi-predictor/data/processed/aqi_cleaned.csv"


def preprocess_data(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    # Load data
    df = pd.read_csv(input_path)

    print("Original shape:", df.shape)

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Convert date and time
    df["datetime"] = pd.to_datetime(
        df["date_ist"] + " " + df["time_ist"],
        errors="coerce"
    )

    # Remove rows where datetime could not be converted
    df = df.dropna(subset=["datetime"])

    # Create time-based features
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.dayofweek

    # Columns we want for ML
    feature_columns = [
        "location",
        "temp_c",
        "humidity",
        "pressure_mb",
        "windspeed_kph",
        "pm2_5",
        "pm10",
        "co",
        "no2",
        "hour",
        "day",
        "month",
        "day_of_week",
        "aqi_index"
    ]

    df = df[feature_columns]

    # Handle missing values
    numeric_columns = df.select_dtypes(include=["number"]).columns

    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].median())

    # Remove rows where target is missing
    df = df.dropna(subset=["aqi_index"])

    # Save processed dataset
    df.to_csv(output_path, index=False)

    print("Processed shape:", df.shape)
    print("Saved to:", output_path)

    return df


if __name__ == "__main__":
    preprocess_data()