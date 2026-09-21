import pandas as pd


def load_raw_data(path="data/raw/aqi.csv"):
    return pd.read_csv(path)


def load_processed_data(path="data/processed/aqi_cleaned.csv"):
    return pd.read_csv(path)