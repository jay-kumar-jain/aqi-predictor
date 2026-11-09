"""Prediction helpers for the saved end-to-end pipeline."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

try:  # Supports both `python -m src.predict` and `python src/predict.py`.
    from .preprocess import FEATURE_COLUMNS
except ImportError:
    from preprocess import FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parent.parent
BEST_MODEL_PATH = ROOT / "models" / "best_model.joblib"


def load_model(path=BEST_MODEL_PATH):
    """Load current pipeline and replace legacy bare-estimator artifacts."""
    path = Path(path)
    model = joblib.load(path) if path.exists() else None
    if not isinstance(model, Pipeline) or "preprocess" not in model.named_steps:
        # Legacy models expected manually encoded columns and cannot safely
        # predict from the raw feature table. Retrain and save a compatible model.
        try:
            from .train import run_training
        except ImportError:
            from train import run_training
        run_training(make_plots=False)
        model = joblib.load(path)
    return model


def predict_sample(sample_df, model=None):
    """Predict from unencoded source columns; the pipeline handles preprocessing."""
    model = model or load_model()
    missing = sorted(set(FEATURE_COLUMNS) - set(sample_df.columns))
    if missing:
        raise ValueError(f"Prediction input is missing required feature columns: {', '.join(missing)}")
    return np.maximum(model.predict(sample_df[FEATURE_COLUMNS]), 0.0)


if __name__ == "__main__":
    data = pd.read_csv(ROOT / "data" / "processed" / "aqi_cleaned.csv")
    model = load_model()
    sample = data.drop(columns=["aqi_index"]).iloc[:3]
    for prediction, actual in zip(predict_sample(sample, model), data["aqi_index"].iloc[:3]):
        print(f"Predicted AQI: {prediction:.1f} | Actual AQI: {actual:.1f}")
