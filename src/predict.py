"""Prediction pipeline and health advisory system for Delhi AQI."""
import sys
from pathlib import Path
from typing import Dict, Any, Union

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

try:
    from . import features
    sys.modules["features"] = features
except Exception:
    try:
        import src.features as features
        sys.modules["features"] = features
    except Exception:
        pass

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

try:  # Supports both `python -m src.predict` and `python src/predict.py`.
    from .preprocess import BASE_FEATURE_COLUMNS
except ImportError:
    from preprocess import BASE_FEATURE_COLUMNS

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS
ROOT = Path(__file__).resolve().parent.parent
BEST_MODEL_PATH = ROOT / "models" / "best_model.joblib"

# CPCB (Central Pollution Control Board, India) AQI Categories & Standards
CPCB_CATEGORIES = [
    {
        "min": 0, "max": 50,
        "category": "Good",
        "color": "#00b050",
        "description": "Minimal health impact. Air quality is considered satisfactory.",
        "mask": "Not required",
        "outdoor": "Ideal for outdoor activities, jogging, and sports.",
        "indoor": "Windows can remain open for fresh air ventilation.",
        "sensitive": "No precautions necessary for sensitive individuals.",
    },
    {
        "min": 51, "max": 100,
        "category": "Satisfactory",
        "color": "#92d050",
        "description": "Minor breathing discomfort to sensitive individuals.",
        "mask": "Optional for unusually sensitive individuals.",
        "outdoor": "Safe for general outdoor recreation.",
        "indoor": "Normal ventilation is acceptable.",
        "sensitive": "People with severe respiratory issues should monitor symptoms.",
    },
    {
        "min": 101, "max": 200,
        "category": "Moderate",
        "color": "#ffc000",
        "description": "Breathing discomfort to people with asthma, lung, and heart diseases.",
        "mask": "Recommended for sensitive individuals in busy traffic zones.",
        "outdoor": "Reduce prolonged or heavy outdoor exertion.",
        "indoor": "Keep windows closed during peak traffic hours.",
        "sensitive": "Children, elderly, and asthma patients should take frequent breaks.",
    },
    {
        "min": 201, "max": 300,
        "category": "Poor",
        "color": "#ff0000",
        "description": "Breathing discomfort to most people on prolonged outdoor exposure.",
        "mask": "N95/KN95 respirator mask recommended outdoors.",
        "outdoor": "Avoid prolonged outdoor exertion; avoid morning jogs during smog.",
        "indoor": "Use HEPA air purifiers indoors and seal window gaps.",
        "sensitive": "Sensitive groups should remain indoors with clean air filtration.",
    },
    {
        "min": 301, "max": 400,
        "category": "Very Poor",
        "color": "#7030a0",
        "description": "Respiratory illness on prolonged exposure; triggers asthma attacks.",
        "mask": "Strictly wear well-fitted N95 masks when stepping outdoors.",
        "outdoor": "Strictly avoid strenuous outdoor activities and sports.",
        "indoor": "Run indoor air purifiers on high; keep doors and windows shut.",
        "sensitive": "High risk for children, elderly, and cardiopulmonary patients.",
    },
    {
        "min": 401, "max": 5000,
        "category": "Severe / Hazardous",
        "color": "#800000",
        "description": "Affects healthy individuals and severely impacts those with pre-existing conditions.",
        "mask": "Mandatory N95/N99 respirator outdoors. Limit stepping outside entirely.",
        "outdoor": "Health emergency level: cancel all outdoor activities.",
        "indoor": "Operate high-efficiency HEPA purifiers continuously; avoid cooking smoke.",
        "sensitive": "Emergency alert: high risk of acute respiratory and cardiovascular events.",
    },
]


def get_cpcb_category(aqi_val: float) -> Dict[str, Any]:
    """Classifies an AQI value according to India's CPCB standard."""
    val = float(aqi_val)
    for cat in CPCB_CATEGORIES:
        if cat["min"] <= val <= cat["max"]:
            return {
                "aqi": val,
                "category": cat["category"],
                "color": cat["color"],
                "description": cat["description"],
                "mask_advice": cat["mask"],
                "outdoor_advice": cat["outdoor"],
                "indoor_advice": cat["indoor"],
                "sensitive_advice": cat["sensitive"],
            }
    # Fallback for extreme cases
    return {
        "aqi": val,
        "category": "Severe / Hazardous",
        "color": "#800000",
        "description": "Emergency level pollution.",
        "mask_advice": "Mandatory N95/N99",
        "outdoor_advice": "Stay indoors",
        "indoor_advice": "Run purifiers",
        "sensitive_advice": "High risk",
    }


def identify_dominant_pollutant(row: Union[pd.Series, dict]) -> Dict[str, Any]:
    """Estimates dominant pollutant driving environmental risk based on normalized thresholds."""
    # Standard 24h/1h benchmark safe thresholds (CPCB)
    # PM2.5: 60 ug/m3, PM10: 100 ug/m3, NO2: 80 ug/m3, CO: 2000 ug/m3 (2 mg/m3)
    standards = {
        "pm2_5": {"name": "PM2.5 (Fine Particulate)", "threshold": 60.0},
        "pm10": {"name": "PM10 (Coarse Particulate)", "threshold": 100.0},
        "no2": {"name": "NO₂ (Nitrogen Dioxide)", "threshold": 80.0},
        "co": {"name": "CO (Carbon Monoxide)", "threshold": 2000.0},
    }
    ratios = {}
    for pol, info in standards.items():
        val = float(row.get(pol, 0.0) if isinstance(row, dict) else row[pol])
        ratios[pol] = val / info["threshold"]

    dominant_key = max(ratios, key=ratios.get)
    return {
        "key": dominant_key,
        "name": standards[dominant_key]["name"],
        "ratio_to_safe_limit": round(ratios[dominant_key], 2),
    }


def load_model(path=BEST_MODEL_PATH):
    """Load trained pipeline from disk or trigger training if not present."""
    path = Path(path)
    if not path.exists():
        try:
            from .train import run_training
        except ImportError:
            from train import run_training
        run_training(make_plots=False)
    return joblib.load(path)


def predict_sample(sample_df: pd.DataFrame, model=None) -> np.ndarray:
    """Predict AQI from raw feature columns; pipeline handles all engineering and transformations."""
    model = model or load_model()
    missing = sorted(set(FEATURE_COLUMNS) - set(sample_df.columns))
    if missing:
        raise ValueError(f"Input is missing required features: {', '.join(missing)}")
    return np.maximum(model.predict(sample_df[FEATURE_COLUMNS]), 0.0)


def predict_with_advisory(sample_input: Union[pd.DataFrame, dict], model=None) -> Dict[str, Any]:
    """Generates AQI prediction alongside CPCB category, health advisory, and dominant pollutant."""
    if isinstance(sample_input, dict):
        df = pd.DataFrame([sample_input])
    else:
        df = sample_input.copy()

    preds = predict_sample(df, model)
    predicted_aqi = float(preds[0])
    category_info = get_cpcb_category(predicted_aqi)
    dominant_info = identify_dominant_pollutant(df.iloc[0])

    return {
        "predicted_aqi": round(predicted_aqi, 1),
        "category": category_info["category"],
        "color": category_info["color"],
        "description": category_info["description"],
        "advisory": {
            "mask": category_info["mask_advice"],
            "outdoor": category_info["outdoor_advice"],
            "indoor": category_info["indoor_advice"],
            "sensitive": category_info["sensitive_advice"],
        },
        "dominant_pollutant": dominant_info,
    }


if __name__ == "__main__":
    test_path = ROOT / "data" / "processed" / "aqi_cleaned.csv"
    if test_path.exists():
        data = pd.read_csv(test_path)
        sample = data.iloc[0].to_dict()
        res = predict_with_advisory(sample)
        print(f"Sample Prediction Test:")
        print(f"Location: {sample.get('location')} | Actual AQI: {sample.get('aqi_index')}")
        print(f"Predicted AQI: {res['predicted_aqi']} ({res['category']})")
        print(f"Dominant Pollutant: {res['dominant_pollutant']['name']} ({res['dominant_pollutant']['ratio_to_safe_limit']}x limit)")
        print(f"Mask: {res['advisory']['mask']}")
