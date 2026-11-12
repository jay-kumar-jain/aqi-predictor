"""Automated Unit and Integration Tests for VayuDrishti AQI Pipeline."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import unittest
import pandas as pd
import numpy as np

from src.features import AQIFeatureEngineer, build_preprocessor, BASE_FEATURE_COLUMNS
from src.predict import (
    get_cpcb_category,
    identify_dominant_pollutant,
    predict_sample,
    predict_with_advisory,
    load_model,
)

ROOT = Path(__file__).resolve().parent.parent


def test_cpcb_category_boundaries():
    assert get_cpcb_category(35)["category"] == "Good"
    assert get_cpcb_category(50)["category"] == "Good"
    assert get_cpcb_category(75)["category"] == "Satisfactory"
    assert get_cpcb_category(100)["category"] == "Satisfactory"
    assert get_cpcb_category(150)["category"] == "Moderate"
    assert get_cpcb_category(200)["category"] == "Moderate"
    assert get_cpcb_category(250)["category"] == "Poor"
    assert get_cpcb_category(300)["category"] == "Poor"
    assert get_cpcb_category(350)["category"] == "Very Poor"
    assert get_cpcb_category(400)["category"] == "Very Poor"
    assert get_cpcb_category(450)["category"] == "Severe / Hazardous"


def test_dominant_pollutant_identification():
    # Case 1: PM2.5 heavily elevated relative to standard (60 ug/m3)
    row_pm25 = {"pm2_5": 240.0, "pm10": 110.0, "no2": 30.0, "co": 800.0}
    dom_pm25 = identify_dominant_pollutant(row_pm25)
    assert dom_pm25["key"] == "pm2_5"
    assert dom_pm25["ratio_to_safe_limit"] == 4.0

    # Case 2: PM10 dominant relative to standard (100 ug/m3)
    row_pm10 = {"pm2_5": 30.0, "pm10": 500.0, "no2": 40.0, "co": 600.0}
    dom_pm10 = identify_dominant_pollutant(row_pm10)
    assert dom_pm10["key"] == "pm10"
    assert dom_pm10["ratio_to_safe_limit"] == 5.0


def test_feature_engineer_transform():
    fe = AQIFeatureEngineer()
    df_sample = pd.DataFrame([{
        "location": "Anand Vihar",
        "temp_c": 20.0, "humidity": 70.0, "pressure_mb": 1012.0, "windspeed_kph": 3.0,
        "pm2_5": 120.0, "pm10": 200.0, "co": 1500.0, "no2": 45.0,
        "hour": 8, "day": 10, "month": 11, "day_of_week": 1,
    }])
    transformed = fe.transform(df_sample)

    assert "pm_total" in transformed.columns
    assert "pm_ratio" in transformed.columns
    assert "stagnation_index" in transformed.columns
    assert "dew_point" in transformed.columns
    assert "sin_hour" in transformed.columns
    assert "cos_hour" in transformed.columns
    assert "is_winter_smog" in transformed.columns
    assert transformed["is_winter_smog"].iloc[0] == 1.0  # November is winter smog season
    assert transformed["is_rush_hour"].iloc[0] == 1.0    # 8 AM is rush hour


def test_prediction_pipeline():
    model = load_model()
    df_sample = pd.DataFrame([{
        "location": "Connaught Place",
        "temp_c": 26.0, "humidity": 55.0, "pressure_mb": 1010.0, "windspeed_kph": 6.0,
        "pm2_5": 65.0, "pm10": 130.0, "co": 700.0, "no2": 35.0,
        "hour": 14, "day": 18, "month": 5, "day_of_week": 3,
    }])
    preds = predict_sample(df_sample, model)
    assert len(preds) == 1
    assert float(preds[0]) > 0.0

    advisory = predict_with_advisory(df_sample, model)
    assert "predicted_aqi" in advisory
    assert "category" in advisory
    assert "dominant_pollutant" in advisory
    assert "advisory" in advisory


if __name__ == "__main__":
    test_cpcb_category_boundaries()
    test_dominant_pollutant_identification()
    test_feature_engineer_transform()
    test_prediction_pipeline()
    print("✅ All tests passed successfully!")
