"""Small Streamlit interface for the selected AQI prediction pipeline."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import load_model, predict_sample


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="AQI Predictor", page_icon="🌫️", layout="centered")
st.title("AQI Predictor")
st.write("Estimate AQI from current air-quality and weather measurements.")
st.caption("This estimates AQI from supplied pollutant readings; it is not a future AQI forecast.")

processed_path = ROOT / "data" / "processed" / "aqi_cleaned.csv"
if processed_path.exists():
    reference = pd.read_csv(processed_path)
    locations = sorted(reference["location"].dropna().unique().tolist())
    medians = reference.select_dtypes(include="number").median()
else:
    locations = ["Anand Vihar", "Connaught Place", "Dwarka", "IGI Airport", "Okhla Phase III", "Rohini"]
    medians = pd.Series(dtype=float)

def default_value(name, fallback):
    value = medians.get(name, fallback)
    return fallback if pd.isna(value) else float(value)


with st.form("aqi_input_form"):
    location = st.selectbox("Monitoring location", locations)
    weather, pollutants, time = st.columns(3)
    with weather:
        st.subheader("Weather")
        temp_c = st.number_input("Temperature (°C)", value=default_value("temp_c", 25.0), step=0.5)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0,
                                   value=default_value("humidity", 60.0), step=1.0)
        pressure_mb = st.number_input("Pressure (mb)", min_value=0.0,
                                      value=default_value("pressure_mb", 1010.0), step=0.5)
        windspeed_kph = st.number_input("Wind speed (km/h)", min_value=0.0,
                                        value=default_value("windspeed_kph", 5.0), step=0.5)
    with pollutants:
        st.subheader("Pollutants")
        pm2_5 = st.number_input("PM2.5", min_value=0.0, value=default_value("pm2_5", 50.0), step=1.0)
        pm10 = st.number_input("PM10", min_value=0.0, value=default_value("pm10", 100.0), step=1.0)
        co = st.number_input("CO", min_value=0.0, value=default_value("co", 500.0), step=10.0)
        no2 = st.number_input("NO₂", min_value=0.0, value=default_value("no2", 30.0), step=1.0)
    with time:
        st.subheader("Time")
        hour = st.slider("Hour", min_value=0, max_value=23, value=12)
        day = st.slider("Day of month", min_value=1, max_value=31, value=15)
        month = st.slider("Month", min_value=1, max_value=12, value=6)
        day_of_week = st.selectbox("Day of week", options=list(range(7)),
                                   format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x])
    submitted = st.form_submit_button("Predict AQI", type="primary", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "location": location, "temp_c": temp_c, "humidity": humidity,
        "pressure_mb": pressure_mb, "windspeed_kph": windspeed_kph,
        "pm2_5": pm2_5, "pm10": pm10, "co": co, "no2": no2,
        "hour": hour, "day": day, "month": month, "day_of_week": day_of_week,
    }])
    try:
        prediction = float(predict_sample(row, load_model())[0])
        st.metric("Predicted AQI", f"{prediction:.1f}")
        metrics_path = ROOT / "results" / "model_metrics.json"
        if metrics_path.exists():
            report = json.loads(metrics_path.read_text(encoding="utf-8"))
            st.caption(f"Selected model: {report.get('best_model', 'trained pipeline')} · "
                       f"held-out MAE: {report.get('model_comparison', [{}])[0].get('test', {}).get('mae', float('nan')):.1f} AQI points")
    except (FileNotFoundError, ValueError, ImportError) as exc:
        st.error(f"Prediction could not be made: {exc}")
        st.info("Train the models first with `python src/train.py`, then restart the app.")
