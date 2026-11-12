# VayuDrishti (वायुदृष्टि): Delhi Air Quality Intelligence & AQI Prediction Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost-orange.svg)](https://scikit-learn.org/)
[![Streamlit App](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![Standard](https://img.shields.io/badge/AQI%20Standard-CPCB%20India-green.svg)](https://cpcb.nic.in/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**VayuDrishti** (*Sanskrit: Vāyu = Air/Wind, Dṛṣṭi = Vision/Insight*) is an end-to-end, production-ready machine learning and environmental intelligence platform designed to predict the **Air Quality Index (AQI)** for the National Capital Region of Delhi.

Using historical hourly environmental records across Delhi's major monitoring stations, VayuDrishti combines **atmospheric physics feature engineering**, **cross-validated ensemble regression models**, **CPCB category classification**, and a **Streamlit geospatial dashboard** with health advisories and policy simulation tools.

---

## 📌 Project Highlights

- **Domain-Specific Atmospheric Physics:** Engineered features including the **Atmospheric Stagnation Index**, **Particulate Burden** ($PM_{2.5} + PM_{10}$), **Combustion Ratio** ($PM_{2.5} / PM_{10}$), **Dew Point approximation**, vehicular interaction proxies, and **cyclical continuous trigonometric time encodings** ($\sin/\cos$).
- **Multi-Model Rigorous Benchmarking:** Evaluates **Linear Regression**, **Random Forest**, and **XGBoost** under strict 5-fold cross-validation and held-out test splits.
- **Top Predictive Performance:** The production Random Forest pipeline achieves an **$R^2$ of 0.9556** and held-out **MAE of 23.53 AQI points** ($R^2 = 0.9714$ for XGBoost).
- **Automated CPCB Health Advisory:** Real-time mapping to Central Pollution Control Board (CPCB) categories (Good, Satisfactory, Moderate, Poor, Very Poor, Severe) with tailored precautions for vulnerable groups, N95 mask advice, and ventilation guidance.
- **Geospatial & Interactive Dashboard:** Multi-tab Streamlit web application featuring a PyDeck 3D station map, dynamic AQI gauge, what-if sensitivity simulator, diurnal trend curves, and ML explainability.
- **Clean Leakage-Safe Architecture:** Full feature transformations, fold-local median imputations, and scalers are bundled directly into the serialized Scikit-learn Pipeline.

---

## 🏛️ System Architecture

```text
               Raw Delhi Environmental Dataset (52,560 Hourly Records)
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │ Data Preprocessing & Validation       │
                      │ (Timestamp parsing, Deduplication)    │
                      └──────────────────┬────────────────────┘
                                         │
                                         ▼
                      ┌───────────────────────────────────────┐
                      │ Domain-Grounded Feature Engineering   │
                      │ - Stagnation Index (1 / (Wind + 0.5)) │
                      │ - Particulate Burden (PM2.5 + PM10)   │
                      │ - Dew Point & Winter Inversion Flags  │
                      │ - Cyclical Sin/Cos Hour & Month       │
                      └──────────────────┬────────────────────┘
                                         │
                                         ▼
                      ┌───────────────────────────────────────┐
                      │ Leakage-Safe ColumnTransformer        │
                      │ (Median Imputation, Scaler, OHE)      │
                      └──────────────────┬────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
          ┌───────────────────┐┌───────────────────┐┌───────────────────┐
          │ Linear Regression ││   Random Forest   ││      XGBoost      │
          │ (Baseline Model)  ││ (Ensemble Winner) ││ (Gradient Boost) │
          └───────────────────┘└─────────┬─────────┘└───────────────────┘
                                         │
                                         ▼
                      ┌───────────────────────────────────────┐
                      │ Model Selection & Serialization       │
                      │ (5-Fold CV MAE Selection, Compressed) │
                      └──────────────────┬────────────────────┘
                                         │
                                         ▼
                 ┌─────────────────────────────────────────────────┐
                 │          VayuDrishti Streamlit Platform         │
                 │  - Real-Time AQI & CPCB Health Advisories       │
                 │  - What-If Sensitivity Policy Simulator        │
                 │  - Geospatial Monitoring Station Map (PyDeck)   │
                 │  - Feature Importance & Model Diagnostics       │
                 └─────────────────────────────────────────────────┘
```

---

## 📊 Dataset Profile

The dataset consists of **52,560 hourly records** collected across 2025 from 6 continuous ambient air quality monitoring stations in Delhi:

| Station Name | Microclimate Profile | Coordinates | Records |
|---|---|---|---:|
| **Anand Vihar** | Interstate Bus Terminal, Heavy Traffic Corridor | 28.6469° N, 77.3160° E | 8,760 |
| **Connaught Place** | Commercial Center, High Vehicular Density | 28.6304° N, 77.2177° E | 8,760 |
| **Dwarka** | Residential Sector, Open Aerodynamic Corridor | 28.5882° N, 77.0494° E | 8,760 |
| **IGI Airport** | Aviation Hub, Highway Intersection | 28.5562° N, 77.1000° E | 8,760 |
| **Okhla Phase III** | Heavy Industrial Area, High Particulate Load | 28.5273° N, 77.2618° E | 8,760 |
| **Rohini** | Suburban Residential & Freight Corridor | 28.7041° N, 77.1025° E | 8,760 |

### Tracked Variables
- **Particulates:** $PM_{2.5}$ ($\mu g/m^3$), $PM_{10}$ ($\mu g/m^3$)
- **Gaseous Pollutants:** $CO$ ($\mu g/m^3$), $NO_2$ ($\mu g/m^3$)
- **Meteorological Conditions:** Ambient Temperature (°C), Relative Humidity (%), Barometric Pressure (mb), Wind Speed (km/h)
- **Target Variable:** `aqi_index` (Continuous AQI scale: 0 to 500+)

---

## 🔬 Feature Engineering & Mathematical Formulations

To empower the models beyond simple raw values, domain physics were incorporated:

1. **Particulate Total Mass:**
   $$\text{PM}_{\text{total}} = \text{PM}_{2.5} + \text{PM}_{10}$$
   *Accounts for >60% of overall predictive importance.*
2. **Fine-to-Coarse Particulate Ratio:**
   $$\text{PM}_{\text{ratio}} = \frac{\text{PM}_{2.5}}{\text{PM}_{10} + 10^{-4}}$$
   *Distinguishes high-toxicity combustion soot from road dust.*
3. **Atmospheric Stagnation Index:**
   $$\text{Stagnation} = \frac{1}{\text{Windspeed} + 0.5}$$
   *Models boundary-layer trapping of pollutants during calm winds.*
4. **Dew Point & Condensation Proxy:**
   $$T_{\text{dew}} \approx T - \left(\frac{100 - \text{RH}}{5}\right)$$
   *Critical for capturing nocturnal winter smog and fog formation.*
5. **Continuous Cyclical Time Encoding:**
   $$\sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \sin\left(\frac{2\pi (\text{month}-1)}{12}\right), \quad \cos\left(\frac{2\pi (\text{month}-1)}{12}\right)$$
   *Preserves continuity between 23:00 and 00:00 and December to January.*

---

## 🏆 Model Benchmarks & Comparison

Models were evaluated across **42,048 training records** with **5-fold cross-validation** and verified against an untouched **held-out test split (10,512 records)**:

| Model Architecture | 5-Fold CV MAE | CV RMSE | 5-Fold CV $R^2$ | Held-Out Test MAE | Test RMSE | Held-Out Test $R^2$ |
|---|---:|---:|---:|---:|---:|---:|
| **Random Forest (Winner)** | **26.96 ± 0.62** | 76.51 | **0.9446** | **23.53** | 68.99 | **0.9556** |
| **XGBoost Regressor** | **29.51 ± 0.34** | 61.20 | **0.9646** | **27.68** | **55.36** | **0.9714** |
| **Linear Regression** | 96.06 ± 0.95 | 176.40 | 0.7146 | 95.33 | 172.86 | 0.7213 |

> **Top Predictive Drivers:** `pm_total` (60.5%), `pm10` (7.7%), `day` (5.2%), `dew_point` (3.7%), `sin_month` (2.4%), `pm2_5` (2.4%), `humidity` (1.9%).

---

## 🏥 CPCB Health Advisory Framework

Predictions are classified using the official Indian CPCB National Air Quality Index (NAQI) scale:

| AQI Range | Category | Color Code | Precautionary Advisory & Public Health Guidance |
|---|---|---|---|
| **0 – 50** | **Good** | Green | Air quality is satisfactory; clean outdoor exercise conditions. |
| **51 – 100** | **Satisfactory** | Light Green | Minor breathing discomfort to unusually sensitive individuals. |
| **101 – 200** | **Moderate** | Amber | Breathing discomfort for asthma/heart patients; reduce intense outdoor sports. |
| **201 – 300** | **Poor** | Red | Breathing discomfort to most people on prolonged exposure; wear N95 mask outdoors. |
| **301 – 400** | **Very Poor** | Purple | Triggers respiratory illness; avoid outdoor activity; run indoor HEPA purifiers. |
| **401 – 500+** | **Severe** | Maroon | Health emergency; mandatory N95/N99 respirators; all outdoor activity suspended. |

---

## 📁 Repository Structure

```text
aqi-predictor/
├── app.py                      # Comprehensive multi-tab Streamlit web application
├── requirements.txt            # Production dependencies
├── README.md                   # Complete architectural and usage documentation
│
├── data/
│   ├── raw/
│   │   └── aqi.csv             # 52,560 hourly records from 6 Delhi stations
│   └── processed/
│       └── aqi_cleaned.csv     # Cleaned dataset with time components & coordinates
│
├── src/
│   ├── __init__.py             # Package marker & namespace resolver
│   ├── preprocess.py          # Data ingestion, validation, and cleaning
│   ├── features.py            # AQIFeatureEngineer & ColumnTransformer pipeline
│   ├── train.py               # 5-fold CV, benchmarking, feature importance & serialization
│   ├── predict.py             # Inference API, CPCB classification & health advisory logic
│   └── analyze.py             # Exploratory data analysis, rank correlation & visualizations
│
├── models/
│   ├── best_model.joblib       # Compressed production Random Forest pipeline (~65MB)
│   └── linear_regression.joblib# Baseline model comparison pipeline
│
├── results/
│   ├── model_comparison.csv   # Tabular cross-validation and test benchmark metrics
│   ├── model_metrics.json     # Full machine learning audit report
│   ├── feature_importance.csv # Relative weights of environmental predictors
│   ├── eda_report.md          # Comprehensive statistical findings
│   └── figures/
│       ├── model_validation.png   # Observed vs Predicted & Residuals scatter
│       ├── model_comparison.png   # CV MAE bar chart with error bounds
│       ├── model_cv_r2.png        # 5-Fold R² stability boxplot
│       ├── feature_importance.png # Top 15 predictive environmental factors
│       ├── eda_overview.png       # 4-Panel distribution, station variance & heatmap
│       └── seasonal_trends.png    # Monthly seasonal AQI trajectory
│
└── tests/
    └── test_pipeline.py       # Automated unit & integration tests
```

---

## 🚀 Quick Start Guide

### 1. Clone & Set Up Environment

```bash
git clone <repository_url>
cd aqi-predictor
python -m venv .venv
```

Activate the environment:
- **Windows:** `.venv\Scripts\activate`
- **Linux / macOS:** `source .venv/bin/activate`

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Preprocessing & Exploratory Analysis

```bash
# Clean raw records
python src/preprocess.py

# Run EDA and generate visualization charts
python src/analyze.py
```

### 3. Train Models & Generate Benchmarks

```bash
python src/train.py
```

### 4. Run Automated Test Suite

```bash
python tests/test_pipeline.py
```

### 5. Launch Interactive Web Dashboard

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## 🔮 Programmatic Inference Example

```python
import pandas as pd
from src.predict import predict_with_advisory

sample_input = {
    "location": "Anand Vihar",
    "temp_c": 14.5,
    "humidity": 78.0,
    "pressure_mb": 1018.0,
    "windspeed_kph": 3.2,
    "pm2_5": 210.0,
    "pm10": 390.0,
    "co": 2100.0,
    "no2": 65.0,
    "hour": 8,
    "day": 15,
    "month": 12,
    "day_of_week": 1,
}

result = predict_with_advisory(sample_input)
print(f"Predicted AQI: {result['predicted_aqi']} ({result['category']})")
print(f"Dominant Pollutant: {result['dominant_pollutant']['name']}")
print(f"Mask Advice: {result['advisory']['mask']}")
```

---

## 📜 License

This project is licensed under the MIT License. Developed for research, academic, and public health environmental intelligence purposes.
