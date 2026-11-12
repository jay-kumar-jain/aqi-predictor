"""VayuDrishti: Delhi Air Quality Intelligence & Predictive Monitoring Platform.

End-to-end interactive Streamlit web application providing real-time AQI estimation,
CPCB health advisories, geospatial station mapping, sensitivity analysis, and ML benchmarks.
"""
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from src.predict import (
    BASE_FEATURE_COLUMNS,
    CPCB_CATEGORIES,
    get_cpcb_category,
    identify_dominant_pollutant,
    load_model,
    predict_sample,
)

# Page Configuration
st.set_page_config(
    page_title="VayuDrishti | Delhi AQI Intelligence",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
PROCESSED_DATA_PATH = ROOT / "data" / "processed" / "aqi_cleaned.csv"
RAW_DATA_PATH = ROOT / "data" / "raw" / "aqi.csv"

# Custom CSS for Modern Dashboard Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        border: 1px solid #eef2f6;
        text-align: center;
    }
    .advisory-box {
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        font-size: 0.95rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0px 0px;
        padding: 8px 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Station Coordinates Reference
STATION_METADATA = {
    "Anand Vihar": {"lat": 28.6469, "lon": 77.3160, "type": "Interstate Transport Hub & Heavy Traffic Zone"},
    "Connaught Place": {"lat": 28.6304, "lon": 77.2177, "type": "Central Commercial & High Density Area"},
    "Dwarka": {"lat": 28.5882, "lon": 77.0494, "type": "South-West Residential & Open Corridor"},
    "IGI Airport": {"lat": 28.5562, "lon": 77.1000, "type": "Aviation Hub & Highway Intersection"},
    "Okhla Phase III": {"lat": 28.5273, "lon": 77.2618, "type": "South-East Heavy Industrial Cluster"},
    "Rohini": {"lat": 28.7041, "lon": 77.1025, "type": "North-West Mixed Suburban & Freight Corridor"},
}

# Load Dataset & Baseline Statistics
@st.cache_data
def get_reference_data():
    path = PROCESSED_DATA_PATH if PROCESSED_DATA_PATH.exists() else RAW_DATA_PATH
    if path.exists():
        df = pd.read_csv(path)
        locations = sorted(df["location"].dropna().unique().tolist())
        medians = df.select_dtypes(include="number").median()
        return df, locations, medians
    fallback_locs = list(STATION_METADATA.keys())
    return None, fallback_locs, pd.Series(dtype=float)

df_ref, locations_list, medians_series = get_reference_data()

def get_med(col_name: str, fallback: float) -> float:
    val = medians_series.get(col_name, fallback)
    return fallback if pd.isna(val) else float(val)

# Load Model Pipeline
@st.cache_resource
def get_cached_pipeline():
    try:
        return load_model()
    except Exception as e:
        return None

pipeline = get_cached_pipeline()

# Load Model Metrics
@st.cache_data
def get_metrics_report():
    metrics_path = RESULTS_DIR / "model_metrics.json"
    if metrics_path.exists():
        try:
            return json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

metrics_report = get_metrics_report()

# -------------------------------------------------------------
# SIDEBAR CONTROLS & PRESET SCENARIOS
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/dust.png", width=64)
    st.title("VayuDrishti Controls")
    st.caption("Delhi Air Quality Intelligence Engine")

    st.markdown("---")
    st.subheader("⚡ Quick Environmental Presets")
    preset_choice = st.selectbox(
        "Select an Atmospheric Scenario:",
        [
            "Custom Manual Input",
            "🌿 Clean Monsoon Morning (July)",
            "🍂 Mild Autumn Afternoon (October)",
            "🚨 Severe Winter Smog Peak (December)",
            "💥 Post-Diwali Night Inversion (November)",
        ],
        index=0,
    )

    # Preset configurations
    presets = {
        "Custom Manual Input": None,
        "🌿 Clean Monsoon Morning (July)": {
            "temp_c": 28.0, "humidity": 88.0, "pressure_mb": 1002.0, "windspeed_kph": 14.0,
            "pm2_5": 22.0, "pm10": 45.0, "co": 350.0, "no2": 18.0,
            "hour": 8, "day": 20, "month": 7, "day_of_week": 1,
        },
        "🍂 Mild Autumn Afternoon (October)": {
            "temp_c": 30.5, "humidity": 52.0, "pressure_mb": 1012.0, "windspeed_kph": 6.5,
            "pm2_5": 58.0, "pm10": 115.0, "co": 750.0, "no2": 38.0,
            "hour": 14, "day": 12, "month": 10, "day_of_week": 3,
        },
        "🚨 Severe Winter Smog Peak (December)": {
            "temp_c": 11.0, "humidity": 85.0, "pressure_mb": 1019.5, "windspeed_kph": 2.0,
            "pm2_5": 285.0, "pm10": 480.0, "co": 2600.0, "no2": 88.0,
            "hour": 9, "day": 22, "month": 12, "day_of_week": 2,
        },
        "💥 Post-Diwali Night Inversion (November)": {
            "temp_c": 16.5, "humidity": 72.0, "pressure_mb": 1016.0, "windspeed_kph": 1.5,
            "pm2_5": 490.0, "pm10": 780.0, "co": 3400.0, "no2": 110.0,
            "hour": 23, "day": 14, "month": 11, "day_of_week": 6,
        },
    }

    st.markdown("---")
    st.subheader("🤖 Production Model Status")
    if metrics_report:
        best_m = metrics_report.get("best_model", "Random Forest")
        winner_data = next((m for m in metrics_report.get("model_comparison", []) if m["model"] == best_m), {})
        cv_mae = winner_data.get("cv", {}).get("mean_mae", 26.96)
        test_r2 = winner_data.get("test", {}).get("r2", 0.9556)
        st.success(f"**Active Model:** {best_m}")
        st.caption(f"5-Fold CV MAE: **{cv_mae:.2f}** AQI pts")
        st.caption(f"Held-Out Test R²: **{test_r2:.4f}**")
    else:
        st.info("Pipeline loaded and ready.")

    st.markdown("---")
    st.caption("CPCB (Central Pollution Control Board) NAQI Standards Reference.")

# -------------------------------------------------------------
# MAIN HEADER
# -------------------------------------------------------------
st.markdown('<div class="main-title">VayuDrishti (वायुदृष्टि)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Delhi Air Quality Intelligence, Real-Time Machine Learning Predictor & Health Advisory Engine</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# DASHBOARD TABS
# -------------------------------------------------------------
tab_pred, tab_map, tab_eda, tab_ml, tab_about = st.tabs([
    "🎯 AQI Predictor & Health Advisory",
    "🗺️ Delhi Station Geospatial Map",
    "📊 Exploratory Insights & Trends",
    "🤖 ML Models & Feature Importance",
    "📖 CPCB Standards & Methodology",
])

# =============================================================
# TAB 1: AQI PREDICTOR & HEALTH ADVISORY
# =============================================================
with tab_pred:
    preset_data = presets.get(preset_choice)

    with st.form("aqi_prediction_form"):
        st.subheader("1. Atmospheric & Environmental Parameters")

        loc_col, info_col = st.columns([1, 2])
        with loc_col:
            selected_location = st.selectbox("Monitoring Station:", locations_list, index=0)
        with info_col:
            loc_info = STATION_METADATA.get(selected_location, {})
            st.info(f"**Station Profile:** {loc_info.get('type', 'Delhi Urban Region')} "
                    f"(Lat: {loc_info.get('lat', 'N/A')}, Lon: {loc_info.get('lon', 'N/A')})")

        st.markdown("---")
        col_weather, col_pollutants, col_time = st.columns(3)

        with col_weather:
            st.markdown("#### ⛅ Meteorological Factors")
            temp_val = preset_data["temp_c"] if preset_data else get_med("temp_c", 25.0)
            hum_val = preset_data["humidity"] if preset_data else get_med("humidity", 60.0)
            pres_val = preset_data["pressure_mb"] if preset_data else get_med("pressure_mb", 1010.0)
            wind_val = preset_data["windspeed_kph"] if preset_data else get_med("windspeed_kph", 5.0)

            temp_c = st.number_input("Ambient Temperature (°C)", value=float(temp_val), step=0.5, format="%.1f")
            humidity = st.slider("Relative Humidity (%)", 0.0, 100.0, float(hum_val), step=1.0)
            pressure_mb = st.number_input("Barometric Pressure (mb)", min_value=950.0, max_value=1050.0,
                                          value=float(pres_val), step=0.5, format="%.1f")
            windspeed_kph = st.number_input("Surface Wind Speed (km/h)", min_value=0.0, max_value=80.0,
                                            value=float(wind_val), step=0.5, format="%.1f")

        with col_pollutants:
            st.markdown("#### 🏭 Pollutant Concentrations")
            pm25_val = preset_data["pm2_5"] if preset_data else get_med("pm2_5", 50.0)
            pm10_val = preset_data["pm10"] if preset_data else get_med("pm10", 100.0)
            co_val = preset_data["co"] if preset_data else get_med("co", 500.0)
            no2_val = preset_data["no2"] if preset_data else get_med("no2", 30.0)

            pm2_5 = st.number_input("PM2.5 (µg/m³) [Safe: 60]", min_value=0.0, max_value=1200.0,
                                    value=float(pm25_val), step=1.0)
            pm10 = st.number_input("PM10 (µg/m³) [Safe: 100]", min_value=0.0, max_value=2000.0,
                                   value=float(pm10_val), step=2.0)
            co = st.number_input("CO (µg/m³) [Safe: 2000]", min_value=0.0, max_value=25000.0,
                                 value=float(co_val), step=50.0)
            no2 = st.number_input("NO₂ (µg/m³) [Safe: 80]", min_value=0.0, max_value=500.0,
                                  value=float(no2_val), step=1.0)

        with col_time:
            st.markdown("#### 🕒 Temporal Context")
            hour_val = preset_data["hour"] if preset_data else 12
            day_val = preset_data["day"] if preset_data else 15
            month_val = preset_data["month"] if preset_data else 6
            dow_val = preset_data["day_of_week"] if preset_data else 2

            hour = st.slider("Hour of Day (IST 24h)", 0, 23, int(hour_val))
            month = st.slider("Month of Year", 1, 12, int(month_val),
                              format="%d")
            day = st.slider("Day of Month", 1, 31, int(day_val))
            day_of_week = st.selectbox(
                "Day of Week",
                options=list(range(7)),
                index=int(dow_val),
                format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
            )

        submit_btn = st.form_submit_button("🔮 Predict Delhi AQI & Generate Advisory", type="primary", use_container_width=True)

    # Prediction Execution
    input_sample = pd.DataFrame([{
        "location": selected_location,
        "temp_c": temp_c, "humidity": humidity, "pressure_mb": pressure_mb, "windspeed_kph": windspeed_kph,
        "pm2_5": pm2_5, "pm10": pm10, "co": co, "no2": no2,
        "hour": hour, "day": day, "month": month, "day_of_week": day_of_week,
    }])

    if submit_btn or preset_data is not None:
        if pipeline is None:
            st.error("Model pipeline artifact not found. Please train models using `python src/train.py`.")
        else:
            try:
                pred_aqi = float(predict_sample(input_sample, pipeline)[0])
                cpcb = get_cpcb_category(pred_aqi)
                dominant = identify_dominant_pollutant(input_sample.iloc[0])

                st.markdown("---")
                st.subheader("2. Intelligence & Assessment Results")

                # Metrics Row
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                with m_col1:
                    st.metric(label="Predicted AQI Index", value=f"{pred_aqi:.1f}")
                with m_col2:
                    st.markdown(f"""
                    <div style="background-color: {cpcb['color']}22; border-left: 5px solid {cpcb['color']}; padding: 10px; border-radius: 6px;">
                        <span style="font-size: 0.8rem; color: #555;">CPCB CATEGORY</span><br>
                        <strong style="font-size: 1.3rem; color: {cpcb['color']};">{cpcb['category']}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                with m_col3:
                    st.metric(
                        label="Dominant Pollutant",
                        value=dominant["name"].split(" ")[0],
                        delta=f"{dominant['ratio_to_safe_limit']}x Safe Threshold",
                        delta_color="inverse",
                    )
                with m_col4:
                    st.metric(
                        label="Station",
                        value=selected_location,
                        help=STATION_METADATA.get(selected_location, {}).get("type", "Delhi Region")
                    )

                # AQI Speedometer / Gauge Bar
                st.markdown("##### AQI Scale Indicator")
                progress_val = min(max(pred_aqi / 500.0, 0.0), 1.0)
                st.progress(progress_val)
                st.caption(f"Scale: 0 (Good) ── 100 (Satisfactory) ── 200 (Moderate) ── 300 (Poor) ── 400 (Very Poor) ── 500+ (Severe) | Current Level: **{pred_aqi:.1f}**")

                # Health Advisory Action Cards
                st.markdown("#### 🛡️ Public Health Action & Precautionary Advisory")
                adv_col1, adv_col2 = st.columns(2)

                with adv_col1:
                    st.markdown(f"""
                    <div class="advisory-box" style="background: #f0f7ff; border-left: 4px solid #0066cc;">
                        <strong>😷 Mask & Protection:</strong><br>{cpcb['mask_advice']}
                    </div>
                    <div class="advisory-box" style="background: #fbf0ff; border-left: 4px solid #7030a0;">
                        <strong>🏃 Outdoor Activities & Sports:</strong><br>{cpcb['outdoor_advice']}
                    </div>
                    """, unsafe_allow_html=True)

                with adv_col2:
                    st.markdown(f"""
                    <div class="advisory-box" style="background: #f2fff2; border-left: 4px solid #00b050;">
                        <strong>🏠 Indoor Air & Ventilation:</strong><br>{cpcb['indoor_advice']}
                    </div>
                    <div class="advisory-box" style="background: #fff8eb; border-left: 4px solid #ff9900;">
                        <strong>⚠️ Sensitive Populations (Elderly / Children / Asthma):</strong><br>{cpcb['sensitive_advice']}
                    </div>
                    """, unsafe_allow_html=True)

                # Interactive Sensitivity Simulation (What-If Analysis)
                with st.expander("🧪 What-If Scenario Simulator (Policy & Mitigation Impact)", expanded=False):
                    st.write("Examine how environmental changes or emission control interventions alter predicted AQI.")
                    sim_col1, sim_col2 = st.columns(2)
                    with sim_col1:
                        sim_wind_boost = st.slider("Simulate Wind Speed Boost (+km/h)", 0.0, 25.0, 5.0, step=1.0)
                    with sim_col2:
                        sim_pm_reduction = st.slider("Simulate Particulate Emission Reduction (%)", 0, 70, 25, step=5)

                    sim_sample = input_sample.copy()
                    sim_sample["windspeed_kph"] = sim_sample["windspeed_kph"] + sim_wind_boost
                    sim_sample["pm2_5"] = sim_sample["pm2_5"] * (1.0 - sim_pm_reduction / 100.0)
                    sim_sample["pm10"] = sim_sample["pm10"] * (1.0 - sim_pm_reduction / 100.0)

                    sim_pred = float(predict_sample(sim_sample, pipeline)[0])
                    sim_cpcb = get_cpcb_category(sim_pred)
                    delta_aqi = sim_pred - pred_aqi

                    sc1, sc2 = st.columns(2)
                    with sc1:
                        st.metric("Simulated AQI", f"{sim_pred:.1f}", delta=f"{delta_aqi:.1f} AQI pts", delta_color="inverse")
                    with sc2:
                        st.markdown(f"**Simulated Category:** <span style='color:{sim_cpcb['color']}; font-weight:bold;'>{sim_cpcb['category']}</span>", unsafe_allow_html=True)

            except Exception as ex:
                st.error(f"Inference error: {ex}")

# =============================================================
# TAB 2: GEOSPATIAL MAP
# =============================================================
with tab_map:
    st.subheader("Delhi Air Quality Monitoring Network (CPCB Stations)")
    st.write("Spatial distribution of monitoring stations across the National Capital Region (NCR).")

    # Prepare map dataframe
    map_rows = []
    for name, meta in STATION_METADATA.items():
        # Get station average AQI from dataset if available
        mean_aqi = 287.0
        if df_ref is not None and "location" in df_ref.columns:
            st_data = df_ref[df_ref["location"] == name]["aqi_index"]
            if len(st_data) > 0:
                mean_aqi = float(st_data.mean())

        cat_info = get_cpcb_category(mean_aqi)
        map_rows.append({
            "station": name,
            "lat": meta["lat"],
            "lon": meta["lon"],
            "type": meta["type"],
            "annual_mean_aqi": round(mean_aqi, 1),
            "category": cat_info["category"],
            "color": [220, 50, 50, 180] if mean_aqi > 250 else [50, 180, 50, 180],
        })
    map_df = pd.DataFrame(map_rows)

    view_state = pdk.ViewState(
        latitude=28.6139,
        longitude=77.2090,
        zoom=10.5,
        pitch=30,
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["lon", "lat"],
        get_color="color",
        get_radius=1800,
        pickable=True,
    )

    text_layer = pdk.Layer(
        "TextLayer",
        data=map_df,
        get_position=["lon", "lat"],
        get_text="station",
        get_size=13,
        get_color=[0, 0, 0, 240],
        get_alignment_baseline="'bottom'",
        get_pixel_offset=[0, -14],
    )

    r = pdk.Deck(
        layers=[layer, text_layer],
        initial_view_state=view_state,
        tooltip={"text": "{station}\nType: {type}\n2025 Annual Mean AQI: {annual_mean_aqi} ({category})"},
        map_style="light",
    )

    st.pydeck_chart(r)

    st.markdown("#### Station Profiles & Baseline Metrics")
    st.dataframe(
        map_df[["station", "type", "annual_mean_aqi", "category"]].rename(
            columns={"station": "Station Name", "type": "Microclimate Profile", "annual_mean_aqi": "2025 Mean AQI", "category": "CPCB Category"}
        ),
        use_container_width=True,
        hide_index=True,
    )

# =============================================================
# TAB 3: EXPLORATORY DATA ANALYSIS & TRENDS
# =============================================================
with tab_eda:
    st.subheader("Statistical Trends & Atmospheric Insights (52,560 Hourly Records)")

    eda_overview_path = RESULTS_DIR / "figures" / "eda_overview.png"
    seasonal_path = RESULTS_DIR / "figures" / "seasonal_trends.png"

    e_col1, e_col2 = st.columns(2)
    with e_col1:
        if eda_overview_path.exists():
            st.image(str(eda_overview_path), caption="Distribution, Location Variance & Correlation Analysis", use_container_width=True)
    with e_col2:
        if seasonal_path.exists():
            st.image(str(seasonal_path), caption="Monthly Seasonal Trajectory (Smog vs Monsoon Cleanse)", use_container_width=True)

    st.markdown("---")
    st.markdown("#### Diurnal (24-Hour) Pollution Cycle in Delhi")
    if df_ref is not None:
        hourly_df = df_ref.groupby("hour")["aqi_index"].agg(["mean", "median"]).reset_index()
        chart = alt.Chart(hourly_df).mark_line(point=True).encode(
            x=alt.X("hour:O", title="Hour of Day (IST 00:00 - 23:00)"),
            y=alt.Y("mean:Q", title="Mean AQI Index"),
            tooltip=["hour", "mean", "median"],
        ).properties(height=320)
        st.altair_chart(chart, use_container_width=True)
        st.caption("Notice the twin peaks during morning (08:00–10:00) and late evening (20:00–23:00) rush hours, driven by heavy vehicular emissions and lower nocturnal atmospheric boundary layer heights.")

# =============================================================
# TAB 4: ML MODEL BENCHMARKS & EXPLAINABILITY
# =============================================================
with tab_ml:
    st.subheader("Machine Learning Model Comparison & Diagnostic Verification")

    comp_csv_path = RESULTS_DIR / "model_comparison.csv"
    if comp_csv_path.exists():
        comp_df = pd.read_csv(comp_csv_path)
        st.dataframe(
            comp_df.rename(columns={
                "model": "Model Candidate",
                "cv_mae_mean": "5-Fold CV MAE (Lower is better)",
                "cv_mae_std": "CV MAE Std",
                "cv_rmse_mean": "CV RMSE",
                "cv_r2_mean": "CV R² Score (Higher is better)",
                "test_mae": "Held-Out Test MAE",
                "test_rmse": "Held-Out Test RMSE",
                "test_r2": "Held-Out Test R²",
            }),
            use_container_width=True,
            hide_index=True,
        )

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        comp_img = RESULTS_DIR / "figures" / "model_comparison.png"
        if comp_img.exists():
            st.image(str(comp_img), caption="Cross-Validated MAE Comparison (with Standard Error)", use_container_width=True)
    with b_col2:
        val_img = RESULTS_DIR / "figures" / "model_validation.png"
        if val_img.exists():
            st.image(str(val_img), caption="Observed vs Predicted AQI & Residual Distribution", use_container_width=True)

    st.markdown("---")
    st.markdown("#### Feature Importance: What Factors Govern Delhi's AQI?")
    feat_img = RESULTS_DIR / "figures" / "feature_importance.png"
    feat_csv = RESULTS_DIR / "feature_importance.csv"

    if feat_img.exists():
        st.image(str(feat_img), caption="Top 15 Predictive Features from Production Pipeline", use_container_width=True)
    elif feat_csv.exists():
        fdf = pd.read_csv(feat_csv).head(15)
        st.bar_chart(fdf.set_index("feature")["importance"])

# =============================================================
# TAB 5: STANDARDS & METHODOLOGY
# =============================================================
with tab_about:
    st.subheader("CPCB National Air Quality Index (NAQI) Framework")
    st.markdown("""
    India's Central Pollution Control Board (CPCB) calculates the National Air Quality Index (NAQI) using 
    standardized sub-indices across major criteria pollutants. The overall AQI corresponds to the worst-case sub-index.

    | AQI Range | Category | Color Code | General Public Health Impact |
    |---|---|---|---|
    | **0 – 50** | **Good** | Green | Minimal health impact; clean and safe breathing conditions. |
    | **51 – 100** | **Satisfactory** | Light Green | Minor breathing discomfort to unusually sensitive individuals. |
    | **101 – 200** | **Moderate** | Yellow / Amber | Breathing discomfort to people with asthma, lung, and heart diseases. |
    | **201 – 300** | **Poor** | Red | Breathing discomfort to most people on prolonged outdoor exposure. |
    | **301 – 400** | **Very Poor** | Purple | Respiratory illness on prolonged exposure; severe asthma triggers. |
    | **401 – 500+** | **Severe** | Maroon | Affects healthy people and seriously impacts those with existing medical conditions. |
    """)

    st.markdown("---")
    st.subheader("Feature Engineering & Physical Principles")
    st.markdown("""
    - **Particulate Burden (`pm_total`):** $\\text{PM}_{2.5} + \\text{PM}_{10}$ captures total airborne particulate mass.
    - **Combustion Ratio (`pm_ratio`):** $\\text{PM}_{2.5} / \\text{PM}_{10}$ distinguishes fine combustion particles from crustal road dust.
    - **Atmospheric Stagnation Index:** $1.0 / (\\text{windspeed} + 0.5)$ models air stability and pollutant trapping.
    - **Dew Point Inversion:** $\\text{temp} - ((100 - \\text{humidity}) / 5)$ approximates condensation point and winter thermal inversion likelihood.
    - **Cyclical Trigonometric Transforms:** $\\sin(2\\pi t / T)$ and $\\cos(2\\pi t / T)$ preserve circular continuity for hours (24h) and months (12m).
    """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>VayuDrishti AQI Platform · Built for Delhi NCR Environmental Monitoring & Public Health Protection</p>", unsafe_allow_html=True)
