# AQI Predictor

A machine learning project for predicting Air Quality Index (AQI) using historical air-quality and weather data.

## Overview

Air pollution is influenced by multiple pollutants and weather conditions. This project uses historical environmental data to build a machine learning model that predicts AQI based on factors such as PM2.5, PM10, CO, NO2, temperature, humidity, pressure, wind speed, location, and time-related features.

The project follows a modular machine learning workflow covering data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, and prediction.

## Features

- Data loading and preprocessing
- Missing-value handling
- Duplicate removal
- Date and time feature extraction
- Exploratory data analysis
- Feature engineering
- AQI prediction using machine learning
- Model evaluation
- Prediction script
- Streamlit web application

## Project Structure

```text
aqi-predictor/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data.py
│   ├── preprocess.py
│   ├── features.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
│
├── models/
│
├── results/
│   └── figures/
│
├── notebooks/
│   └── analysis.ipynb
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

The included `data/raw/aqi.csv` contains 52,560 hourly records with air-quality and weather fields such as:

- PM2.5
- PM10
- CO
- NO2
- Temperature
- Humidity
- Atmospheric pressure
- Wind speed
- Location
- Date and time
- AQI

The target is `aqi_index`. The preprocessing step removes duplicate rows and rows without a usable target/time, derives hour/day/month/weekday, and keeps feature missingness for fold-local imputation.

## Machine Learning Pipeline

```text
Raw Data
   ↓
Data Loading
   ↓
Data Preprocessing and Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Train / Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
AQI Prediction
   ↓
Streamlit Application
```

## Target Variable

The target variable is:

```text
aqi_index
```

The model will learn the relationship between environmental conditions, pollutant concentrations, time-related information, and AQI.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Jupyter Notebook
- Streamlit
- Joblib

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd aqi-predictor
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Preprocess the data

```bash
python src/preprocess.py
```

### Explore the data

```bash
python src/analyze.py
```

This writes `results/eda_report.md`, numeric and location summaries, missingness and correlation tables, plus `results/figures/eda_overview.png`.

The complete interactive analysis is in `notebooks/analysis.ipynb` and includes the exploration, split-safe preprocessing, regression diagnostics, cross-validation, and sample predictions.

### Train the model

```bash
python src/train.py
```

This compares Linear Regression, Random Forest, and XGBoost using the same five-fold split and preprocessing. Selection uses mean CV MAE; the untouched test split is reported for the comparison. The winner is refit on all rows and saved to `models/best_model.joblib`. Per-model results are saved to `results/model_comparison.csv` and `results/model_metrics.json`. If XGBoost is unavailable, that candidate is skipped with a note.

### Make a prediction

```bash
python src/predict.py
```

### Run the web application

```bash
streamlit run app.py
```

## Model Evaluation

The trained model will be evaluated using regression metrics such as:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

Model metrics, validation plots, and EDA outputs are stored in:

```text
results/
```

The application takes AQI-related pollutant and weather measurements as inputs. It estimates AQI for those measurements; it does not forecast future AQI.

## Current Status

## Analysis and validation notes

- Numeric feature missing values are median-imputed, with missingness indicators; `location` is mode-imputed and one-hot encoded. Imputers, encoder, and standard scaler are part of the estimator pipeline, so they are fitted on training data only and refitted within each cross-validation fold.
- Candidate estimators are multiple linear regression, random forest, and XGBoost. Each uses the same fold-local imputation, categorical encoding, and scaling pipeline. The selected model is the lowest mean CV MAE on the training partition; the test partition is reserved for final comparison. Predictions are floored at zero because AQI cannot be negative.
- The web app reads `models/best_model.joblib`; run model training before launching the app. Training also saves the Linear Regression baseline separately for comparison.
- The dataset is hourly. Random splits can put nearby observations in both train and test, so reported metrics measure interpolation/generalization across sampled records. For forecasting future dates, use a chronological holdout and time-series cross-validation.
- Pollutant concentrations are also inputs to the target AQI. If the intended product must predict AQI before pollutant measurements are available, remove those fields and retrain; current scores do not establish that use case.

🚧 **Project in development**

- [x] Project structure
- [x] Dataset collection
- [x] Initial preprocessing
- [x] Exploratory data analysis
- [x] Feature engineering
- [x] Model training
- [x] Model evaluation
- [x] Prediction pipeline
- [x] Streamlit application
- [x] Model comparison and documentation

## Future Improvements

- Improve feature engineering
- Perform hyperparameter tuning
- Add time-series forecasting
- Improve the Streamlit dashboard
- Add model explainability
- Deploy the application

## License

This project is for educational and research purposes.
