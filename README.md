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

The project uses historical air-quality and weather data containing variables such as:

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

The raw dataset is not included in the repository. It should be placed inside:

```text
data/raw/
```

## Machine Learning Pipeline

```text
Raw Data
   ↓
Data Loading
   ↓
Data Preprocessing
   ↓
Exploratory Data Analysis
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

### Train the model

```bash
python src/train.py
```

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

Model performance and visualizations will be stored in:

```text
results/
```

## Current Status

🚧 **Project in development**

- [x] Project structure
- [x] Dataset collection
- [x] Initial preprocessing
- [ ] Exploratory data analysis
- [ ] Feature engineering
- [ ] Model training
- [ ] Model evaluation
- [ ] Prediction pipeline
- [ ] Streamlit application
- [ ] Final documentation

## Future Improvements

- Compare multiple machine learning models
- Improve feature engineering
- Perform hyperparameter tuning
- Add time-series forecasting
- Improve the Streamlit dashboard
- Add model explainability
- Deploy the application

## License

This project is for educational and research purposes.
