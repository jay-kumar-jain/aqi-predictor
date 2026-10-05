import joblib
import pandas as pd


def load_artifacts():
    model = joblib.load("models/linear_regression.joblib")
    scaler = joblib.load("models/scaler.joblib")
    return model, scaler


def predict_sample(sample_df):
    model, scaler = load_artifacts()

    # Scale the input features using saved scaler
    sample_scaled = scaler.transform(sample_df)
    sample_scaled_df = pd.DataFrame(sample_scaled, columns=scaler.feature_names_in_)
    predictions = model.predict(sample_scaled_df)
    return predictions


if __name__ == "__main__":
    # Test prediction on a sample row
    df = pd.read_csv("data/processed/aqi_cleaned.csv")
    sample = df.drop(columns=["aqi_index"]).iloc[:3]
    sample = pd.get_dummies(sample, columns=["location"], drop_first=True)

    # Ensure all feature columns from training are present
    model, scaler = load_artifacts()
    expected_cols = scaler.feature_names_in_
    for col in expected_cols:
        if col not in sample.columns:
            sample[col] = 0
    sample = sample[expected_cols]

    preds = predict_sample(sample)
    actuals = df["aqi_index"].iloc[:3].values

    print("\nSample Predictions:")
    for i in range(len(preds)):
        print(f"Predicted AQI: {preds[i]:.1f} | Actual AQI: {actuals[i]}")
