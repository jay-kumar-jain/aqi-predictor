import os
import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import prepare_data
from src.model import get_model


def evaluate(y_true, y_pred, split_name="Test"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n--- {split_name} Performance ---")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")

    return {"mae": mae, "rmse": rmse, "r2": r2}


def run_training():
    # 1. Load scaled features and train-test splits
    X_train, X_test, y_train, y_test, scaler = prepare_data()

    # 2. Train linear regression model
    model = get_model()
    model.fit(X_train, y_train)

    # 3. Model validation on train and test (detect overfitting / underfitting)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_metrics = evaluate(y_train, train_pred, split_name="Train")
    test_metrics = evaluate(y_test, test_pred, split_name="Test")

    # 4. Check Bias-Variance tradeoff
    r2_gap = train_metrics["r2"] - test_metrics["r2"]
    print(f"\nVariance check (Train R2 - Test R2): {r2_gap:.4f}")
    if abs(r2_gap) < 0.05:
        print("Diagnosis: Model generalizes well (low variance, no overfitting).")
    else:
        print("Diagnosis: Potential overfitting detected (high variance).")

    # 5. 5-Fold Cross-Validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    print("\n--- 5-Fold Cross Validation ---")
    print("Fold R2 Scores:", np.round(cv_scores, 4))
    print(f"Mean CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # 6. Save model and scaler artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/linear_regression.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    print("\nArtifacts saved in models/")

    return model, scaler


if __name__ == "__main__":
    run_training()
