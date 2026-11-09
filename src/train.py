9o"""Compare AQI regression models, validate them, and save the CV winner."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

try:  # Supports both `python -m src.train` and `python src/train.py`.
    from .features import build_features, build_preprocessor
    from .preprocess import OUTPUT_PATH, preprocess_data
except ImportError:
    from features import build_features, build_preprocessor
    from preprocess import OUTPUT_PATH, preprocess_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
LINEAR_MODEL_PATH = MODELS_DIR / "linear_regression.joblib"
RANDOM_STATE = 42


def regression_metrics(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0.0)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def diagnose(train, test, cv_mean):
    gap = train["r2"] - test["r2"]
    if train["r2"] < 0.5 and abs(gap) < 0.1:
        label = "Likely underfitting (low train and test R²); model/features may be insufficient."
    elif gap > 0.1:
        label = "Possible overfitting/high variance (train R² materially exceeds test R²)."
    elif abs(gap) <= 0.1 and abs(cv_mean - test["r2"]) <= 0.15:
        label = "Train/test and cross-validation are reasonably consistent."
    else:
        label = "Results are mixed; inspect fold spread and time-based generalization."
    return {"train_test_r2_gap": float(gap), "diagnosis": label}


def model_candidates():
    """Return models available in this environment and an availability note."""
    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=160, min_samples_leaf=2, random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    notes = {}
    try:
        from xgboost import XGBRegressor
        candidates["XGBoost"] = XGBRegressor(
            objective="reg:squarederror", n_estimators=250, max_depth=6,
            learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, random_state=RANDOM_STATE, n_jobs=-1,
            tree_method="hist", verbosity=0,
        )
    except ImportError:
        notes["XGBoost"] = "Not evaluated: install xgboost to include this candidate."
    return candidates, notes


def run_training(data_path=None, make_plots=True):
    # Always regenerate from raw input to avoid using legacy globally-imputed data.
    if data_path is None:
        preprocess_data()
        data_path = OUTPUT_PATH
    df = pd.read_csv(Path(data_path))
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scorers = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    candidates, notes = model_candidates()
    models = {}
    comparison = []

    print(f"Comparing {len(candidates)} models on {len(X_train):,} training rows (5-fold CV)...")
    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("preprocess", build_preprocessor()),
            ("regression", estimator),
        ])
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=scorers,
            return_train_score=True, n_jobs=1,
        )
        cv_summary = {
            "fold_mae": (-scores["test_mae"]).astype(float).tolist(),
            "fold_rmse": (-scores["test_rmse"]).astype(float).tolist(),
            "fold_r2": scores["test_r2"].astype(float).tolist(),
            "mean_mae": float(-scores["test_mae"].mean()),
            "std_mae": float(scores["test_mae"].std()),
            "mean_rmse": float(-scores["test_rmse"].mean()),
            "std_rmse": float(scores["test_rmse"].std()),
            "mean_r2": float(scores["test_r2"].mean()),
            "std_r2": float(scores["test_r2"].std()),
        }
        pipeline.fit(X_train, y_train)
        train_metrics = regression_metrics(y_train, pipeline.predict(X_train))
        test_metrics = regression_metrics(y_test, pipeline.predict(X_test))
        models[name] = pipeline
        comparison.append({
            "model": name, "cv": cv_summary,
            "train": train_metrics, "test": test_metrics,
        })
        print(f"{name}: CV MAE={cv_summary['mean_mae']:.2f} ± {cv_summary['std_mae']:.2f}; "
              f"CV R²={cv_summary['mean_r2']:.4f}; test MAE={test_metrics['mae']:.2f}; "
              f"test R²={test_metrics['r2']:.4f}")

    # Model selection uses training-only CV MAE. The held-out test split is not
    # used to pick the winner and remains an unbiased final comparison.
    comparison.sort(key=lambda row: row["cv"]["mean_mae"])
    winner_name = comparison[0]["model"]
    winner = models[winner_name]
    winner_result = next(row for row in comparison if row["model"] == winner_name)
    diagnosis = diagnose(
        winner_result["train"], winner_result["test"], winner_result["cv"]["mean_r2"]
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_frame = pd.DataFrame([{
        "model": row["model"],
        "cv_mae_mean": row["cv"]["mean_mae"], "cv_mae_std": row["cv"]["std_mae"],
        "cv_rmse_mean": row["cv"]["mean_rmse"], "cv_r2_mean": row["cv"]["mean_r2"],
        "test_mae": row["test"]["mae"], "test_rmse": row["test"]["rmse"], "test_r2": row["test"]["r2"],
    } for row in comparison])
    comparison_frame.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    report = {
        "target": "aqi_index", "rows": int(len(df)), "features": list(X.columns),
        "split": {"train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
                  "test_fraction": 0.2, "random_state": RANDOM_STATE},
        "selection_metric": "lowest mean 5-fold CV MAE on training partition",
        "best_model": winner_name, "model_comparison": comparison,
        "availability_notes": notes, "bias_variance": diagnosis,
    }
    (RESULTS_DIR / "model_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Keep holdout evaluation intact above, then refit selected model on all
    # available rows for deployment. Also preserve a dedicated LR comparison model.
    winner.fit(X, y)
    joblib.dump(winner, BEST_MODEL_PATH)
    linear = Pipeline([
        ("preprocess", build_preprocessor()),
        ("regression", LinearRegression()),
    ]).fit(X, y)
    joblib.dump(linear, LINEAR_MODEL_PATH)

    print(f"\nSelected by CV MAE: {winner_name}")
    print(f"Held-out test: MAE={winner_result['test']['mae']:.2f}, "
          f"RMSE={winner_result['test']['rmse']:.2f}, R²={winner_result['test']['r2']:.4f}")
    print(f"Saved best full-data pipeline: {BEST_MODEL_PATH}")
    print(f"Saved model comparison: {RESULTS_DIR / 'model_comparison.csv'}")

    if make_plots:
        try:
            import matplotlib.pyplot as plt
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            test_pred = np.maximum(models[winner_name].predict(X_test), 0.0)
            residuals = y_test.to_numpy() - test_pred
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].scatter(y_test, test_pred, alpha=0.35, s=12)
            low = float(min(y_test.min(), test_pred.min()))
            high = float(max(y_test.max(), test_pred.max()))
            axes[0].plot([low, high], [low, high], "r--")
            axes[0].set(xlabel="Observed AQI", ylabel="Predicted AQI", title=f"{winner_name}: observed vs predicted")
            axes[1].scatter(test_pred, residuals, alpha=0.35, s=12)
            axes[1].axhline(0, color="r", linestyle="--")
            axes[1].set(xlabel="Predicted AQI", ylabel="Residual", title="Held-out test residuals")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "model_validation.png", dpi=160)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 4.5))
            ordered = comparison_frame.sort_values("cv_mae_mean", ascending=True)
            ax.bar(ordered["model"], ordered["cv_mae_mean"], yerr=ordered["cv_mae_std"], capsize=4)
            ax.set(ylabel="5-fold CV MAE (AQI points; lower is better)", title="Regression model comparison")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=160)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.boxplot([row["cv"]["fold_r2"] for row in comparison],
                       tick_labels=[row["model"] for row in comparison])
            ax.set(ylabel="R²", title="Cross-validation R² by model")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "model_cv_r2.png", dpi=160)
            plt.close(fig)
        except ImportError:
            print("Matplotlib unavailable; skipped comparison plots.")

    return winner, report


if __name__ == "__main__":
    run_training()
