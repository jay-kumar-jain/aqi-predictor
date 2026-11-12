"""Comprehensive Model Training, Cross-Validation & Diagnostics Pipeline."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

try:  # Supports both `python -m src.train` and `python src/train.py`.
    from .features import build_features, build_preprocessor
    from .preprocess import OUTPUT_PATH, preprocess_data, BASE_FEATURE_COLUMNS
except ImportError:
    from features import build_features, build_preprocessor
    from preprocess import OUTPUT_PATH, preprocess_data, BASE_FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
LINEAR_MODEL_PATH = MODELS_DIR / "linear_regression.joblib"
FEATURE_IMP_PATH = RESULTS_DIR / "feature_importance.csv"
RANDOM_STATE = 42


def regression_metrics(y_true, y_pred):
    """Calculates MAE, RMSE, and R2 with non-negative lower bound for AQI."""
    y_pred = np.maximum(y_pred, 0.0)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def diagnose_bias_variance(train, test, cv_mean_r2):
    """Analyzes generalization gap to diagnose overfitting vs underfitting."""
    gap = train["r2"] - test["r2"]
    if train["r2"] < 0.55 and abs(gap) < 0.1:
        label = "Likely underfitting: model architecture or linear assumptions lack expressive power."
    elif gap > 0.08:
        label = "Mild variance / possible overfitting: training performance exceeds test generalization."
    elif abs(gap) <= 0.08 and abs(cv_mean_r2 - test["r2"]) <= 0.1:
        label = "Strong generalization: cross-validation and held-out test metrics align harmoniously."
    else:
        label = "Moderate variation: performance is acceptable, review individual fold distribution."
    return {"train_test_r2_gap": float(gap), "diagnosis": label}


def model_candidates():
    """Initializes competitive candidate regressors."""
    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=180,
            max_depth=22,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    notes = {}
    try:
        from xgboost import XGBRegressor
        candidates["XGBoost"] = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=7,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )
    except ImportError:
        notes["XGBoost"] = "XGBoost not installed; candidate omitted."
    return candidates, notes


def extract_feature_importance(pipeline, feature_names):
    """Extracts normalized feature importance from tree-based estimators."""
    regressor = pipeline.named_steps.get("regression")
    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
        # Clean up feature names
        clean_names = [f.replace("numeric__", "").replace("categorical__", "") for f in feature_names]
        df_imp = pd.DataFrame({"feature": clean_names, "importance": importances})
        return df_imp.sort_values(by="importance", ascending=False).reset_index(drop=True)
    return None


def run_training(data_path=None, make_plots=True):
    """Executes the full machine learning training, cross-validation, and selection cycle."""
    if data_path is None or not Path(data_path).exists():
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

    print(f"\n=================================================================")
    print(f"🚀 Training & Benchmarking Models on {len(X_train):,} Delhi AQI records (5-fold CV)")
    print(f"=================================================================")

    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("preprocess", build_preprocessor()),
            ("regression", estimator),
        ])

        print(f"-> Cross-validating {name}...")
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

        # Fit on training split to measure held-out test metrics
        pipeline.fit(X_train, y_train)
        train_metrics = regression_metrics(y_train, pipeline.predict(X_train))
        test_metrics = regression_metrics(y_test, pipeline.predict(X_test))

        models[name] = pipeline
        comparison.append({
            "model": name,
            "cv": cv_summary,
            "train": train_metrics,
            "test": test_metrics,
        })

        print(f"   [CV 5-Fold]   MAE: {cv_summary['mean_mae']:.2f} ± {cv_summary['std_mae']:.2f} | R²: {cv_summary['mean_r2']:.4f}")
        print(f"   [Held-Out 20%] MAE: {test_metrics['mae']:.2f} | RMSE: {test_metrics['rmse']:.2f} | R²: {test_metrics['r2']:.4f}\n")

    # Winner selection strictly by lowest 5-fold cross-validated MAE
    comparison.sort(key=lambda row: row["cv"]["mean_mae"])
    winner_name = comparison[0]["model"]
    winner_pipeline = models[winner_name]
    winner_result = next(row for row in comparison if row["model"] == winner_name)
    diagnosis = diagnose_bias_variance(
        winner_result["train"], winner_result["test"], winner_result["cv"]["mean_r2"]
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Save tabular comparison
    comp_df = pd.DataFrame([{
        "model": row["model"],
        "cv_mae_mean": row["cv"]["mean_mae"],
        "cv_mae_std": row["cv"]["std_mae"],
        "cv_rmse_mean": row["cv"]["mean_rmse"],
        "cv_r2_mean": row["cv"]["mean_r2"],
        "test_mae": row["test"]["mae"],
        "test_rmse": row["test"]["rmse"],
        "test_r2": row["test"]["r2"],
    } for row in comparison])
    comp_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    # Extract feature names & importances from winning pipeline
    try:
        col_trans = winner_pipeline.named_steps["preprocess"].named_steps["col_transform"]
        feat_names = col_trans.get_feature_names_out()
        feat_imp = extract_feature_importance(winner_pipeline, feat_names)
        if feat_imp is not None:
            feat_imp.to_csv(FEATURE_IMP_PATH, index=False)
            top_feats = feat_imp.head(5).to_dict(orient="records")
        else:
            top_feats = []
    except Exception as e:
        print(f"Feature importance extraction notice: {e}")
        top_feats = []

    # Finalize model report
    report = {
        "target": "aqi_index",
        "rows": int(len(df)),
        "base_features": BASE_FEATURE_COLUMNS,
        "split": {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_fraction": 0.2,
            "random_state": RANDOM_STATE,
        },
        "selection_metric": "lowest mean 5-fold CV MAE on training partition",
        "best_model": winner_name,
        "model_comparison": comparison,
        "bias_variance": diagnosis,
        "top_features": top_feats,
        "availability_notes": notes,
    }
    (RESULTS_DIR / "model_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Fit best model on all available data for production inference, saving with compression
    print(f"Refitting top-performing model ({winner_name}) on full dataset ({len(df):,} rows)...")
    winner_pipeline.fit(X, y)
    joblib.dump(winner_pipeline, BEST_MODEL_PATH, compress=3)

    # Save linear regression baseline for comparison
    linear_pipe = Pipeline([
        ("preprocess", build_preprocessor()),
        ("regression", LinearRegression()),
    ]).fit(X, y)
    joblib.dump(linear_pipe, LINEAR_MODEL_PATH, compress=3)

    print(f"\n✅ SUCCESS!")
    print(f"Best Model Selected: {winner_name}")
    print(f"Saved Production Pipeline to: {BEST_MODEL_PATH} (compressed)")
    print(f"Saved Metrics Report to: {RESULTS_DIR / 'model_metrics.json'}")

    # Generate visual evaluation plots
    if make_plots:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_theme(style="whitegrid")

            # 1. Observed vs Predicted & Residuals
            test_preds = np.maximum(models[winner_name].predict(X_test), 0.0)
            residuals = y_test.to_numpy() - test_preds

            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            axes[0].scatter(y_test, test_preds, alpha=0.3, s=12, color="#1f77b4")
            min_val = min(y_test.min(), test_preds.min())
            max_val = max(y_test.max(), test_preds.max())
            axes[0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect Fit (y=x)")
            axes[0].set_xlabel("Observed AQI", fontsize=11)
            axes[0].set_ylabel("Predicted AQI", fontsize=11)
            axes[0].set_title(f"{winner_name}: Observed vs Predicted (Held-Out Test)", fontsize=12, fontweight="bold")
            axes[0].legend()

            axes[1].scatter(test_preds, residuals, alpha=0.3, s=12, color="#e6550d")
            axes[1].axhline(0, color="black", linestyle="--", linewidth=1.2)
            axes[1].set_xlabel("Predicted AQI", fontsize=11)
            axes[1].set_ylabel("Residual (Actual - Predicted)", fontsize=11)
            axes[1].set_title("Residual Distribution (Error Analysis)", fontsize=12, fontweight="bold")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "model_validation.png", dpi=200)
            plt.close(fig)

            # 2. Model Comparison Bar Chart
            fig, ax = plt.subplots(figsize=(8, 4.8))
            sorted_comp = comp_df.sort_values("cv_mae_mean", ascending=True)
            bars = ax.bar(
                sorted_comp["model"],
                sorted_comp["cv_mae_mean"],
                yerr=sorted_comp["cv_mae_std"],
                capsize=5,
                color=["#2ca02c", "#1f77b4", "#ff7f0e"][:len(sorted_comp)],
                alpha=0.85
            )
            ax.set_ylabel("5-Fold CV MAE (Lower is Better)", fontsize=11)
            ax.set_title("Model Comparison: 5-Fold Cross-Validation MAE", fontsize=13, fontweight="bold")
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:.2f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 6), textcoords="offset points",
                            ha="center", va="bottom", fontweight="bold")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=200)
            plt.close(fig)

            # 3. Boxplot of Fold R2
            fig, ax = plt.subplots(figsize=(8, 4.8))
            ax.boxplot([row["cv"]["fold_r2"] for row in comparison],
                       tick_labels=[row["model"] for row in comparison],
                       patch_artist=True,
                       boxprops=dict(facecolor="#c6dbef", color="#08519c"))
            ax.set_ylabel("Cross-Validation R² Score", fontsize=11)
            ax.set_title("5-Fold Cross-Validation R² Stability", fontsize=13, fontweight="bold")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "model_cv_r2.png", dpi=200)
            plt.close(fig)

            # 4. Feature Importance Horizontal Bar Chart
            if feat_imp is not None:
                top_15 = feat_imp.head(15).iloc[::-1]
                fig, ax = plt.subplots(figsize=(9, 6))
                ax.barh(top_15["feature"], top_15["importance"], color="#3182bd", alpha=0.9)
                ax.set_xlabel("Relative Importance Score", fontsize=11)
                ax.set_title(f"Top 15 Predictive Factors for Delhi AQI ({winner_name})", fontsize=13, fontweight="bold")
                fig.tight_layout()
                fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=200)
                plt.close(fig)

        except Exception as err:
            print(f"Plotting notice: {err}")

    return winner_pipeline, report


if __name__ == "__main__":
    run_training()
