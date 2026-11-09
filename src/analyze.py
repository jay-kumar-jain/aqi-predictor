"""Exploratory statistics and plots for the AQI dataset."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:  # Supports both `python -m src.analyze` and `python src/analyze.py`.
    from .preprocess import FEATURE_COLUMNS, TARGET_COLUMN, OUTPUT_PATH, preprocess_data
except ImportError:  # Direct script execution puts src/, rather than project root, on sys.path.
    from preprocess import FEATURE_COLUMNS, TARGET_COLUMN, OUTPUT_PATH, preprocess_data


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def run_analysis(data_path=None):
    if data_path is None:
        df = preprocess_data()
    else:
        df = pd.read_csv(Path(data_path))
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    numeric = df.select_dtypes(include="number")
    describe = numeric.describe().T
    describe.to_csv(RESULTS / "numeric_summary.csv")
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": df.isna().mean().mul(100).round(3),
        "dtype": df.dtypes.astype(str),
    }).sort_values("missing_count", ascending=False)
    missing.to_csv(RESULTS / "missing_values.csv")
    categorical_summary = df["location"].value_counts(dropna=False).rename_axis("location").reset_index(name="rows")
    categorical_summary.to_csv(RESULTS / "location_counts.csv", index=False)
    correlations = numeric.corr(method="pearson")
    correlations.to_csv(RESULTS / "correlations.csv")

    from scipy.stats import skew, spearmanr
    target = df[TARGET_COLUMN]
    target_stats = {
        "count": int(target.count()), "mean": float(target.mean()), "median": float(target.median()),
        "std": float(target.std()), "min": float(target.min()), "q1": float(target.quantile(.25)),
        "q3": float(target.quantile(.75)), "max": float(target.max()), "skewness": float(skew(target, nan_policy="omit")),
        "iqr_outlier_count": int(((target < target.quantile(.25) - 1.5 * (target.quantile(.75) - target.quantile(.25))) |
                                   (target > target.quantile(.75) + 1.5 * (target.quantile(.75) - target.quantile(.25)))).sum()),
    }
    # Exploratory associations only; high correlation is not causal evidence.
    association = {}
    for col in [c for c in FEATURE_COLUMNS if c != "location"]:
        valid = df[[col, TARGET_COLUMN]].dropna()
        if len(valid) > 1:
            rho, pval = spearmanr(valid[col], valid[TARGET_COLUMN])
            association[col] = {"spearman_rho": float(rho), "p_value": float(pval), "n": int(len(valid))}
    report_data = {
        "rows": int(len(df)), "columns": list(df.columns), "duplicate_rows": int(df.duplicated().sum()),
        "locations": int(df["location"].nunique()), "location_rows": {str(k): int(v) for k, v in df["location"].value_counts(dropna=False).items()},
        "target": target_stats, "target_spearman_associations": association,
    }
    (RESULTS / "eda_summary.json").write_text(json.dumps(report_data, indent=2), encoding="utf-8")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].hist(target, bins=40, edgecolor="white")
    axes[0, 0].set_title("AQI distribution")
    places = list(df["location"].dropna().unique())
    # `tick_labels` is the supported Matplotlib argument; `labels` was removed in 3.11.
    axes[0, 1].boxplot(
        [df.loc[df["location"] == place, TARGET_COLUMN].dropna() for place in places],
        tick_labels=places,
    )
    axes[0, 1].tick_params(axis="x", rotation=25)
    axes[0, 1].set_title("AQI by location")
    hourly = df.groupby("hour", observed=False)[TARGET_COLUMN].agg(["mean", "median", "std"])
    axes[1, 0].plot(hourly.index, hourly["mean"], marker="o", label="Mean")
    axes[1, 0].plot(hourly.index, hourly["median"], marker=".", label="Median")
    axes[1, 0].fill_between(hourly.index, hourly["mean"] - hourly["std"], hourly["mean"] + hourly["std"], alpha=.15)
    axes[1, 0].set(xlabel="Hour of day", ylabel="AQI", title="AQI by hour (mean ± 1 SD)")
    axes[1, 0].legend()
    image = axes[1, 1].imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axes[1, 1].set_xticks(range(len(correlations.columns)), correlations.columns, rotation=90, fontsize=7)
    axes[1, 1].set_yticks(range(len(correlations.index)), correlations.index, fontsize=7)
    fig.colorbar(image, ax=axes[1, 1], fraction=.046, pad=.04)
    axes[1, 1].set_title("Pearson correlations")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_overview.png", dpi=160)
    plt.close(fig)

    lines = ["# AQI exploratory data analysis", "", f"- Rows: {len(df):,}", f"- Locations: {df['location'].nunique()}",
             f"- AQI mean / median / SD: {target.mean():.2f} / {target.median():.2f} / {target.std():.2f}",
             f"- AQI range: {target.min():.2f}–{target.max():.2f}", f"- AQI skewness: {target_stats['skewness']:.3f}",
             f"- IQR rule flags: {target_stats['iqr_outlier_count']:,} target observations (flags are not automatic errors)",
             "", "## Missingness", "", "| Column | Missing count | Missing % | Type |", "|---|---:|---:|---|"]
    for col, row in missing.iterrows():
        lines.append(f"| {col} | {int(row['missing_count']):,} | {row['missing_percent']:.3f}% | {row['dtype']} |")
    lines.extend(["", "## AQI rank correlations", "",
             "Spearman associations below describe monotonic association, not causation.", "",
             "| Feature | Spearman ρ | p-value | n |", "|---|---:|---:|---:|"])
    for col, stats in sorted(association.items(), key=lambda item: abs(item[1]["spearman_rho"]), reverse=True):
        lines.append(f"| {col} | {stats['spearman_rho']:.3f} | {stats['p_value']:.3g} | {stats['n']:,} |")
    lines.extend(["", "## Files", "", "See `figures/eda_overview.png`, `numeric_summary.csv`, `location_counts.csv`, and `correlations.csv`.",
                  "", "Because the data are hourly, random split/CV scores may be optimistic for forecasting future dates. Use a chronological holdout and time-series CV when the intended use is future prediction."])
    (RESULTS / "eda_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"EDA complete: {RESULTS}")
    print(f"Rows={len(df):,}; locations={df['location'].nunique()}; AQI mean={target.mean():.2f}, median={target.median():.2f}, SD={target.std():.2f}")
    print(f"Missing cells in model features: {int(df[FEATURE_COLUMNS].isna().sum().sum()):,}")
    return report_data


if __name__ == "__main__":
    run_analysis()
