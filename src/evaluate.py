import sys
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))  # makes 'src' importable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import shap
import joblib
import json
import os
from catboost import CatBoostRegressor
from src.features import FEATURE_COLS

os.makedirs("plots", exist_ok=True)


def load_artifacts() -> Tuple[pd.DataFrame, Dict[str, Any], Any, CatBoostRegressor]:
    """Load predictions, metrics, and models for evaluation."""
    preds = pd.read_csv("models/test_predictions.csv")
    preds["date"] = pd.to_datetime(preds["date"])
    with open("models/metrics.json") as f:
        metrics = json.load(f)
    xgb = joblib.load("models/xgboost.joblib")
    cat = CatBoostRegressor()
    cat.load_model("models/catboost.cbm")
    return preds, metrics, xgb, cat


def plot_forecast_vs_actual(preds: pd.DataFrame, metrics: Dict[str, Any]) -> None:
    """Generate and save a multi-panel plot comparing forecasts to actuals."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        "Uzbekistan Payment Volume Forecasting — Model Comparison\n2023–2024 Test Period",
        fontsize=14, fontweight="bold", y=0.98
    )

    models = [
        ("pred_linear",   "Linear Regression", "#e74c3c", metrics["linear"]["smape"]),
        ("pred_xgboost",  "XGBoost",           "#f39c12", metrics["xgboost"]["smape"]),
        ("pred_catboost", "CatBoost",           "#8e44ad", metrics["catboost"]["smape"]),
    ]

    for ax, (col, label, color, smape) in zip(axes, models):
        ax.plot(preds["date"], preds["transaction_volume"],
                color="#2c3e50", linewidth=1.2, label="Actual", alpha=0.9)
        ax.plot(preds["date"], preds[col],
                color=color, linewidth=1.2, linestyle="--",
                label=f"{label} (SMAPE {smape:.2f}%)", alpha=0.85)
        ax.fill_between(preds["date"],
                        preds["transaction_volume"], preds[col],
                        alpha=0.12, color=color)
        ax.set_ylabel("Transaction Volume", fontsize=10)
        ax.legend(loc="upper left", fontsize=9)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")
        )
        ax.grid(True, alpha=0.3)

        # Mark Uzbekistan-specific events
        for event_date, event_label in [
            ("2023-03-21", "Navruz"), ("2024-03-21", "Navruz"),
            ("2023-03-22", "Ramadan Start"), ("2024-03-10", "Ramadan Start"),
        ]:
            ed = pd.Timestamp(event_date)
            if preds["date"].min() <= ed <= preds["date"].max():
                ax.axvline(ed, color="gray", linestyle=":", alpha=0.5, linewidth=1)
                ax.text(ed, ax.get_ylim()[1] * 0.92, event_label,
                        fontsize=7, color="gray", rotation=45)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/forecast_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("✅ Saved → plots/forecast_vs_actual.png")


def plot_metrics_comparison(metrics: Dict[str, Any]) -> None:
    """Generate and save a bar chart comparing performance metrics across models."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")

    models = list(metrics.keys())
    colors = ["#e74c3c", "#f39c12", "#8e44ad"]
    metric_names = ["mae", "rmse", "smape"]
    titles = ["MAE (lower = better)", "RMSE (lower = better)", "SMAPE % (lower = better)"]

    for ax, metric, title in zip(axes, metric_names, titles):
        values = [metrics[m][metric] for m in models]
        bars = ax.bar(models, values, color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(metric.upper())
        for bar, val in zip(bars, values):
            label = f"{val:.2f}%" if metric == "smape" else f"{val/1000:.0f}K"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    label, ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, max(values) * 1.2)

    plt.tight_layout()
    plt.savefig("plots/metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("✅ Saved → plots/metrics_comparison.png")


def plot_shap(xgb_model: Any, preds: pd.DataFrame) -> None:
    """Generate and save a SHAP summary plot for feature importance."""
    X_test = preds[FEATURE_COLS]

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar",
                      max_display=15, show=False,
                      color="#f39c12")
    plt.title("XGBoost — Top 15 Most Important Features (SHAP)\nUzbekistan Payment Volume Forecasting",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("✅ Saved → plots/shap_importance.png")


def plot_residuals(preds: pd.DataFrame) -> None:
    """Generate and save a residual plot to analyze prediction errors over time."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Residual Analysis — Prediction Errors by Month",
                 fontsize=13, fontweight="bold")

    models = [
        ("pred_linear",   "Linear Regression", "#e74c3c"),
        ("pred_xgboost",  "XGBoost",           "#f39c12"),
        ("pred_catboost", "CatBoost",           "#8e44ad"),
    ]

    preds = preds.copy()
    preds["month_label"] = preds["date"].dt.strftime("%b %Y")

    for ax, (col, label, color) in zip(axes, models):
        residuals = preds["transaction_volume"] - preds[col]
        ax.scatter(preds["date"], residuals, alpha=0.3, s=8, color=color)
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel("Residual (Actual − Predicted)")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K")
        )
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig("plots/residuals.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("✅ Saved → plots/residuals.png")


if __name__ == "__main__":
    logger.info("📊 Generating evaluation plots...")
    preds, metrics, xgb, cat = load_artifacts()

    plot_forecast_vs_actual(preds, metrics)
    plot_metrics_comparison(metrics)
    plot_shap(xgb, preds)
    plot_residuals(preds)

    logger.info("\n✅ All plots saved to /plots — ready for LinkedIn & README!")
    logger.info("\nPlots generated:")
    for f in sorted(os.listdir("plots")):
        if f != ".gitkeep":
            logger.info(f"  📈 plots/{f}")

