import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # makes 'src' importable when run directly

import pandas as pd
import numpy as np
import joblib
import json
from datetime import timedelta
from catboost import CatBoostRegressor
from src.features import build_features, FEATURE_COLS


def load_models(models_dir: str = "models") -> dict:
    """Load all trained models and metadata from disk."""
    models = {}

    models["linear"] = joblib.load(f"{models_dir}/linear_regression.joblib")

    xgb = joblib.load(f"{models_dir}/xgboost.joblib")
    models["xgboost"] = xgb

    cat = CatBoostRegressor()
    cat.load_model(f"{models_dir}/catboost.cbm")
    models["catboost"] = cat

    with open(f"{models_dir}/metrics.json") as f:
        models["metrics"] = json.load(f)

    # Load enriched dataset to use as history for lag/rolling features
    models["history_df"] = pd.read_csv("data/uzbekistan_payments_enriched.csv")
    models["history_df"]["date"] = pd.to_datetime(models["history_df"]["date"])

    return models


def make_prediction(target_date: str, model_name: str, models: dict) -> dict:
    """
    Predict transaction volume for target_date.

    Strategy: build a 60-row history window ending at target_date - 1 day,
    run build_features(), take the last row, and run inference.
    For tree models trained on log(volume / rolling_mean_30), denormalize
    using the rolling_mean_30 value for that row.
    """
    target_dt = pd.Timestamp(target_date)
    history_df = models["history_df"]

    # Build a window of real historical data ending the day before target_date
    # We need at least 30 days of history for rolling features
    window_start = target_dt - timedelta(days=60)
    window = history_df[
        (history_df["date"] >= window_start) & (history_df["date"] < target_dt)
    ].copy()

    # Append the target date row (volume unknown — will be predicted)
    # Fill transaction_volume with the last known value as a placeholder
    # (the lag/rolling features will be built from real history; the target row's
    # own volume is what we're predicting and won't be used as a feature)
    last_known_volume = window["transaction_volume"].iloc[-1]
    last_known_avg_value = window["avg_transaction_value"].iloc[-1]
    last_known_rate = window["usd_uzs_rate"].iloc[-1] if "usd_uzs_rate" in window.columns else None

    target_row = {
        "date": target_dt,
        "transaction_volume": last_known_volume,     # placeholder — not used as feature
        "avg_transaction_value": last_known_avg_value,
        "region": "Tashkent",   # not in FEATURE_COLS, has no effect on predictions
    }
    if last_known_rate is not None:
        target_row["usd_uzs_rate"] = last_known_rate

    window = pd.concat([window, pd.DataFrame([target_row])], ignore_index=True)

    # Build features — build_features drops NaN rows (first ~30), so target row
    # will be the LAST row after feature engineering
    featured = build_features(window)

    # Verify the last row corresponds to target_date
    if featured["date"].iloc[-1] != target_dt:
        raise ValueError(
            f"Feature engineering produced unexpected last date: "
            f"{featured['date'].iloc[-1]} (expected {target_dt})"
        )

    # Subset to FEATURE_COLS and take the last row
    X_pred = featured[FEATURE_COLS].iloc[[-1]]
    rolling_mean_30 = featured["volume_rolling_mean_30"].iloc[-1]

    model = models[model_name]
    metrics = models["metrics"][model_name]
    target_type = metrics.get("target", "raw")

    if target_type == "log_ratio_to_rolling_mean_30":
        # Tree model: predict log-deviation, denormalize using rolling_mean_30
        raw_pred = model.predict(X_pred)
        if hasattr(raw_pred, "__len__"):
            raw_pred = raw_pred[0]
        baseline = np.log1p(rolling_mean_30)
        predicted_volume = int(np.expm1(raw_pred + baseline))
    else:
        # Linear Regression: predicts raw volume directly
        raw_pred = model.predict(X_pred)
        if hasattr(raw_pred, "__len__"):
            raw_pred = raw_pred[0]
        predicted_volume = int(max(0, raw_pred))

    # Detect Uzbekistan-specific events for the confidence note
    is_navruz = (target_dt.month == 3) and (target_dt.day in [20, 21, 22, 23])
    is_payday = target_dt.day in [10, 25]
    is_weekend = target_dt.dayofweek >= 5

    ramadan_dates = [
        ("2019-05-05", "2019-06-03"),
        ("2020-04-23", "2020-05-23"),
        ("2021-04-12", "2021-05-12"),
        ("2022-04-02", "2022-05-02"),
        ("2023-03-22", "2023-04-21"),
        ("2024-03-10", "2024-04-09"),
        ("2025-03-01", "2025-03-30"),
        ("2026-02-18", "2026-03-19"),
    ]
    is_ramadan = any(
        pd.Timestamp(s) <= target_dt <= pd.Timestamp(e)
        for s, e in ramadan_dates
    )

    notes = []
    if is_navruz:
        notes.append("Navruz holiday detected — expect elevated volumes")
    if is_ramadan:
        notes.append("Ramadan period — expect reduced daytime volumes")
    if is_payday:
        notes.append("Payday spike expected (10th or 25th of month)")
    if is_weekend:
        notes.append("Weekend — expect ~22% lower volumes")
    confidence_note = " | ".join(notes) if notes else "Normal trading day"

    return {
        "date": target_date,
        "predicted_volume": predicted_volume,
        "model_used": model_name,
        "smape_on_test": metrics["smape"],
        "is_navruz": is_navruz,
        "is_ramadan": is_ramadan,
        "is_payday": is_payday,
        "is_weekend": is_weekend,
        "confidence_note": confidence_note,
    }


if __name__ == "__main__":
    print("🔮 Loading models...")
    models = load_models()

    # Quick smoke test
    for model_name in ["linear", "xgboost", "catboost"]:
        result = make_prediction("2024-06-15", model_name, models)
        print(
            f"  [{model_name.upper():>8}] "
            f"Predicted: {result['predicted_volume']:>10,}  |  "
            f"SMAPE (test): {result['smape_on_test']:.2f}%"
        )
    print("\n✅ predict.py smoke test passed")
