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
import joblib
import json
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.features import build_features, FEATURE_COLS, TARGET_COL

def train_test_split_temporal(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-series split:
    - Train: 2019-01-01 to 2022-08-31
    - Val:   2022-09-01 to 2022-12-31 (Used for early stopping)
    - Test:  2023-01-01 to 2024-12-31 (True holdout)
    """
    train = df[df["date"] < "2022-09-01"].copy()
    val   = df[(df["date"] >= "2022-09-01") & (df["date"] < "2023-01-01")].copy()
    test  = df[df["date"] >= "2023-01-01"].copy()
    
    logger.info(f"Split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    return train, val, test

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, float]:
    """Calculate and return evaluation metrics for predictions."""
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
    logger.info(f"  [{model_name}] MAE: {mae:,.0f} | RMSE: {rmse:,.0f} | SMAPE: {smape:.2f}%")
    return {"mae": round(float(mae), 2), "rmse": round(float(rmse), 2), "smape": round(float(smape), 4)}

def train_all(data_path: str = "data/uzbekistan_payments_enriched.csv") -> Dict[str, Any]:
    """Executes the full training pipeline with proper validation set."""
    os.makedirs("models", exist_ok=True)

    logger.info("Loading and building features...")
    df_raw = pd.read_csv(data_path)
    df = build_features(df_raw)

    logger.info("Splitting data into Train, Validation, and Test sets...")
    train, val, test = train_test_split_temporal(df)

    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_val, y_val     = val[FEATURE_COLS], val[TARGET_COL]
    X_test, y_test   = test[FEATURE_COLS], test[TARGET_COL]

    # Normalization baselines
    train_baseline = np.log1p(train["volume_rolling_mean_30"].values)
    val_baseline   = np.log1p(val["volume_rolling_mean_30"].values)
    test_baseline  = np.log1p(test["volume_rolling_mean_30"].values)
    
    y_train_norm   = np.log1p(y_train.values) - train_baseline
    y_val_norm     = np.log1p(y_val.values)   - val_baseline
    y_test_norm    = np.log1p(y_test.values)  - test_baseline

    all_metrics: Dict[str, Any] = {}

    # 1. Linear Regression (baseline - trained on Train+Val)
    logger.info("Training Linear Regression baseline...")
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])
    lr_pipeline.fit(X_train_val, y_train_val)
    lr_preds = lr_pipeline.predict(X_test)
    all_metrics["linear"] = compute_metrics(y_test.values, lr_preds, "Linear")
    joblib.dump(lr_pipeline, "models/linear_regression.joblib")

    # 2. XGBoost (Early stopping on Validation set)
    logger.info("Training XGBoost with validation-based early stopping...")
    xgb_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=50,
        eval_metric="rmse",
    )
    xgb_model.fit(
        X_train, y_train_norm,
        eval_set=[(X_val, y_val_norm)],
        verbose=False
    )
    xgb_preds = np.expm1(xgb_model.predict(X_test) + test_baseline)
    all_metrics["xgboost"] = compute_metrics(y_test.values, xgb_preds, "XGBoost")
    joblib.dump(xgb_model, "models/xgboost.joblib")

    # 3. CatBoost (Early stopping on Validation set)
    logger.info("Training CatBoost with validation-based early stopping...")
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=7,
        loss_function="RMSE",
        random_seed=42,
        early_stopping_rounds=50,
        verbose=False,
    )
    cat_model.fit(
        X_train, y_train_norm,
        eval_set=(X_val, y_val_norm),
        verbose=False
    )
    cat_preds = np.expm1(cat_model.predict(X_test) + test_baseline)
    all_metrics["catboost"] = compute_metrics(y_test.values, cat_preds, "CatBoost")
    cat_model.save_model("models/catboost.cbm")

    # Save true test predictions for evaluation
    test_results = test.copy()
    test_results["pred_linear"]  = lr_preds
    test_results["pred_xgboost"] = xgb_preds
    test_results["pred_catboost"] = cat_preds
    test_results.to_csv("models/test_predictions.csv", index=False)

    # Save metrics JSON
    all_metrics["xgboost"]["target"]  = "log_ratio_to_rolling_mean_30"
    all_metrics["catboost"]["target"] = "log_ratio_to_rolling_mean_30"
    all_metrics["linear"]["target"]   = "raw"
    with open("models/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info("Training pipeline complete. Models saved to models/")
    
    # Summary Table
    logger.info("\n" + "="*55)
    logger.info(f"{'Model':<20} {'MAE':>12} {'RMSE':>12} {'SMAPE':>10}")
    logger.info("-"*55)
    for model, m in all_metrics.items():
        logger.info(f"{model:<20} {m['mae']:>12,.0f} {m['rmse']:>12,.0f} {m['smape']:>9.2f}%")
    logger.info("="*55)
    
    return all_metrics

if __name__ == "__main__":
    train_all()
