import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ── Temporal features ──────────────────────────────────────────
    df["day_of_week"]   = df["date"].dt.dayofweek
    df["month"]         = df["date"].dt.month
    df["year"]          = df["date"].dt.year
    df["day_of_month"]  = df["date"].dt.day
    df["quarter"]       = df["date"].dt.quarter
    df["week_of_year"]  = df["date"].dt.isocalendar().week.astype(int)
    df["days_elapsed"]  = (df["date"] - df["date"].min()).dt.days

    # ── Sine/cosine cyclical encoding ──────────────────────────────
    df["month_sin"]        = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]        = np.cos(2 * np.pi * df["month"] / 12)
    df["day_of_week_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_month_sin"] = np.sin(2 * np.pi * df["day_of_month"] / 31)
    df["day_of_month_cos"] = np.cos(2 * np.pi * df["day_of_month"] / 31)

    # ── Uzbekistan-specific calendar flags ─────────────────────────
    ramadan_dates = [
        ("2019-05-05", "2019-06-03"),
        ("2020-04-23", "2020-05-23"),
        ("2021-04-12", "2021-05-12"),
        ("2022-04-02", "2022-05-02"),
        ("2023-03-22", "2023-04-21"),
        ("2024-03-10", "2024-04-09"),
    ]
    df["is_ramadan"] = 0
    for start, end in ramadan_dates:
        df.loc[(df["date"] >= start) & (df["date"] <= end), "is_ramadan"] = 1

    df["is_navruz"]    = ((df["month"] == 3) & (df["day_of_month"].isin([20,21,22,23]))).astype(int)
    df["is_payday"]    = df["day_of_month"].isin([10, 25]).astype(int)
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_covid_dip"] = ((df["year"] == 2020) & (df["quarter"] == 2)).astype(int)

    # ── Lag features ───────────────────────────────────────────────
    for lag in [1, 7, 14, 30]:
        df[f"volume_lag_{lag}"] = df["transaction_volume"].shift(lag)

    # ── Rolling window features ────────────────────────────────────
    for window in [7, 14, 30]:
        df[f"volume_rolling_mean_{window}"] = (
            df["transaction_volume"].shift(1).rolling(window).mean()
        )
        df[f"volume_rolling_std_{window}"] = (
            df["transaction_volume"].shift(1).rolling(window).std()
        )

    df["avg_value_rolling_7"] = (
        df["avg_transaction_value"].shift(1).rolling(7).mean()
    )

    # ── Log-transformed features — help tree models learn the trend ─
    # Trees can't extrapolate raw volumes beyond training range;
    # log-scale makes the growth curve gentle and learnable.
    df["log_volume_lag_1"]    = np.log1p(df["volume_lag_1"])
    df["log_volume_lag_7"]    = np.log1p(df["volume_lag_7"])
    df["log_volume_lag_30"]   = np.log1p(df["volume_lag_30"])
    df["log_rolling_mean_7"]  = np.log1p(df["volume_rolling_mean_7"])
    df["log_rolling_mean_30"] = np.log1p(df["volume_rolling_mean_30"])

    # ── Macro: CBU exchange rate features ─────────────────────────
    # Recomputed here so build_features is self-contained even if the
    # caller passes a DataFrame that only has usd_uzs_rate (not the lags).
    if "usd_uzs_rate" in df.columns:
        df["usd_uzs_rate_lag7"]   = df["usd_uzs_rate"].shift(7)
        df["usd_uzs_rate_change"] = df["usd_uzs_rate"].pct_change(7).round(6)

    # Drop NaN rows produced by lags/rolling (first ~30 rows)
    df = df.dropna().reset_index(drop=True)

    return df


FEATURE_COLS = [
    "days_elapsed",
    "day_of_week", "month", "quarter", "week_of_year", "day_of_month",
    "month_sin", "month_cos",
    "day_of_week_sin", "day_of_week_cos",
    "day_of_month_sin", "day_of_month_cos",
    "is_ramadan", "is_navruz", "is_payday", "is_weekend", "is_covid_dip",
    "volume_lag_1", "volume_lag_7", "volume_lag_14", "volume_lag_30",
    "volume_rolling_mean_7", "volume_rolling_mean_14", "volume_rolling_mean_30",
    "volume_rolling_std_7", "volume_rolling_std_14", "volume_rolling_std_30",
    "avg_value_rolling_7",
    # Log-transformed trend features — critical for tree models
    "log_volume_lag_1", "log_volume_lag_7", "log_volume_lag_30",
    "log_rolling_mean_7", "log_rolling_mean_30",
    # Macro: real CBU USD/UZS data
    "usd_uzs_rate", "usd_uzs_rate_lag7", "usd_uzs_rate_change",
]

TARGET_COL = "transaction_volume"
