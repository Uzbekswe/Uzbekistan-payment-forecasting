import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
from src.features import build_features, FEATURE_COLS, _TRAINING_START


def _make_sample_df(n_days: int = 60, start: str = "2023-01-01") -> pd.DataFrame:
    """Create a minimal DataFrame that build_features() can process."""
    dates = pd.date_range(start=start, periods=n_days, freq="D")
    np.random.seed(42)
    return pd.DataFrame({
        "date": dates,
        "transaction_volume": np.random.randint(1_000_000, 3_000_000, n_days),
        "avg_transaction_value": np.random.randint(10_000, 15_000, n_days),
        "usd_uzs_rate": np.linspace(11_000, 11_500, n_days),
    })


def test_build_features_contains_all_feature_cols():
    """build_features() output must contain every column in FEATURE_COLS."""
    df = _make_sample_df()
    result = build_features(df)
    missing = [col for col in FEATURE_COLS if col not in result.columns]
    assert missing == [], f"Missing columns after build_features: {missing}"


def test_build_features_no_nans_in_feature_cols():
    """No NaN values in FEATURE_COLS after build_features (NaN rows are dropped)."""
    df = _make_sample_df()
    result = build_features(df)
    nan_cols = [col for col in FEATURE_COLS if result[col].isna().any()]
    assert nan_cols == [], f"NaN values found in columns: {nan_cols}"


def test_days_elapsed_uses_fixed_training_start():
    """days_elapsed must be computed relative to _TRAINING_START, not df.min().
    This ensures inference on a short window produces the same scale as training.
    """
    df = _make_sample_df(n_days=60, start="2024-01-01")
    result = build_features(df)
    # The first date in the window is 2024-01-01 (+30 rows dropped by rolling)
    # days_elapsed should reflect distance from 2019-01-01, not from 2024-01-01
    first_date = result["date"].iloc[0]
    expected_days = (first_date - _TRAINING_START).days
    actual_days = result["days_elapsed"].iloc[0]
    assert actual_days == expected_days, (
        f"days_elapsed={actual_days} but expected {expected_days} "
        f"(relative to _TRAINING_START={_TRAINING_START.date()}). "
        f"If this fails, build_features is using df['date'].min() instead of _TRAINING_START."
    )


def test_ramadan_flag_fires_correctly():
    """is_ramadan must be 1 during Ramadan 2024 and 0 outside it."""
    # 150 days from 2024-02-01 covers through late June — includes both
    # Ramadan (Mar 10–Apr 9) and a clearly post-Ramadan date (May 15).
    df = _make_sample_df(n_days=150, start="2024-02-01")
    result = build_features(df)

    ramadan_row = result[result["date"] == pd.Timestamp("2024-03-20")]
    non_ramadan_row = result[result["date"] == pd.Timestamp("2024-05-15")]

    assert not ramadan_row.empty, "2024-03-20 not found after feature engineering"
    assert ramadan_row["is_ramadan"].iloc[0] == 1, "Expected is_ramadan=1 on 2024-03-20"

    assert not non_ramadan_row.empty, "2024-05-15 not found after feature engineering"
    assert non_ramadan_row["is_ramadan"].iloc[0] == 0, "Expected is_ramadan=0 on 2024-05-15"


def test_navruz_flag_fires_correctly():
    """is_navruz must be 1 on March 21 and 0 on other days."""
    df = _make_sample_df(n_days=90, start="2024-02-01")
    result = build_features(df)

    navruz_row = result[result["date"] == pd.Timestamp("2024-03-21")]
    non_navruz_row = result[result["date"] == pd.Timestamp("2024-03-15")]

    assert not navruz_row.empty
    assert navruz_row["is_navruz"].iloc[0] == 1

    assert not non_navruz_row.empty
    assert non_navruz_row["is_navruz"].iloc[0] == 0
