import requests
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading

# Paths relative to this script — works no matter where you run it from
DATA_DIR = Path(__file__).parent
PAYMENTS_CSV = DATA_DIR / "uzbekistan_payments.csv"
RATES_CSV = DATA_DIR / "exchange_rates_cbu.csv"
ENRICHED_CSV = DATA_DIR / "uzbekistan_payments_enriched.csv"

# Thread-local storage so each worker thread uses its own requests.Session
_thread_local = threading.local()


def _get_session() -> requests.Session:
    """Return a per-thread requests.Session (avoids thread-safety issues)."""
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def _fetch_single_date(date_str: str, retries: int = 3) -> dict | None:
    """Fetch USD/UZS rate for one date from CBU. Returns None on failure."""
    url = f"https://cbu.uz/en/arkhiv-kursov-valyut/json/USD/{date_str}/"
    session = _get_session()

    for attempt in range(retries):
        try:
            response = session.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                for item in data:
                    if item.get("Ccy") == "USD":
                        return {
                            "date": date_str,
                            "usd_uzs_rate": float(item.get("Rate", 0)),
                        }
        except Exception as e:
            if attempt == retries - 1:
                print(f"  ⚠️  Failed for {date_str} after {retries} attempts: {e}")
    return None


def fetch_cbu_exchange_rates(
    start_date: str, end_date: str, max_workers: int = 20
) -> pd.DataFrame:
    """
    Fetch USD/UZS exchange rates from Central Bank of Uzbekistan public API.
    API: https://cbu.uz/en/arkhiv-kursov-valyut/

    Uses a thread pool so all dates are fetched concurrently instead of one
    by one — reduces runtime from ~55 min to ~2–3 min for a 6-year range.
    """
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    all_dates = []
    while current <= end:
        all_dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    total = len(all_dates)
    print(
        f"Fetching CBU exchange rates {start_date} → {end_date} "
        f"({total} days, {max_workers} parallel workers)..."
    )

    records = []
    completed = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_single_date, d): d for d in all_dates
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                records.append(result)
            with lock:
                completed += 1
                if completed % 300 == 0:
                    print(f"  Progress: {completed}/{total} days fetched...")

    print(f"  Total records collected: {len(records)}")
    if not records:
        print("  ❌ No records collected! Check API or network connection.")
        return pd.DataFrame(columns=["date", "usd_uzs_rate"])

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"✅ Fetched {len(df)} exchange rate records")
    return df


def merge_with_payments(payments_path: Path, rates_df: pd.DataFrame) -> pd.DataFrame:
    payments = pd.read_csv(payments_path)
    payments["date"] = pd.to_datetime(payments["date"])

    merged = payments.merge(rates_df, on="date", how="left")

    # Forward-fill gaps (weekends / public holidays when CBU doesn't publish)
    merged["usd_uzs_rate"] = merged["usd_uzs_rate"].ffill()

    # Derived features
    merged["usd_uzs_rate_lag7"] = merged["usd_uzs_rate"].shift(7)
    merged["usd_uzs_rate_change"] = merged["usd_uzs_rate"].pct_change(7).round(6)

    print(f"✅ Merged dataset shape: {merged.shape}")
    print(
        f"   Exchange rate range: "
        f"{merged['usd_uzs_rate'].min():,.0f} — "
        f"{merged['usd_uzs_rate'].max():,.0f} UZS per USD"
    )
    return merged


if __name__ == "__main__":
    rates_df = fetch_cbu_exchange_rates("2019-01-01", "2024-12-31")
    rates_df.to_csv(RATES_CSV, index=False)
    print(f"   Saved → {RATES_CSV}")

    merged = merge_with_payments(PAYMENTS_CSV, rates_df)
    merged.to_csv(ENRICHED_CSV, index=False)
    print(f"   Saved → {ENRICHED_CSV}")

    print("\n📊 Sample of enriched data:")
    print(
        merged[
            ["date", "transaction_volume", "usd_uzs_rate", "usd_uzs_rate_change"]
        ].head(10)
    )
