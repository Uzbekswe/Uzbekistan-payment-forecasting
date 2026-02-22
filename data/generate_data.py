import pandas as pd
import numpy as np

np.random.seed(42)

def generate_uzbekistan_payment_data(): 
    dates = pd.date_range(start="2019-01-01", end="2024-12-31", freq="D")
    df = pd.DataFrame({'date': dates})

    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_month'] = df['date'].dt.day
    df['quarter'] = df['date'].dt.quarter

    # Base volume trend (exponential growth of UZ digital payments)
    days_elapsed = (df['date'] - df['date'].min()).dt.days
    base_volume = (
        500_000
        * np.exp(days_elapsed * 0.00085)            # exponential adoption curve
        * np.where(days_elapsed > 730,  1.35, 1.0)  # structural break 2021: FPS launch
        * np.where(days_elapsed > 1460, 1.20, 1.0)  # structural break 2023: Uzum Bank launch
    )

    # Weekly seasonality - weekends lower in UZ
    weekly = np.where(df['day_of_week'] >= 5, 0.78, 1.0)

    # Monthly seasonality - higher in end of month (Navruz), December (holiday shopping)
    monthly_factors = {1: 1.12, 2: 0.95, 3: 1.25, 4: 0.98,
                       5: 1.02, 6: 0.97, 7: 0.96, 8: 1.0,
                       9: 1.05, 10: 1.08, 11: 1.1, 12: 1.18}
    monthly = df['month'].map(monthly_factors).values

    # Ramadan effect - reduced daytime transactions, increased evening transactions(iftar time)
    # Approximate Ramadan windows (shifts each year)
    ramadan_dates = [
        ("2019-05-05", "2019-06-03"),
        ("2020-04-23", "2020-05-23"),
        ("2021-04-12", "2021-05-12"),
        ("2022-04-02", "2022-05-02"),
        ("2023-03-22", "2023-04-21"),
        ("2024-03-10", "2024-04-09"),
    ]
    ramadan_mask = np.zeros(len(df))
    for start, end in ramadan_dates: 
        mask = (df['date'] >= start) & (df['date'] <= end)
        ramadan_mask[mask] = 1.0
    df['is_ramadan'] = ramadan_mask
    ramadan_effect = np.where(ramadan_mask, 0.88, 1.0)

    # Payday spikes — 10th and 25th of month
    payday_mask = df["day_of_month"].isin([10, 25])
    df["is_payday"] = payday_mask.astype(int)
    payday_effect = np.where(payday_mask, 1.22, 1.0)

    # Navruz holiday (March 21)
    navruz_mask = (df["month"] == 3) & (df["day_of_month"].isin([20, 21, 22, 23]))
    df["is_navruz"] = navruz_mask.astype(int)
    navruz_effect = np.where(navruz_mask, 1.35, 1.0)

    # COVID dip (2020 Q2)
    covid_mask = (df["year"] == 2020) & (df["quarter"] == 2)
    covid_effect = np.where(covid_mask, 0.65, 1.0)

    # Noise and random fluctuations
    noise = np.random.normal(1.0, 0.04, size=len(df))

    df['transaction_volume'] = (
        base_volume * weekly * monthly * ramadan_effect * payday_effect * navruz_effect * covid_effect * noise
    ).astype(int)

    df["avg_transaction_value"] = (
        12_000 + days_elapsed * 3.5 +
        np.random.normal(0, 800, len(df))
    ).astype(int)  # in UZS (thousands)

    df['region'] = np.random.choice (
        ['Tashkent', 'Samarkand', 'Fergana', 'Namangan', 'Andijan'],
        size=len(df),
        p=[0.45, 0.20, 0.15, 0.10, 0.10]
    )

    return df

if __name__ == "__main__":
    df = generate_uzbekistan_payment_data()
    df.to_csv("data/uzbekistan_payments.csv", index=False)
    print(f"✅ Generated {len(df)} rows of data")
    print(df.head())
    print(df.describe())