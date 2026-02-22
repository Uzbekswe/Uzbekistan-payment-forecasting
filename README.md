# 🇺🇿 Uzbekistan Payment Volume Forecasting

> **Production-grade time-series forecasting system predicting daily digital payment transaction volumes across Uzbekistan's fast-growing payment ecosystem — built with XGBoost, CatBoost, Linear Regression, FastAPI, and Docker.**

---

## 📌 Project Overview

Uzbekistan's digital payment market is one of the fastest-growing in Central Asia. The Fast Payment System processed over **$10.2 billion in transactions by mid-2025**, with platforms like Payme, Click, Uzum, and Paynet collectively serving tens of millions of users. Accurate forecasting of transaction volumes is critical for:

- **Capacity planning** — how many servers does a payment platform need tomorrow?
- **Liquidity management** — how much float does a bank need to maintain?
- **Anomaly detection** — is today's volume suspiciously low or high?
- **Business intelligence** — which seasons, holidays, and economic events drive payment behavior?

This project builds a complete, production-style forecasting pipeline that answers these questions — from raw data to a deployed REST API endpoint.

---

## 🎯 What This Project Demonstrates

| Skill | Implementation |
|---|---|
| Time-series feature engineering | Lag features, rolling windows, sine/cosine cyclical encoding |
| Uzbekistan domain knowledge | Ramadan, Navruz, payday cycles, COVID impact |
| Model comparison | Linear Regression vs XGBoost vs CatBoost |
| Real data integration | Live USD/UZS rates from Central Bank of Uzbekistan API |
| Model explainability | SHAP feature importance analysis |
| Production ML serving | FastAPI REST endpoint with input validation |
| Containerization | Docker + docker-compose deployment |
| Evaluation rigor | MAE, RMSE, SMAPE across time segments |

---

## 🏗️ Architecture

```
Central Bank of Uzbekistan API (cbu.uz)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│              DATA LAYER                              │
│  Synthetic Payment Volumes + Real CBU Exchange Rates │
│  (2019–2024, 2,192 daily records)                   │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│           FEATURE ENGINEERING PIPELINE               │
│  • Temporal: day/week/month/quarter encoding         │
│  • Cyclical: sine/cosine for seasonality             │
│  • UZ Calendar: Ramadan, Navruz, payday flags        │
│  • Lag features: t-1, t-7, t-14, t-30               │
│  • Rolling stats: 7/14/30-day mean & std             │
│  • Macro: USD/UZS rate, 7-day rate change            │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│              MODEL TRAINING                          │
│                                                      │
│  ┌─────────────┐  ┌──────────┐  ┌──────────────┐   │
│  │   Linear    │  │ XGBoost  │  │  CatBoost    │   │
│  │ Regression  │  │          │  │              │   │
│  │ (baseline)  │  │(ensemble)│  │  (CIS-made)  │   │
│  └─────────────┘  └──────────┘  └──────────────┘   │
│                                                      │
│         Train: 2019–2022 | Test: 2023–2024          │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│           EVALUATION & EXPLAINABILITY                │
│  • MAE, RMSE, SMAPE per model                       │
│  • Forecast vs actual plots                          │
│  • SHAP feature importance                           │
│  • Residual analysis                                 │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│              PRODUCTION SERVING                      │
│                                                      │
│   POST /predict → FastAPI → Best Model → Forecast   │
│   GET  /health  → Status check                      │
│   GET  /models  → Available models + metrics        │
│                                                      │
│   Containerized with Docker + docker-compose        │
└─────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
uzbekistan-payment-forecasting/
│
├── data/
│   ├── generate_data.py              # Synthetic UZ payment data generator
│   ├── fetch_exchange_rates.py       # Live data from CBU API (cbu.uz)
│   ├── uzbekistan_payments.csv       # Raw synthetic data
│   ├── uzbekistan_payments_enriched.csv  # + Real CBU exchange rates
│   └── README.md                     # Data sources & methodology
│
├── notebooks/
│   └── 01_eda.ipynb                  # Exploratory analysis & visualizations
│
├── src/
│   ├── features.py                   # Feature engineering pipeline
│   ├── train.py                      # Train all 3 models & save artifacts
│   ├── evaluate.py                   # Metrics, SHAP plots, forecast charts
│   └── predict.py                    # Inference logic (used by API)
│
├── api/
│   └── main.py                       # FastAPI application
│
├── models/                           # Saved model files (.joblib)
│   └── .gitkeep
│
├── plots/                            # Output charts (for LinkedIn/README)
│   └── .gitkeep
│
├── Dockerfile                        # Multi-stage production Dockerfile
├── docker-compose.yml                # Full stack deployment
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 📊 Data

### Synthetic Payment Volumes
Transaction volume data is **synthetically generated but calibrated** to aggregate figures published in CBU Statistical Bulletins and public reports about Uzbekistan's digital payment ecosystem. Key patterns encoded:

- **Exponential growth trend** — reflecting UZ digital payment adoption (500K → 3.6M daily transactions, 2019–2024)
- **Weekly seasonality** — weekends see ~22% lower volumes in Uzbekistan
- **Ramadan effect** — ~12% reduction during the holy month (dates shift yearly)
- **Navruz spike** — March 20–23, ~35% volume increase (national holiday)
- **Payday cycles** — 10th and 25th of each month show ~22% spikes
- **COVID-19 dip** — Q2 2020 shows ~35% volume reduction
- **Regional distribution** — Tashkent (45%), Samarkand (20%), Fergana (15%), Namangan (10%), Andijan (10%)

### Real Exchange Rate Data
USD/UZS exchange rates are fetched directly from the **Central Bank of Uzbekistan's public API** (`cbu.uz/en/arkhiv-kursov-valyut/json/{date}/`) covering 2019–2024. This provides a genuine macroeconomic signal — currency depreciation correlates with shifts in consumer spending behavior.

**Exchange rate range:** ~8,400 → ~12,700 UZS per USD (2019–2024)

---

## 🔧 Feature Engineering

28 features engineered from raw date + economic signals:

**Temporal features** — `days_elapsed`, `day_of_week`, `month`, `quarter`, `week_of_year`, `day_of_month`

**Cyclical encoding** (same technique used in production forecasting systems) — `month_sin/cos`, `day_of_week_sin/cos`, `day_of_month_sin/cos`. These prevent the model from thinking December (12) and January (1) are far apart.

**Uzbekistan calendar flags** — `is_ramadan`, `is_navruz`, `is_payday`, `is_weekend`, `is_covid_dip`

**Lag features** — `volume_lag_1`, `volume_lag_7`, `volume_lag_14`, `volume_lag_30` (what happened 1/7/14/30 days ago)

**Rolling statistics** — 7/14/30-day rolling mean and standard deviation (trend + volatility signals)

**Macro features** — `usd_uzs_rate`, `usd_uzs_rate_lag7`, `usd_uzs_rate_change` (real CBU data)

---

## 🤖 Models

### 1. Linear Regression (Baseline)
Standard OLS regression. Establishes the performance floor. Good at capturing the overall trend but struggles with nonlinear seasonal patterns — exactly what your friend's project demonstrated with disease data.

### 2. XGBoost
Gradient boosted trees. Captures nonlinear interactions between features (e.g., the combination of Ramadan + weekend behaves differently than either alone). Typically 25–40% better SMAPE than linear baseline on this type of data.

### 3. CatBoost
Yandex's gradient boosting library — particularly relevant for CIS-region companies (Yandex technology is deeply embedded in Uzbekistan's tech ecosystem). Handles categorical features natively and often outperforms XGBoost on tabular data with less hyperparameter tuning.

### Train/Test Split
- **Training:** 2019-01-01 → 2022-12-31 (4 years)
- **Testing:** 2023-01-01 → 2024-12-31 (2 years, never seen during training)

This is a **walk-forward** split — the model is always tested on future data, never random shuffling (which would cause data leakage in time-series).

---

## 📈 Evaluation Metrics

| Metric | What it measures |
|---|---|
| **MAE** | Average absolute error in transaction count |
| **RMSE** | Penalizes large errors more heavily |
| **SMAPE** | Symmetric percentage error — scale-independent, comparable across models |

Results are computed both overall and broken down by year, quarter, and Uzbekistan-specific periods (Ramadan, Navruz, COVID).

---

## 🚀 Quick Start

### Option 1: Run locally

```bash
# Clone and set up
git clone https://github.com/yourusername/uzbekistan-payment-forecasting
cd uzbekistan-payment-forecasting
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Generate data
python data/generate_data.py
python data/fetch_exchange_rates.py   # fetches real CBU data (~3 min)

# Build features & train models
python src/train.py

# Evaluate & generate plots
python src/evaluate.py

# Start API
uvicorn api.main:app --reload --port 8000
```

### Option 2: Docker (recommended)

```bash
docker-compose up --build
```

API will be available at `http://localhost:8000`

---

## 🌐 API Reference

### `GET /health`
```json
{ "status": "healthy", "models_loaded": ["linear", "xgboost", "catboost"] }
```

### `GET /models`
Returns available models and their test set metrics (MAE, RMSE, SMAPE).

### `POST /predict`
```json
// Request
{
  "date": "2025-03-21",
  "model": "catboost"
}

// Response
{
  "date": "2025-03-21",
  "predicted_volume": 2847392,
  "model_used": "catboost",
  "is_navruz": true,
  "is_ramadan": false,
  "confidence_note": "Navruz holiday detected — expect elevated volumes"
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML Models | scikit-learn, XGBoost, CatBoost |
| Explainability | SHAP |
| Feature Engineering | pandas, numpy |
| API | FastAPI + Uvicorn |
| Containerization | Docker, docker-compose |
| Visualization | matplotlib, seaborn |
| Data Source | CBU Uzbekistan public API |

---

## 💡 Key Learnings & Design Decisions

**Why not use LSTM/Prophet?** XGBoost with carefully engineered lag features consistently outperforms deep learning models on tabular time-series data of this scale. Prophet struggles with multiple seasonal patterns (weekly + Ramadan + Navruz simultaneously).

**Why CatBoost specifically?** CatBoost is developed by Yandex, which has deep infrastructure roots in Uzbekistan and CIS markets. Companies like Uzum, which grew out of this ecosystem, are familiar with and often prefer CatBoost for production deployments.

**Why synthetic data?** Uzbekistan's payment platforms (Payme, Click, Uzum) do not publish granular daily transaction data. The synthetic data is calibrated to CBU-published aggregate annual figures and enriched with genuine macroeconomic signals (real USD/UZS rates). This mirrors real-world ML practice — data scientists often build and validate pipelines on synthetic data before production data access is granted.

---

## 📋 Relevance to Uzbekistan Job Market

This project is directly applicable to engineering roles at:

- **Uzum** — marketplace + BNPL + digital banking with 20M monthly users needing demand forecasting
- **Alif** — core business depends on transaction volume prediction for liquidity
- **Payme / Click / Paynet** — payment platforms needing capacity planning
- **Ipoteka Bank / Hamkorbank** — banking sector ML applications
- **EPAM Uzbekistan** — actively hiring ML Engineers building fintech solutions

---

## 👤 Author

**Mukhammadali**
Bachelor's degree in AI and Big Data
Building production ML systems for Uzbekistan's growing tech ecosystem.

- GitHub: [github.com/myusername](https://github.com/Uzbekswe)
- LinkedIn: [linkedin.com/in/yourprofile](www.linkedin.com/in/bakhodirovs)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Data disclaimer: Transaction volume data is synthetically generated and calibrated to publicly available CBU aggregate statistics. USD/UZS exchange rates are sourced directly from the Central Bank of Uzbekistan's public API. This project is for portfolio and educational purposes.*
