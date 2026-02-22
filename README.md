# 🇺🇿 Uzbekistan Payment Volume Forecasting

> **Production-grade time-series forecasting system predicting daily digital payment transaction volumes across Uzbekistan's fast-growing payment ecosystem — built with XGBoost, CatBoost, Linear Regression, FastAPI, and Docker.**

---

🌐 **Language / Til:** &nbsp; [🇬🇧 English](#-project-overview) &nbsp;|&nbsp; [🇺🇿 O'zbekcha](#-loyiha-haqida)

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
git clone https://github.com/Uzbekswe/uzbekistan-payment-forecasting
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

- GitHub: [github.com/Uzbekswe](https://github.com/Uzbekswe)
- LinkedIn: [linkedin.com/in/bakhodirovs](https://www.linkedin.com/in/bakhodirovs)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Data disclaimer: Transaction volume data is synthetically generated and calibrated to publicly available CBU aggregate statistics. USD/UZS exchange rates are sourced directly from the Central Bank of Uzbekistan's public API. This project is for portfolio and educational purposes.*

---
---

# 🇺🇿 O'zbekiston To'lov Hajmini Prognozlash

> **O'zbekistonning jadal rivojlanayotgan raqamli to'lov ekotizimida kunlik tranzaksiya hajmlarini bashorat qiluvchi ishlab chiqarish darajasidagi tizim — XGBoost, CatBoost, Chiziqli Regressiya, FastAPI va Docker yordamida qurilgan.**

---

## 📌 Loyiha Haqida

O'zbekiston raqamli to'lov bozori Markaziy Osiyodagi eng tez rivojlanayotgan bozorlardan biridir. Tezkor To'lov Tizimi **2025-yil o'rtasida $10.2 milliarddan ortiq tranzaksiyalarni qayta ishladi**, Payme, Click, Uzum va Paynet kabi platformalar o'nlab millionlab foydalanuvchilarga xizmat ko'rsatmoqda. Tranzaksiya hajmlarini aniq prognozlash quyidagиlar uchun juda muhim:

- **Quvvatni rejalashtirish** — to'lov platformasi ertaga qancha serverga muhtoj?
- **Likvidlikni boshqarish** — bank qancha aylanma mablag' saqlashi kerak?
- **Anomaliyalarni aniqlash** — bugungi hajm shubhali darajada past yoki yuqorimi?
- **Biznes-tahlil** — qaysi mavsumlar, bayramlar va iqtisodiy voqealar to'lov xulq-atvorini belgilaydi?

Ushbu loyiha xom ma'lumotlardan joylashtirilgan REST API gacha bo'lgan to'liq, ishlab chiqarish uslubidagi prognozlash quvurini yaratadi.

---

## 🎯 Loyiha Nimalарni Ko'rsatadi

| Ko'nikma | Amalga oshirish |
|---|---|
| Vaqt qatori xususiyatlar muhandisligi | Kechikish xususiyatlari, harakatlanuvchi oynalar, sinus/kosinus davriy kodlash |
| O'zbekiston bo'yicha bilim | Ramazon, Navro'z, maosh kunlari tsikllari, COVID ta'siri |
| Modellarni solishtirish | Chiziqli Regressiya vs XGBoost vs CatBoost |
| Haqiqiy ma'lumotlarni integratsiya | O'zbekiston Markaziy Bankidan jonli USD/UZS kurslari |
| Modelni tushuntirish | SHAP xususiyat ahamiyati tahlili |
| Ishlab chiqarishda ML xizmati | Kirishni tekshirish bilan FastAPI REST endpointi |
| Konteynerizatsiya | Docker + docker-compose orqali joylashtirish |
| Baholash qat'iyligi | Vaqt segmentlari bo'yicha MAE, RMSE, SMAPE |

---

## 🏗️ Arxitektura

```
O'zbekiston Markaziy Banki API (cbu.uz)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│              MA'LUMOTLAR QATLAMI                     │
│  Sintetik To'lov Hajmlari + Haqiqiy CBU Valyuta      │
│  Kurslari (2019–2024, 2,192 kunlik yozuv)           │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│         XUSUSIYATLAR MUHANDISLIGI QUVURI             │
│  • Vaqtinchalik: kun/hafta/oy/chorak kodlash         │
│  • Davriy: mavsumiylik uchun sinus/kosinus           │
│  • UZ Kalendari: Ramazon, Navro'z, maosh bayroqlari  │
│  • Kechikish xususiyatlari: t-1, t-7, t-14, t-30    │
│  • Harakatlanuvchi statistika: 7/14/30 kunlik        │
│  • Makro: USD/UZS kursi, 7 kunlik kurs o'zgarishi    │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│              MODELLARNI O'QITISH                     │
│                                                      │
│  ┌─────────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  Chiziqli   │  │ XGBoost  │  │  CatBoost    │   │
│  │ Regressiya  │  │          │  │              │   │
│  │  (bazaviy)  │  │(ansambl) │  │  (MDA-dan)   │   │
│  └─────────────┘  └──────────┘  └──────────────┘   │
│                                                      │
│      O'qitish: 2019–2022 | Sinov: 2023–2024         │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│           BAHOLASH VA TUSHUNTIRISH                   │
│  • Har bir model uchun MAE, RMSE, SMAPE             │
│  • Prognoz va haqiqiy qiymatlar grafigi              │
│  • SHAP xususiyat ahamiyati                          │
│  • Qoldiq tahlil                                     │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│           ISHLAB CHIQARISHDA XIZMAT KO'RSATISH       │
│                                                      │
│   POST /predict → FastAPI → Eng Yaxshi Model        │
│   GET  /health  → Holat tekshiruvi                  │
│   GET  /models  → Mavjud modellar + metrikalar      │
│                                                      │
│   Docker + docker-compose orqali konteynerizatsiya  │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Ma'lumotlar

### Sintetik To'lov Hajmlari
Tranzaksiya hajmi ma'lumotlari **sintetik tarzda yaratilgan, lekin** O'zbekistonning raqamli to'lov ekotizimi haqidagi CBU Statistik Byulletenlari va ommaviy hisobotlarda e'lon qilingan umumiy raqamlarga **moslashtirilgan**. Asosiy kodlangan naqshlar:

- **Eksponensial o'sish trendi** — UZ raqamli to'lovlarni joriy etishni aks ettiradi (kuniga 500K → 3.6M tranzaksiya, 2019–2024)
- **Haftalik mavsumiylik** — dam olish kunlarida O'zbekistonda hajm ~22% past bo'ladi
- **Ramazon ta'siri** — muqaddas oy davomida ~12% kamayish (sanalar har yili o'zgaradi)
- **Navro'z o'sishi** — 20–23 mart, ~35% hajm oshishi (milliy bayram)
- **Maosh kunlari tsikllari** — har oyning 10 va 25-sanalarida ~22% o'sish
- **COVID-19 pasayishi** — 2020-yil II choragi ~35% hajm kamayishini ko'rsatadi
- **Hududiy taqsimot** — Toshkent (45%), Samarqand (20%), Farg'ona (15%), Namangan (10%), Andijon (10%)

### Haqiqiy Valyuta Kursi Ma'lumotlari
USD/UZS valyuta kurslari bevosita **O'zbekiston Markaziy Bankining ommaviy API**sidan (`cbu.uz/en/arkhiv-kursov-valyut/json/{date}/`) 2019–2024 yillar uchun olinadi. Bu haqiqiy makroiqtisodiy signal beradi — valyuta qadrsizlanishi iste'molchi xarajatlar xulq-atvoridagi o'zgarishlar bilan bog'liq.

**Valyuta kursi diapazoni:** ~8,400 → ~12,700 so'm / 1 USD (2019–2024)

---

## 🔧 Xususiyatlar Muhandisligi

Xom sana + iqtisodiy signallardan 28 ta xususiyat yaratildi:

**Vaqtinchalik xususiyatlar** — `days_elapsed`, `day_of_week`, `month`, `quarter`, `week_of_year`, `day_of_month`

**Davriy kodlash** (ishlab chiqarish prognozlash tizimlarida qo'llaniladigan texnika) — `month_sin/cos`, `day_of_week_sin/cos`, `day_of_month_sin/cos`. Bu model dekabr (12) va yanvar (1) o'rtasida katta masofa borligini o'ylamasligini ta'minlaydi.

**O'zbekiston kalendar bayroqlari** — `is_ramadan`, `is_navruz`, `is_payday`, `is_weekend`, `is_covid_dip`

**Kechikish xususiyatlari** — `volume_lag_1`, `volume_lag_7`, `volume_lag_14`, `volume_lag_30` (1/7/14/30 kun oldin nima bo'lganini ko'rsatadi)

**Harakatlanuvchi statistika** — 7/14/30 kunlik harakatlanuvchi o'rtacha va standart og'ish (trend + o'zgaruvchanlik signallari)

**Makro xususiyatlar** — `usd_uzs_rate`, `usd_uzs_rate_lag7`, `usd_uzs_rate_change` (haqiqiy CBU ma'lumotlari)

---

## 🤖 Modellar

### 1. Chiziqli Regressiya (Bazaviy)
Standart OLS regressiya. Ishlash samaradorligining quyi chegarasini belgilaydi. Umumiy tendentsiyani yaxshi ushlab turadi, lekin chiziqli bo'lmagan mavsumiy naqshlar bilan kurasha olmaydi.

### 2. XGBoost
Gradient kuchaytirish daraxtlari. Xususiyatlar o'rtasidagi chiziqli bo'lmagan o'zaro ta'sirlarni ushlaydi (masalan, Ramazon + dam olish kuni kombinatsiyasi ularning har biridan alohida farq qiladi). Bunday turdagi ma'lumotlarda chiziqli bazaviyga nisbatan odatda 25–40% yaxshi SMAPE ko'rsatadi.

### 3. CatBoost
Yandex tomonidan ishlab chiqilgan gradient kuchaytirish kutubxonasi — MDAʼdagi kompaniyalar uchun ayniqsa dolzarb (Yandex texnologiyasi O'zbekiston texnologiya ekotizimiga chuqur singib ketgan). Kategorik xususiyatlarni nativ tarzda boshqaradi va ko'pincha kamroq gipeparametr sozlash bilan jadval ma'lumotlarida XGBoostdan ustun keladi.

### O'qitish/Sinov Bo'linishi
- **O'qitish:** 2019-01-01 → 2022-12-31 (4 yil)
- **Sinov:** 2023-01-01 → 2024-12-31 (2 yil, o'qitish davomida hech qachon ko'rilmagan)

Bu **walk-forward** bo'linish — model har doim kelajakdagi ma'lumotlarda sinovdan o'tkaziladi, tasodifiy aralashtirish emas (bu vaqt qatorlarida ma'lumotlar sizib chiqishiga olib keladi).

---

## 📈 Baholash Metrikалari

| Metrika | Nima o'lchanadi |
|---|---|
| **MAE** | Tranzaksiya sonidagi o'rtacha mutlaq xato |
| **RMSE** | Katta xatolarni ko'proq jazolaydi |
| **SMAPE** | Simmetrik foizli xato — masshtabdan mustaqil, modellar o'rtasida taqqoslanadi |

Natijalar ham umumiy, ham yil, chorak va O'zbekistonga xos davrlar (Ramazon, Navro'z, COVID) bo'yicha hisoblanadi.

---

## 🚀 Tez Boshlash

### 1-variant: Mahalliy ishlatish

```bash
# Klonlash va sozlash
git clone https://github.com/Uzbekswe/uzbekistan-payment-forecasting
cd uzbekistan-payment-forecasting
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Ma'lumotlar yaratish
python data/generate_data.py
python data/fetch_exchange_rates.py   # haqiqiy CBU ma'lumotlarini oladi (~3 daqiqa)

# Xususiyatlar yaratish va modellarni o'qitish
python src/train.py

# Baholash va grafiklar yaratish
python src/evaluate.py

# APIni ishga tushirish
uvicorn api.main:app --reload --port 8000
```

### 2-variant: Docker (tavsiya etiladi)

```bash
docker-compose up --build
```

API `http://localhost:8000` manzilida mavjud bo'ladi

---

## 🌐 API Ma'lumotnomasi

### `GET /health`
```json
{ "status": "healthy", "models_loaded": ["linear", "xgboost", "catboost"] }
```

### `GET /models`
Mavjud modellar va ularning sinov to'plamidagi metrikalarini (MAE, RMSE, SMAPE) qaytaradi.

### `POST /predict`
```json
// So'rov
{
  "date": "2025-03-21",
  "model": "catboost"
}

// Javob
{
  "date": "2025-03-21",
  "predicted_volume": 2847392,
  "model_used": "catboost",
  "is_navruz": true,
  "is_ramadan": false,
  "confidence_note": "Navro'z bayrami aniqlandi — hajm oshishi kutilmoqda"
}
```

---

## 🛠️ Texnologiya To'plami

| Qatlam | Texnologiya |
|---|---|
| Dasturlash tili | Python 3.11 |
| ML Modellari | scikit-learn, XGBoost, CatBoost |
| Tushuntirish | SHAP |
| Xususiyatlar muhandisligi | pandas, numpy |
| API | FastAPI + Uvicorn |
| Konteynerizatsiya | Docker, docker-compose |
| Vizualizatsiya | matplotlib, seaborn |
| Ma'lumot manbai | O'zbekiston MBning ommaviy API |

---

## 💡 Asosiy Xulosalar va Dizayn Qarorlari

**Nega LSTM/Prophet emas?** Sinchiklab ishlab chiqilgan kechikish xususiyatlari bilan XGBoost bunday hajmdagi jadval vaqt qatorlari ma'lumotlarida chuqur o'rganish modellaridan doimiy ravishda ustun keladi. Prophet bir vaqtning o'zida bir nechta mavsumiy naqshlar bilan (haftalik + Ramazon + Navro'z) kurasha olmaydi.

**Nega aynan CatBoost?** CatBoost Yandex tomonidan ishlab chiqilgan bo'lib, O'zbekiston va MDA bozorlarida chuqur ildiz otgan. Ushbu ekotizimdan o'sib chiqqan Uzum kabi kompaniyalar CatBoostni biladi va ko'pincha ishlab chiqarish joylashtirishida afzal ko'radi.

**Nega sintetik ma'lumotlar?** O'zbekistonning to'lov platformalari (Payme, Click, Uzum) batafsil kunlik tranzaksiya ma'lumotlarini e'lon qilmaydi. Sintetik ma'lumotlar CBU tomonidan e'lon qilingan yillik umumiy ko'rsatkichlarga moslashtirilgan va haqiqiy makroiqtisodiy signallar (haqiqiy USD/UZS kurslari) bilan boyitilgan. Bu real dunyo ML amaliyotini aks ettiradi — ma'lumotlar olimlari ko'pincha ishlab chiqarish ma'lumotlariga kirish huquqi berilishidan oldin sintetik ma'lumotlarda quvurlarni quradi va tekshiradi.

---

## 📋 O'zbekiston Ish Bozorida Ahamiyati

Ushbu loyiha quyidagi kompaniyalardagi muhandislik lavozimlariga to'g'ridan-to'g'ri qo'llaniladi:

- **Uzum** — talab prognoziga muhtoj 20 million oylik foydalanuvchi bilan bozor + BNPL + raqamli bank
- **Alif** — asosiy biznes likvidlik uchun tranzaksiya hajmini prognozlashga bog'liq
- **Payme / Click / Paynet** — quvvatni rejalashtirish zarur bo'lgan to'lov platformalari
- **Ipoteka Bank / Hamkorbank** — bank sektorida ML dasturlari
- **EPAM O'zbekiston** — fintech yechimlarini quradigan ML muhandislarini faol qabul qilmoqda

---

## 👤 Muallif

**Mukhammadali**
Sun'iy intellekt va Katta Ma'lumotlar bo'yicha bakalavr darajasi
O'zbekistonning rivojlanayotgan texnologiya ekotizimi uchun ishlab chiqarish ML tizimlarini qurmoqda.

- GitHub: [github.com/Uzbekswe](https://github.com/Uzbekswe)
- LinkedIn: [linkedin.com/in/bakhodirovs](https://www.linkedin.com/in/bakhodirovs)

---

## 📄 Litsenziya

MIT Litsenziyasi — foydalanish, o'zgartirish va tarqatish bepul.

---

*Ma'lumotlar bo'yicha eslatma: Tranzaksiya hajmi ma'lumotlari sintetik tarzda yaratilgan va CBU tomonidan e'lon qilingan umumiy statistikaga moslashtirilgan. USD/UZS valyuta kurslari bevosita O'zbekiston Markaziy Bankining ommaviy API'sidan olingan. Ushbu loyiha portfolio va ta'lim maqsadlari uchun mo'ljallangan.*
