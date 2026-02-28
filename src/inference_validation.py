import pandas as pd
from datetime import timedelta
from src.predict import load_models, make_prediction
import warnings
warnings.filterwarnings("ignore")

def test_2026_forecast():
    print("Loading models and historical data...")
    models = load_models()
    
    # We need to simulate step-by-step to build the lag features correctly
    start_date = models["history_df"]["date"].max() + timedelta(days=1)
    end_date = pd.Timestamp("2026-03-25")
    
    print(f"Iteratively generating forecasts from {start_date.date()} to {end_date.date()}...")
    
    current_dt = start_date
    model_name = "catboost"
    
    predictions_2026 = []
    
    while current_dt <= end_date:
        date_str = current_dt.strftime("%Y-%m-%d")
        
        # Predict the current day
        res = make_prediction(date_str, model_name, models)
        pred_vol = res["predicted_volume"]
        
        # We need to append this prediction back into the history_df so the next day
        # can use it for rolling means (e.g. volume_rolling_mean_7, volume_lag_1)
        last_row = models["history_df"].iloc[-1].copy()
        last_row["date"] = current_dt
        last_row["transaction_volume"] = pred_vol
        
        # Append to history
        models["history_df"] = pd.concat([models["history_df"], pd.DataFrame([last_row])], ignore_index=True)
        
        # Save 2026 dates for printing
        if current_dt.year == 2026:
            predictions_2026.append(res)
            
        current_dt += timedelta(days=1)

    print()
    print("="*60)
    print("🔮 2026 PREDICTION SAMPLE (March 2026)")
    print("="*60)
    for r in predictions_2026[-15:]: # Show mid-to-late March 2026
        print(f"Date: {r['date']} | Vol: {r['predicted_volume']:>9,} | {r['confidence_note']}")
    print("="*60)

if __name__ == "__main__":
    test_2026_forecast()
