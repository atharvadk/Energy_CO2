# ==========================================
# 🌍 Hybrid CO₂ Forecasting (Prophet + LightGBM)
# ==========================================

import pandas as pd
import numpy as np
from prophet import Prophet
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# 1️⃣ Load preprocessed datasets
# -------------------------------
prophet_df = pd.read_csv("clean_energy_co2_prophet.csv")
lightgbm_df = pd.read_csv("clean_energy_co2_lightgbm.csv")

print(f"✅ Loaded Prophet data: {prophet_df.shape}")
print(f"✅ Loaded LightGBM data: {lightgbm_df.shape}")

# -------------------------------
# 2️⃣ Prepare storage for results
# -------------------------------
results = []

countries = sorted(set(prophet_df["country"]).intersection(set(lightgbm_df["country"])))
print(f"🌍 Training for {len(countries)} countries...")

# -------------------------------
# 3️⃣ Loop through each country
# -------------------------------
for country in countries:
    try:
        # -------------------------------
        # Prophet part
        # -------------------------------
        df_p = prophet_df[prophet_df["country"] == country][["ds", "y"]].dropna()
        if len(df_p) < 10:
            continue  # skip countries with too little data

        model_p = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False)
        model_p.fit(df_p)

        future = model_p.make_future_dataframe(periods=10, freq="Y")
        forecast = model_p.predict(future)
        forecast = forecast[["ds", "yhat"]]
        forecast["country"] = country
        forecast["source"] = "Prophet"

        # -------------------------------
        # LightGBM part
        # -------------------------------
        df_l = lightgbm_df[lightgbm_df["country"] == country].copy()
        if len(df_l) < 10:
            continue

        # Identify target column
        if "co2" in df_l.columns:
            y = df_l["co2"]
        elif "co2_co2" in df_l.columns:
            y = df_l["co2_co2"]
        else:
            continue

        feature_cols = [c for c in df_l.columns if c not in ["country", "year", "co2", "co2_co2"]]
        X = df_l[feature_cols]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model_lgb = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9
        )
        model_lgb.fit(X_train, y_train)

        preds_test = model_lgb.predict(X_test)
        score = r2_score(y_test, preds_test)
        print(f"✅ {country}: LightGBM R² = {score:.3f}")

        # Predict next 10 years beyond last known year
        last_year = int(df_l["year"].max())
        future_years = list(range(last_year + 1, last_year + 11))

        # Use last known feature row to extrapolate (simple assumption)
        last_row = df_l.iloc[-1:]
        future_rows = pd.concat([last_row] * 10, ignore_index=True)
        future_rows["year"] = future_years

        preds_future = model_lgb.predict(future_rows[feature_cols])
        df_future = pd.DataFrame({
            "country": country,
            "year": future_years,
            "lightgbm_pred": preds_future
        })

        # -------------------------------
        # Combine Prophet + LightGBM
        # -------------------------------
        merged = pd.merge(
            forecast,
            df_future,
            left_on=["country", forecast["ds"].dt.year],
            right_on=["country", "year"],
            how="inner"
        )

        merged = merged[["country", "year", "yhat", "lightgbm_pred"]]
        merged["hybrid_forecast"] = merged["yhat"] + (merged["lightgbm_pred"] - merged["lightgbm_pred"].mean())

        results.append(merged)

    except Exception as e:
        print(f"⚠️ {country}: Skipped due to error → {e}")

# -------------------------------
# 4️⃣ Combine all country forecasts
# -------------------------------
if results:
    hybrid_df = pd.concat(results, ignore_index=True)
    hybrid_df.to_csv("hybrid_forecast.csv", index=False)
    print("\n✅ Hybrid forecast successfully saved as 'hybrid_forecast.csv'")
else:
    print("❌ No forecasts generated (check data coverage)")

# -------------------------------
# 5️⃣ Sample output preview
# -------------------------------
if results:
    print("\n📊 Sample forecast:")
    print(hybrid_df.head(10))
