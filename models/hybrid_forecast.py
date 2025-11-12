import pandas as pd
from prophet import Prophet
import lightgbm as lgb
import numpy as np

# ==========================================================
# 1️⃣ Load Clean Data
# ==========================================================
energy_file = "clean_energy_co2_lightgbm.csv"
prophet_file = "clean_energy_co2_prophet.csv"

print("✅ Loading input data...")
df_energy = pd.read_csv(energy_file)
df_prophet = pd.read_csv(prophet_file)

countries = df_energy["country"].unique()
target_year = 2035  # fixed end year
results = []

print(f"🌍 Found {len(countries)} countries. Forecasting Energy + CO₂ until {target_year}...")

# ==========================================================
# 2️⃣ Per-country Prophet + LightGBM Modeling
# ==========================================================
for c in countries:
    c_energy = df_energy[df_energy["country"] == c].copy()
    c_prophet = df_prophet[df_prophet["country"] == c].copy()

    if c_energy.empty or c_prophet.empty:
        continue

    c_energy = c_energy.sort_values("year")
    c_prophet["ds"] = pd.to_datetime(c_prophet["ds"], errors="coerce")

    last_year = int(c_energy["year"].max())
    forecast_horizon = max(0, target_year - last_year)
    if forecast_horizon == 0:
        print(f"⚠️ {c}: already up to {target_year}, skipping Prophet forecast.")
        continue

    # ------------------------------
    # Prophet Forecast for CO₂
    # ------------------------------
    try:
        m_co2 = Prophet()
        m_co2.fit(c_prophet)
        future_co2 = m_co2.make_future_dataframe(periods=forecast_horizon, freq="YE")
        fc_co2 = m_co2.predict(future_co2)
        co2_forecast = fc_co2[["ds", "yhat"]].rename(columns={"yhat": "yhat_co2"})
        co2_forecast["year"] = co2_forecast["ds"].dt.year
    except Exception as e:
        print(f"⚠️ Prophet failed for CO₂ ({c}): {e}")
        continue

    # ------------------------------
    # Prophet Forecast for Energy
    # ------------------------------
    if "primary_energy_consumption_energy" not in c_energy.columns:
        print(f"⚠️ Energy column missing for {c}")
        continue

    energy_df = c_energy[["year", "primary_energy_consumption_energy"]].dropna()
    if len(energy_df) < 3:
        print(f"⚠️ Not enough data for energy in {c}")
        continue

    energy_df["ds"] = pd.to_datetime(energy_df["year"], format="%Y")
    energy_df = energy_df.rename(columns={"primary_energy_consumption_energy": "y"})

    try:
        m_energy = Prophet()
        m_energy.fit(energy_df)
        future_energy = m_energy.make_future_dataframe(periods=forecast_horizon, freq="YE")
        fc_energy = m_energy.predict(future_energy)
        energy_forecast = fc_energy[["ds", "yhat"]].rename(columns={"yhat": "yhat_energy"})
        energy_forecast["year"] = energy_forecast["ds"].dt.year
    except Exception as e:
        print(f"⚠️ Prophet failed for energy ({c}): {e}")
        continue

    # ------------------------------
    # LightGBM Forecasts
    # ------------------------------
    features = [
        "gdp_energy", "population_energy",
        "energy_per_capita_energy", "energy_intensity",
        "co2_intensity", "primary_energy_consumption_energy"
    ]
    X = c_energy[features].fillna(0)
    y_co2 = c_energy["co2"].fillna(0)
    y_energy = c_energy["primary_energy_consumption_energy"].fillna(0)

    if X.empty or len(X) < 5:
        print(f"⚠️ Not enough valid rows for LightGBM in {c}")
        continue

    model_co2 = lgb.LGBMRegressor(n_estimators=100)
    model_co2.fit(X, y_co2)
    co2_pred = model_co2.predict(X)

    model_energy = lgb.LGBMRegressor(n_estimators=100)
    model_energy.fit(X, y_energy)
    energy_pred = model_energy.predict(X)

    # ------------------------------
    # Hybrid Forecast Merge (up to 2035)
    # ------------------------------
    full_years = np.arange(c_energy["year"].min(), target_year + 1)
    merged = pd.DataFrame({"country": c, "year": full_years})

    merged["yhat"] = np.interp(full_years, co2_forecast["year"], co2_forecast["yhat_co2"])
    merged["lightgbm_pred"] = np.interp(full_years, c_energy["year"], co2_pred)
    merged["hybrid_forecast"] = (merged["yhat"] + merged["lightgbm_pred"]) / 2

    merged["energy_yhat"] = np.interp(full_years, energy_forecast["year"], energy_forecast["yhat_energy"])
    merged["energy_lightgbm_pred"] = np.interp(full_years, c_energy["year"], energy_pred)
    merged["energy_forecast"] = (merged["energy_yhat"] + merged["energy_lightgbm_pred"]) / 2

    results.append(merged)
    print(f"✅ Completed forecasts for {c}")

# ==========================================================
# 3️⃣ Save Output
# ==========================================================
if results:
    final = pd.concat(results, ignore_index=True)
    final.to_csv("hybrid_forecast.csv", index=False)
    print(f"✅ Saved hybrid_forecast.csv with both CO₂ and Energy forecasts up to {target_year}.")
    print(final.groupby("country")["year"].max())
else:
    print("❌ No valid forecasts generated. Please check input data.")
