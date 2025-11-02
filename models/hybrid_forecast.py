import pandas as pd
from prophet import Prophet
import lightgbm as lgb
import numpy as np

# ==========================================================
# Load Clean Data
# ==========================================================
energy_file = "clean_energy_co2_lightgbm.csv"
prophet_file = "clean_energy_co2_prophet.csv"

print("✅ Loading input data...")
df_energy = pd.read_csv(energy_file)
df_prophet = pd.read_csv(prophet_file)

countries = df_energy["country"].unique()
forecast_horizon = 10
results = []

print(f"🌍 Found {len(countries)} countries. Forecasting Energy + CO₂...")

# ==========================================================
# Per-country Prophet + LightGBM modeling
# ==========================================================
for c in countries:
    c_energy = df_energy[df_energy["country"] == c].copy()
    c_prophet = df_prophet[df_prophet["country"] == c].copy()

    if c_energy.empty or c_prophet.empty:
        continue

    # Ensure proper sorting and dtype
    c_energy = c_energy.sort_values("year")
    c_prophet["ds"] = pd.to_datetime(c_prophet["ds"], errors="coerce")

    # --------------------------------------
    # Prophet forecast for CO₂
    # --------------------------------------
    try:
        m_co2 = Prophet()
        m_co2.fit(c_prophet)
        future_co2 = m_co2.make_future_dataframe(periods=forecast_horizon, freq="Y")
        fc_co2 = m_co2.predict(future_co2)
        co2_forecast = fc_co2[["ds", "yhat"]].rename(columns={"yhat": "yhat_co2"})
        co2_forecast["year"] = co2_forecast["ds"].dt.year
    except Exception as e:
        print(f"⚠️ Prophet failed for CO₂ ({c}): {e}")
        continue

    # --------------------------------------
    # Prophet forecast for Energy
    # --------------------------------------
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
        future_energy = m_energy.make_future_dataframe(periods=forecast_horizon, freq="Y")
        fc_energy = m_energy.predict(future_energy)
        energy_forecast = fc_energy[["ds", "yhat"]].rename(columns={"yhat": "yhat_energy"})
        energy_forecast["year"] = energy_forecast["ds"].dt.year
    except Exception as e:
        print(f"⚠️ Prophet failed for energy ({c}): {e}")
        continue

    # --------------------------------------
    # LightGBM forecast (CO₂)
    # --------------------------------------
    features = [
        "gdp_energy", "population_energy",
        "energy_per_capita_energy", "energy_intensity",
        "co2_intensity", "primary_energy_consumption_energy"
    ]
    X = c_energy[features].fillna(0)
    y = c_energy["co2"].fillna(0)

    if len(X) < 5:
        continue

    model_co2 = lgb.LGBMRegressor(n_estimators=100)
    model_co2.fit(X, y)
    co2_pred = model_co2.predict(X)

    # --------------------------------------
    # LightGBM forecast (Energy)
    # --------------------------------------
    y_energy = c_energy["primary_energy_consumption_energy"].fillna(0)
    model_energy = lgb.LGBMRegressor(n_estimators=100)
    model_energy.fit(X, y_energy)
    energy_pred = model_energy.predict(X)

    # --------------------------------------
    # Hybrid forecast merge
    # --------------------------------------
    merged = pd.DataFrame({
        "country": c,
        "year": c_energy["year"].values,
        "yhat": np.interp(c_energy["year"], co2_forecast["year"], co2_forecast["yhat_co2"]),
        "lightgbm_pred": co2_pred,
        "hybrid_forecast": (np.interp(c_energy["year"], co2_forecast["year"], co2_forecast["yhat_co2"]) + co2_pred) / 2,
        "energy_yhat": np.interp(c_energy["year"], energy_forecast["year"], energy_forecast["yhat_energy"]),
        "energy_lightgbm_pred": energy_pred,
        "energy_forecast": (np.interp(c_energy["year"], energy_forecast["year"], energy_forecast["yhat_energy"]) + energy_pred) / 2
    })

    results.append(merged)

# ==========================================================
# Save Output
# ==========================================================
final = pd.concat(results, ignore_index=True)
final.to_csv("hybrid_forecast.csv", index=False)
print("✅ Saved hybrid_forecast.csv with both CO₂ and Energy forecasts.")
