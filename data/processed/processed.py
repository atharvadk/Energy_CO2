import pandas as pd
import numpy as np
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)

# ===============================
# 1️⃣ Load datasets
# ===============================
energy = pd.read_csv("https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv")
co2 = pd.read_csv("https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv")

print(f"✅ Loaded datasets:\n  Energy: {energy.shape}\n  CO₂: {co2.shape}")

# ===============================
# 2️⃣ Filter post-1990 & top countries
# ===============================
energy = energy[energy["year"] >= 1990]
co2 = co2[co2["year"] >= 1990]

top = (
    energy.groupby("country")["primary_energy_consumption"]
    .max()
    .sort_values(ascending=False)
    .head(50)
    .index
)
energy = energy[energy["country"].isin(top)]
co2 = co2[co2["country"].isin(top)]

# ===============================
# 3️⃣ Merge
# ===============================
merged = pd.merge(
    energy,
    co2,
    on=["country", "year"],
    how="outer",
    suffixes=("_energy", "_co2"),
)

print(f"✅ Merged shape: {merged.shape}")

# ===============================
# 4️⃣ Clean missing values
# ===============================
merged = merged.sort_values(["country", "year"])
merged = merged.groupby("country", group_keys=False).apply(lambda g: g.ffill().bfill())
merged = merged.fillna(merged.median(numeric_only=True))

# ===============================
# 5️⃣ Derived features
# ===============================
def safe_div(num, denom):
    return num / (denom.replace(0, np.nan) + 1e-6)

merged["energy_per_capita_calc"] = safe_div(
    merged["primary_energy_consumption_energy"], merged["population_energy"]
)
merged["co2_per_capita_calc"] = safe_div(
    merged["co2"], merged["population_co2"]
)
merged["energy_intensity"] = safe_div(
    merged["primary_energy_consumption_energy"], merged["gdp_energy"]
)
merged["co2_intensity"] = safe_div(
    merged["co2"], merged["gdp_co2"]
)

# ===============================
# 6️⃣ Lag features
# ===============================
def create_lags(df, cols, lags=[1, 2, 3]):
    for c in cols:
        if c in df.columns:
            for l in lags:
                df[f"{c}_lag{l}"] = df.groupby("country")[c].shift(l)
    return df

lag_cols = [
    "primary_energy_consumption_energy",
    "co2",
    "energy_intensity",
    "co2_intensity",
]
merged = create_lags(merged, lag_cols)
merged = merged.dropna().reset_index(drop=True)

# ===============================
# 7️⃣ Prophet dataset
# ===============================
prophet = merged[["country", "year", "co2"]].rename(columns={"year": "ds", "co2": "y"})
prophet["ds"] = pd.to_datetime(prophet["ds"], format="%Y")

# ===============================
# 8️⃣ Save outputs
# ===============================
merged.to_csv("clean_energy_co2_lightgbm.csv", index=False)
prophet.to_csv("clean_energy_co2_prophet.csv", index=False)

print("\n✅ CSVs created successfully!")
print("  → clean_energy_co2_lightgbm.csv")
print("  → clean_energy_co2_prophet.csv")

print("\n📊 Prophet sample:")
print(prophet.head(3))
