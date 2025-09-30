import pandas as pd
import numpy as np

# -------------------------
# 1. Load raw datasets
# -------------------------
df_energy = pd.read_csv("data/raw/owid-energy-data.csv")
df_co2 = pd.read_csv("data/raw/owid-co2-data.csv")

# -------------------------
# 2. Select & rename columns
# -------------------------
energy_cols = [
    "iso_code", "country", "year", "population", "gdp",
    "primary_energy_consumption", "coal_consumption", "oil_consumption",
    "gas_consumption", "renewables_consumption", "electricity_generation"
]

co2_cols = [
    "iso_code", "year", "co2", "coal_co2", "oil_co2",
    "gas_co2", "cement_co2", "flaring_co2", "consumption_co2"
]

df_energy = df_energy[energy_cols].copy()
df_co2 = df_co2[co2_cols].copy()

# Rename columns for clarity
df_energy.rename(columns={
    "primary_energy_consumption": "primary_energy_TWh",
    "coal_consumption": "coal_TWh",
    "oil_consumption": "oil_TWh",
    "gas_consumption": "gas_TWh",
    "renewables_consumption": "renewables_TWh",
    "electricity_generation": "electricity_TWh"
}, inplace=True)

df_co2.rename(columns={
    "co2": "total_CO2_Mt",
    "coal_co2": "coal_CO2_Mt",
    "oil_co2": "oil_CO2_Mt",
    "gas_co2": "gas_CO2_Mt",
    "cement_co2": "cement_CO2_Mt",
    "flaring_co2": "flaring_CO2_Mt",
    "consumption_co2": "consumption_CO2_Mt"
}, inplace=True)

# -------------------------
# 3. Standardize units
# -------------------------
energy_columns = ["primary_energy_TWh", "coal_TWh", "oil_TWh", "gas_TWh", "renewables_TWh", "electricity_TWh"]
for col in energy_columns:
    if df_energy[col].max() > 1e6:  # likely MWh
        df_energy[col] = df_energy[col] / 1e6  # convert to TWh

# -------------------------
# 4. Merge datasets
# -------------------------
df_merged = df_energy.merge(df_co2, on=["iso_code", "year"], how="left")

# -------------------------
# 5. Handle missing data
# -------------------------
# Interpolate short gaps (≤2 years) per country
df_merged = df_merged.sort_values(["iso_code", "year"])

columns_to_interp = energy_columns + [
    "total_CO2_Mt","coal_CO2_Mt","oil_CO2_Mt","gas_CO2_Mt",
    "cement_CO2_Mt","flaring_CO2_Mt","consumption_CO2_Mt"
]

df_merged[columns_to_interp] = df_merged.groupby("iso_code")[columns_to_interp].transform(
    lambda x: x.interpolate(limit=2)
)

# Optional: flag countries with longer gaps
long_gaps = df_merged.groupby("iso_code")[columns_to_interp].apply(
    lambda g: g.isna().any()
)
print("Countries with long gaps (>2 years):")
print(long_gaps[long_gaps.any(axis=1)])

# -------------------------
# 6. Save outputs
# -------------------------
df_energy.to_csv("data/processed/detailed_energy.csv", index=False)
df_co2.to_csv("data/processed/detailed_co2.csv", index=False)
df_merged.to_csv("data/processed/joined_energy_co2.csv", index=False)

