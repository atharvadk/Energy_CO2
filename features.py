import pandas as pd

# 1️⃣ Load the joined dataset
df = pd.read_csv("data/processed/joined_energy_co2.csv")

# Ensure correct sorting for lag/rolling calculations
df = df.sort_values(['country', 'year']).reset_index(drop=True)

# 2️⃣ Core features: fuel shares
df['coal_share']       = df['coal_TWh'] / df['primary_energy_TWh']
df['oil_share']        = df['oil_TWh'] / df['primary_energy_TWh']
df['gas_share']        = df['gas_TWh'] / df['primary_energy_TWh']
df['renewables_share'] = df['renewables_TWh'] / df['primary_energy_TWh']
df['electricity_share']= df['electricity_TWh'] / df['primary_energy_TWh']

# 3️⃣ Core features: energy per capita & intensity
df['energy_per_capita'] = df['primary_energy_TWh'] / df['population']
df['energy_intensity']  = df['primary_energy_TWh'] / df['gdp']

# 4️⃣ Lag features for energy and CO2
df['energy_lag_1'] = df.groupby('country')['primary_energy_TWh'].shift(1)
df['energy_lag_2'] = df.groupby('country')['primary_energy_TWh'].shift(2)

df['co2_lag_1'] = df.groupby('country')['total_CO2_Mt'].shift(1)
df['co2_lag_2'] = df.groupby('country')['total_CO2_Mt'].shift(2)

# 5️⃣ Rolling mean features (3-year)
df['energy_roll_3'] = df.groupby('country')['primary_energy_TWh'].transform(lambda x: x.rolling(3).mean())
df['co2_roll_3']    = df.groupby('country')['total_CO2_Mt'].transform(lambda x: x.rolling(3).mean())

# 6️⃣ Year-over-year growth rates
df['yoy_energy_growth'] = df.groupby('country')['primary_energy_TWh'].pct_change()
df['yoy_co2_growth']    = df.groupby('country')['total_CO2_Mt'].pct_change()

# 7️⃣ Categorical features (if available, else merge from metadata)
# Example: df['region'], df['income_group']
# df['region'] = df['region'].astype('category')
# df['income_group'] = df['income_group'].astype('category')

# 8️⃣ Handle missing values
lag_roll_cols = ['energy_lag_1','energy_lag_2','energy_roll_3','yoy_energy_growth',
                 'co2_lag_1','co2_lag_2','co2_roll_3','yoy_co2_growth']

# Flag NAs
for col in lag_roll_cols:
    df[f'{col}_na'] = df[col].isna().astype(int)

# Fill NAs with 0 (optional)
df[lag_roll_cols] = df[lag_roll_cols].fillna(0)

# 9️⃣ Save processed feature table
df.to_csv("data/processed/features_table.csv", index=False)

print("✅ Feature table created: data/processed/features_table.csv")
