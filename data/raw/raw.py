import pandas as pd

# URLs
energy_url = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
co2_url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"

# File paths
energy_file = "data/raw/owid-energy-data.csv"
co2_file = "data/raw/owid-co2-data.csv"

# Download datasets
energy_df = pd.read_csv(energy_url)
energy_df.to_csv(energy_file, index=False)

co2_df = pd.read_csv(co2_url)
co2_df.to_csv(co2_file, index=False)

print("✅ Datasets downloaded and saved successfully.")
