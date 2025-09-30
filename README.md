# Energy & CO₂ Prediction Project

This project analyzes and predicts per-country energy consumption and CO₂ emissions using OWID datasets.  

## Structure
- `data/raw/` — original datasets (not tracked in git)
- `data/processed/` — cleaned datasets
- `notebooks/` — Jupyter notebooks for EDA and prototyping
- `models/` — trained ML models
- `app/` — deployment code (dashboard/API)
- `reports/` — generated reports & visualizations
- `src/` — reusable source code (data processing, ML pipeline, utils)

## Setup
```bash
git clone https://github.com/atharvadk/Energy_CO2.git
cd energy_co2_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
