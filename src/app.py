# src/app.py
"""
Top-level runner for Phase 5 modeling.

Run as: python -m src.app
It will:
 - run the energy pipeline (Prophet + panel LGBM)
 - run the CO2 pipeline (baseline + residual LGBM)
Outputs are saved under models/ and reports/
"""
import os
from modeling.energy_forecast import run_full_energy_pipeline
from modeling.co2_link_model import run_co2_pipeline

DEFAULT_JOINED = "data/processed/joined_energy_co2.csv"

def main():
    print("=== Phase 5: Modeling runner ===")
    if not os.path.exists(DEFAULT_JOINED):
        raise FileNotFoundError(f"Expected processed joined file at {DEFAULT_JOINED}. Please create it first.")

    print("-> Running energy forecasting pipeline...")
    prop_models, panel_model = run_full_energy_pipeline(input_csv=DEFAULT_JOINED)

    print("-> Running CO2 linkage pipeline...")
    co2_model, co2_metrics = run_co2_pipeline(input_csv=DEFAULT_JOINED)

    print("=== Done. Models saved to models/ and reports/ ===")

if __name__ == "__main__":
    main()
