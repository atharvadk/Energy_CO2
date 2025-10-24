# src/modeling/co2_link_model.py
"""
CO2 linkage modeling:
 - compute physics baseline using emission factors (user should verify factors)
 - fit a LightGBM on residual = observed_co2 - baseline_co2
 - save model to models/lgbm_co2.joblib
 - evaluate baseline vs ML-enhanced prediction
"""
import os
import pandas as pd
from lightgbm import LGBMRegressor
from joblib import dump
from .utils import ensure_dirs, save_model, regression_metrics, save_metrics_df
import numpy as np

# default emission factors (placeholder) in MtCO2 per TWh or appropriate units.
# **REPLACE** with authoritative factors (IPCC / IEA) for production use.
DEFAULT_EMISSION_FACTORS = {
    # column_name_in_df: factor (co2_units per energy_unit)
    # Example assumes energy in TWh and factor in MtCO2/TWh (this is illustrative)
    "coal_TWh": 2.4,        # MtCO2 per TWh (placeholder)
    "oil_TWh": 2.1,
    "gas_TWh": 2.0,
    "electricity_TWh": 0.5  # average grid carbon intensity (placeholder)
}

def compute_baseline_co2(df, emission_factors=None):
    """
    Sum over fuel columns * emission factor. If the exact fuel columns are missing, tries a few heuristics.
    """
    emission_factors = emission_factors or DEFAULT_EMISSION_FACTORS
    df = df.copy()
    baseline = pd.Series(0.0, index=df.index, dtype=float)
    used = []
    for col, factor in emission_factors.items():
        if col in df.columns:
            baseline += df[col].fillna(0.0) * float(factor)
            used.append(col)
    # if no columns matched, attempt heuristic names
    if not used:
        for name in ['coal','gas','oil','electricity']:
            if name in df.columns:
                baseline += df[name].fillna(0.0) * emission_factors.get(f"{name}_TWh", 1.0)
                used.append(name)
    df['baseline_co2'] = baseline
    return df

def train_co2_residual_model(df,
                             features,
                             target_col="observed_co2",
                             out_path="models/lgbm_co2.joblib"):
    """
    df must include observed_co2 and the fuel columns used for baseline.
    Trains an LGBM to predict residual = observed_co2 - baseline_co2.
    Returns (model, metrics_dict)
    """
    ensure_dirs(os.path.dirname(out_path) or "models")
    df = compute_baseline_co2(df)
    if 'baseline_co2' not in df.columns:
        raise ValueError("Could not compute baseline_co2; check fuel column names.")

    df = df.copy()
    df['residual'] = df[target_col] - df['baseline_co2']

    # drop rows with missing features or target
    df_train = df.dropna(subset=features + ['residual'])
    if df_train.empty:
        raise ValueError("Training data for CO2 residual model is empty after dropping NAs. Check features.")

    # temporal split: last 6 months as validation if date column exists
    if 'date' in df_train.columns:
        df_train['date'] = pd.to_datetime(df_train['date'])
        max_date = df_train['date'].max()
        val_cut = max_date - pd.DateOffset(months=6)
        train_mask = df_train['date'] <= val_cut
        if train_mask.sum() < 20:
            train_mask = None
    else:
        train_mask = None

    if train_mask is None:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(df_train[features], df_train['residual'], test_size=0.2, random_state=42)
    else:
        X_train = df_train.loc[train_mask, features]
        y_train = df_train.loc[train_mask, 'residual']
        X_val = df_train.loc[~train_mask, features]
        y_val = df_train.loc[~train_mask, 'residual']

    model = LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=31, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=100)
    save_model(model, out_path)

    # validation predictions
    pred_resid = model.predict(X_val, num_iteration=model.best_iteration_)
    pred_total = df_train.loc[X_val.index, 'baseline_co2'] + pred_resid
    y_true_total = df_train.loc[X_val.index, target_col]

    baseline_metrics = regression_metrics(y_true_total, df_train.loc[X_val.index, 'baseline_co2'])
    ml_metrics = regression_metrics(y_true_total, pred_total)

    metrics = {
        "baseline": baseline_metrics,
        "ml_enhanced": ml_metrics
    }
    return model, metrics

def run_co2_pipeline(input_csv="data/processed/joined_energy_co2.csv",
                     features=None,
                     out_model="models/lgbm_co2.joblib",
                     reports_dir="reports"):
    ensure_dirs(reports_dir, os.path.dirname(out_model) or "models")
    df = pd.read_csv(input_csv)
    # guess common observed co2 column name
    for c in ['observed_co2','co2','co2_mt','co2_mt_per_month']:
        if c in df.columns:
            df = df.rename(columns={c:'observed_co2'})
            break
    if 'observed_co2' not in df.columns:
        raise ValueError("Cannot find observed CO2 column in input CSV. Expected one of observed_co2, co2, co2_mt")

    # default features if not supplied: fuel shares + socioeconomics if present
    if features is None:
        candidate_shares = [col for col in df.columns if col.endswith('_share')]
        cand_base = ['gdp_per_capita','population','month_sin','month_cos']
        features = [c for c in candidate_shares + cand_base if c in df.columns]
        if not features:
            # fallback: use fuel columns directly + gdp/pop if present
            fallback = [c for c in ['coal_TWh','gas_TWh','oil_TWh','electricity_TWh','gdp_per_capita','population'] if c in df.columns]
            if not fallback:
                raise ValueError("Could not auto-select CO2 features; pass `features` explicitly.")
            features = fallback

    model, metrics = train_co2_residual_model(df, features=features, target_col='observed_co2', out_path=out_model)

    # write metrics out
    metrics_path = os.path.join(reports_dir, "co2_model_metrics.json")
    import json
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved CO2 model metrics to {metrics_path}")
    return model, metrics
