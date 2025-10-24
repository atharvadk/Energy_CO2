# src/modeling/energy_forecast.py
"""
Per-country Prophet + panel LightGBM energy forecasting pipeline.

Expected input: data/processed/joined_energy_co2.csv or similar file with columns:
  - iso_code (string)
  - date (YYYY-MM-DD or YYYY-MM)
  - energy (float)  # target energy consumption/production in consistent unit
  - optional socio/economic or fuel columns used as features

Outputs (saved):
  - models/prophet/{iso_code}.joblib
  - models/lgbm_energy.joblib
  - reports/energy_forecast_metrics.csv
  - reports/forecast_plots/{iso_code}.png
"""
import os
import pandas as pd
import numpy as np
from prophet import Prophet
from lightgbm import LGBMRegressor
from joblib import dump, load
import matplotlib.pyplot as plt

from ..features import make_panel_features
from .utils import ensure_dirs, save_model, regression_metrics, save_metrics_df

DEFAULT_INPUT = "data/processed/joined_energy_co2.csv"

def train_prophet_per_country(df,
                              iso_col="iso_code",
                              date_col="date",
                              target_col="energy",
                              prophet_outdir="models/prophet",
                              min_obs=24):
    """
    Fit Prophet per country and save each model.
    Returns dict of fitted models for convenience.
    """
    ensure_dirs(prophet_outdir)
    models = {}
    for iso in df[iso_col].unique():
        sub = df[df[iso_col] == iso].sort_values(date_col)[[date_col, target_col]].dropna()
        if len(sub) < min_obs:
            # skip very short series (still could be trained but we skip by default)
            continue
        sub_ren = sub.rename(columns={date_col: "ds", target_col: "y"})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(sub_ren)
        path = os.path.join(prophet_outdir, f"{iso}.joblib")
        save_model(m, path)
        models[iso] = m
        print(f"Saved Prophet for {iso} -> {path}")
    return models

def prophet_forecast_for_country(model, periods=12, freq='MS'):
    future = model.make_future_dataframe(periods=periods, freq=freq)
    fcst = model.predict(future)
    return fcst[['ds','yhat','yhat_lower','yhat_upper']]

def train_panel_lgbm(panel_df,
                     target_col="energy",
                     iso_col="iso_code",
                     date_col="date",
                     out_path="models/lgbm_energy.joblib",
                     feature_exclude=None):
    """
    Train a LightGBM regressor on the panel. We assume panel_df already has features from make_panel_features.
    Splits by time: last 6 months held for validation.
    """
    feature_exclude = feature_exclude or [iso_col, date_col, target_col]
    X = panel_df.drop(columns=[c for c in feature_exclude if c in panel_df.columns], errors='ignore')
    y = panel_df[target_col].values
    # temporal split
    panel_df[date_col] = pd.to_datetime(panel_df[date_col])
    max_date = panel_df[date_col].max()
    val_cut = max_date - pd.DateOffset(months=6)
    train_mask = panel_df[date_col] <= val_cut
    if train_mask.sum() < 50:
        # fallback: 80/20 random split if not enough time points
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train = X.loc[train_mask]
        y_train = y[train_mask.values]
        X_val = X.loc[~train_mask]
        y_val = y[~train_mask.values]

    model = LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=100)
    save_model(model, out_path)
    preds = model.predict(X_val, num_iteration=model.best_iteration_)
    metrics = regression_metrics(y_val, preds)
    print(f"Panel LGBM metrics: {metrics}")
    return model, metrics

def run_full_energy_pipeline(input_csv=DEFAULT_INPUT,
                             prophet_min_obs=24,
                             panel_lags=(1,3,12),
                             panel_rolls=(3,12),
                             save_reports_dir="reports"):
    ensure_dirs("models/prophet", "models", save_reports_dir, os.path.join(save_reports_dir, "forecast_plots"))
    df = pd.read_csv(input_csv)
    # unify column names that we expect
    expected_target = None
    for candidate in ["energy", "energy_TWh", "energy_MWh", "energy_consumption"]:
        if candidate in df.columns:
            expected_target = candidate
            break
    if expected_target is None:
        raise ValueError("Cannot find an energy column in the input file. Expected one of: energy, energy_TWh, energy_MWh, energy_consumption")
    df = df.rename(columns={expected_target: "energy"})
    # Prophet per-country
    prop_models = train_prophet_per_country(df, target_col="energy", min_obs=prophet_min_obs)
    # Create panel features
    panel = make_panel_features(df, target_col="energy", lags=panel_lags, rolling_windows=panel_rolls)
    # optional: add prophet yhat as a feature (if model exists for the iso and date)
    # compute a simple per-country prophet baseline for dates present in panel and merge as feature
    prophet_feature_rows = []
    for iso, m in prop_models.items():
        # build forecast for the date range of panel for that iso
        iso_dates = panel.loc[panel['iso_code']==iso, 'date'].drop_duplicates()
        if iso_dates.empty:
            continue
        first = pd.to_datetime(iso_dates.min())
        last = pd.to_datetime(iso_dates.max())
        periods = int(((last - first).days / 30) + 6)  # a heuristic
        fcst = prophet_forecast_for_country(m, periods=periods, freq='MS')
        # clip to date range and keep yhat
        fcst['ds'] = pd.to_datetime(fcst['ds'])
        mask = (fcst['ds'] >= first - pd.DateOffset(months=1)) & (fcst['ds'] <= last + pd.DateOffset(months=1))
        fcst_use = fcst.loc[mask, ['ds','yhat']].rename(columns={'ds':'date','yhat':'prophet_yhat'})
        fcst_use['iso_code'] = iso
        prophet_feature_rows.append(fcst_use)
    if prophet_feature_rows:
        prophet_feats = pd.concat(prophet_feature_rows, ignore_index=True)
        # unify date formats
        prophet_feats['date'] = pd.to_datetime(prophet_feats['date'])
        panel['date'] = pd.to_datetime(panel['date'])
        panel = panel.merge(prophet_feats, on=['iso_code','date'], how='left')

    # drop rows with NaN in key features; LightGBM can handle some NaNs but we prefer to drop rows missing lags
    model, metrics = train_panel_lgbm(panel, target_col="energy")
    # save metrics and a quick sample plot for US,CN,IN (if present)
    metrics_list = [{"model":"lgbm_energy","mae":metrics['mae'],"rmse":metrics['rmse'],"mape":metrics['mape'],"r2":metrics['r2']}]
    save_metrics_df(metrics_list, os.path.join(save_reports_dir, "energy_forecast_metrics.csv"))

    # Plot sample countries vs predictions
    for iso in ["US","CN","IN"]:
        try:
            plot_sample_country(panel, model, iso, outdir=os.path.join(save_reports_dir,"forecast_plots"))
        except Exception as e:
            print(f"Could not plot {iso}: {e}")

    return prop_models, model

def plot_sample_country(panel_df, model, iso, date_col="date", target_col="energy", outdir="reports/forecast_plots"):
    ensure_dirs(outdir)
    p = panel_df[panel_df['iso_code']==iso].sort_values(date_col)
    if p.empty:
        raise ValueError(f"No data for {iso}")
    # prepare X for most recent range
    feat_cols = [c for c in p.columns if c not in ['iso_code','date',target_col]]
    X = p[feat_cols]
    preds = model.predict(X)
    plt.figure(figsize=(10,4))
    plt.plot(p[date_col], p[target_col], label="actual", marker='o')
    plt.plot(p[date_col], preds, label="panel_lgbm_pred", marker='x')
    if 'prophet_yhat' in p.columns:
        plt.plot(p[date_col], p['prophet_yhat'], label="prophet_yhat", linestyle='--')
    plt.title(f"{iso} energy: actual vs predictions")
    plt.legend()
    out = os.path.join(outdir, f"{iso}_forecast.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved forecast plot: {out}")
