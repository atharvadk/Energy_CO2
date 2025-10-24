# src/features.py
"""
Feature engineering helpers for panel energy forecasting.

Produces lag features, rolling means, seasonal sines/cosines, and
basic fuel-share features if fuel columns exist.
"""
import pandas as pd
import numpy as np

def make_panel_features(df,
                        date_col="date",
                        iso_col="iso_code",
                        target_col="energy",
                        lags=(1,3,12),
                        rolling_windows=(3,12)):
    """
    Input: DataFrame with columns [iso_code, date, energy, ...optional fuel cols / socioecon...]
    Output: DataFrame enriched with lag/roll/seasonal features (sorted)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([iso_col, date_col]).reset_index(drop=True)

    # base time features
    df['month'] = df[date_col].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    df['t'] = df.groupby(iso_col).cumcount() + 1

    # lags
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby(iso_col)[target_col].shift(lag)

    # rolling means (use shifted series)
    for w in rolling_windows:
        df[f'rollmean_{w}'] = (df.groupby(iso_col)[target_col]
                                  .shift(1)
                                  .rolling(window=w, min_periods=1)
                                  .mean()
                                  .reset_index(level=0, drop=True))

    # fuel share features if fuel columns exist (common names)
    fuel_cols = [c for c in df.columns if c.lower().endswith(('_twh','_twh'))]  # heuristic
    # also common explicit names:
    for name in ['coal_TWh','gas_TWh','oil_TWh','electricity_TWh','coal','gas','oil','electricity']:
        if name in df.columns and name not in fuel_cols:
            fuel_cols.append(name)
    # compute shares if at least two fuel cols present
    fuel_cols = list(dict.fromkeys(fuel_cols))  # dedupe
    if len(fuel_cols) >= 1:
        df['fuel_total'] = df[fuel_cols].sum(axis=1).replace({0: np.nan})
        for c in fuel_cols:
            safe_name = c.replace(' ', '_').replace('-', '_')
            df[f'{safe_name}_share'] = df[c] / df['fuel_total']

    # drop helper column if created
    if 'fuel_total' in df.columns:
        df.drop(columns=['fuel_total'], inplace=True)

    # drop rows with NaN in the required lag features (so LGBM training won't choke)
    required_lags = [f'lag_{l}' for l in lags]
    df = df.dropna(subset=required_lags, how='any').reset_index(drop=True)

    return df
