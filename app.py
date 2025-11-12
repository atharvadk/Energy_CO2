import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
import datetime
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
#   export GOOGLE_API_KEY="your_key"   (Linux/Mac)
#   setx GOOGLE_API_KEY "your_key"     (Windows)
# ------------------ SETUP ------------------ #
st.set_page_config(page_title="🌍 Energy & CO₂ Forecasting", layout="wide")
st.title("🌍 Energy & CO₂ Forecasting")

# ------------------ LOAD DATA ------------------ #
@st.cache_data
def load_data():
    co2 = pd.read_csv("data/processed/detailed_co2.csv")
    energy = pd.read_csv("data/processed/detailed_energy.csv")
    features = pd.read_csv("data/processed/features_table.csv")
    joined = pd.read_csv("data/processed/joined_energy_co2.csv")
    return co2, energy, features, joined

co2_df, energy_df, features_df, joined_df = load_data()

# ------------------ SIDEBAR ------------------ #
country = st.sidebar.selectbox("Select Country", joined_df["country"].unique())
metric = st.sidebar.selectbox("Select Metric", ["primary_energy_TWh", "total_CO2_Mt"])
horizon = st.sidebar.slider("Forecast Horizon (Years)", 5, 20, 10)

# ------------------ HISTORICAL DATA ------------------ #
subset = joined_df[joined_df["country"] == country].copy()
subset = subset[["year", metric]].dropna()

prophet_df = subset.rename(columns={"year": "ds", metric: "y"})
prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], format="%Y")

st.subheader(f"📊 Historical Data for {country} - {metric}")
if subset.empty:
    st.error("No data available for this selection.")
else:
    st.line_chart(prophet_df.set_index("ds")["y"])

# ------------------ PROPHET FORECAST ------------------ #
st.subheader("🔮 Prophet Forecast")

if prophet_df.shape[0] < 2:
    st.warning("⚠️ Not enough rows for Prophet — showing random forecast instead.")
    future_years = pd.date_range(datetime.date.today(), periods=horizon, freq="Y")
    fake_forecast = pd.DataFrame({
        "ds": future_years,
        "yhat": np.random.randint(500, 3000, len(future_years))
    })
    fig1 = px.line(fake_forecast, x="ds", y="yhat", title="Demo Forecast (Random Data)")
    st.plotly_chart(fig1, use_container_width=True)
else:
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=horizon, freq="Y")
    forecast = model.predict(future)

    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1, use_container_width=True)

    st.write("Forecast sample:")
    st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon))



# ------------------ LIGHTGBM PANEL MODEL ------------------ #
st.subheader("🤖 LightGBM Panel Model")

panel = features_df.copy()
panel = panel.sort_values(["country", "year"])

# Choose feature set
if "energy_lag_1" in panel.columns:
    X = panel[["gdp", "population", "energy_lag_1"]]
else:
    st.warning("Lag features not found — using GDP & Population only.")
    X = panel[["gdp", "population"]]

y = panel["primary_energy_TWh"]

# Drop rows with NaN
mask = ~(X.isna().any(axis=1) | y.isna())
X, y = X.loc[mask], y.loc[mask]

if len(X) > 10:
    train_size = int(0.8 * len(X))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    dtrain = lgb.Dataset(X_train, label=y_train)
    params = {"objective": "regression", "verbosity": -1}
    model_lgb = lgb.train(params, dtrain, num_boost_round=50)

    preds = model_lgb.predict(X_test)

    # Ensure no NaNs in preds
    preds = np.nan_to_num(preds)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

    st.write(f"**MAE:** {mae:.2f}, **RMSE:** {rmse:.2f}, **MAPE:** {mape:.2f}%")

    compare_df = pd.DataFrame({"Actual": y_test.values, "Predicted": preds})
    st.line_chart(compare_df)
else:
    st.info("Not enough rows to train LightGBM. Showing random demo predictions.")
    fake_preds = np.random.randint(500, 3000, 20)
    st.line_chart(pd.DataFrame({"Demo Predictions": fake_preds}))
