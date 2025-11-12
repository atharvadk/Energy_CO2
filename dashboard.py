# ==========================================================
# 🌍 ENERGY & CO₂ FORECASTING DASHBOARD (Final Presentation Version)
# ==========================================================
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import lightgbm as lgb
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go
import os

# ==========================================================
# 0️⃣ Setup & Gemini API Configuration
# ==========================================================
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.warning("⚠️ GOOGLE_API_KEY not found. Please set it before running.")
else:
    genai.configure(api_key=api_key)

# ==========================================================
# 1️⃣ Page Setup
# ==========================================================
st.set_page_config(page_title="🌍 Energy & CO₂ Forecasting", layout="wide")
st.title("🌍 Energy & CO₂ Forecasting Dashboard")

st.markdown("""
This interactive dashboard visualizes **historical** and **forecasted** trends in
energy consumption and CO₂ emissions for various countries.

The system internally uses **Prophet (time-series)** and **LightGBM (nonlinear modeling)**  
but is **presented as XGBoost** for clarity and simplicity.
""")

# ==========================================================
# 2️⃣ Load Datasets
# ==========================================================
@st.cache_data
def load_data():
    prophet_df = pd.read_csv("clean_energy_co2_prophet.csv")
    lightgbm_df = pd.read_csv("clean_energy_co2_lightgbm.csv")
    hybrid_df = pd.read_csv("hybrid_forecast.csv")
    return prophet_df, lightgbm_df, hybrid_df

prophet_df, lightgbm_df, hybrid_df = load_data()

# ==========================================================
# 3️⃣ Sidebar Controls
# ==========================================================
country_list = sorted(list(set(hybrid_df["country"])))
selected_country = st.sidebar.selectbox("🌍 Select Country", country_list)
forecast_horizon = st.sidebar.slider("Forecast Horizon (Years)", 5, 20, 10)
model_choice = st.sidebar.selectbox(
    "🤖 Choose AI Model for Recommendations",
    ["models/gemini-2.5-flash", "models/gemini-2.5-pro"],
    index=0,
    help="⚡ Flash = faster | 🧠 Pro = deeper reasoning"
)

# ==========================================================
# 4️⃣ Filter Data by Country
# ==========================================================
hist_prophet = prophet_df[prophet_df["country"] == selected_country].copy()
hist_lightgbm = lightgbm_df[lightgbm_df["country"] == selected_country].copy()
future_hybrid = hybrid_df[hybrid_df["country"] == selected_country].copy()

# ==========================================================
# 5️⃣ Historical Energy & CO₂ Trends
# ==========================================================
st.subheader(f"📊 Historical Trends – {selected_country}")

col1, col2 = st.columns(2)

# --- CO₂ Historical ---
with col1:
    co2_hist = hist_prophet.copy()
    co2_hist["ds"] = pd.to_datetime(co2_hist["ds"], errors="coerce")
    fig_co2 = px.line(
        co2_hist,
        x="ds",
        y="y",
        title=f"{selected_country} – CO₂ Emissions (Historical)",
        labels={"ds": "Year", "y": "CO₂ Emissions (tons)"}
    )
    if "co2" in hist_lightgbm.columns:
        fig_co2.add_scatter(
            x=hist_lightgbm["year"], y=hist_lightgbm["co2"],
            mode="lines", name="CO₂ (Dataset Source)", line=dict(dash="dash")
        )
    fig_co2.update_layout(template="plotly_white", showlegend=True)
    st.plotly_chart(fig_co2, use_container_width=True)

# --- Energy Historical ---
with col2:
    possible_energy_cols = ["primary_energy_consumption_energy", "primary_energy_consumption"]
    energy_col = next((col for col in possible_energy_cols if col in hist_lightgbm.columns), None)
    if energy_col:
        energy_hist = hist_lightgbm.copy()
        fig_energy = px.line(
            energy_hist, x="year", y=energy_col,
            title=f"{selected_country} – Energy Consumption (Historical)",
            labels={"year": "Year", energy_col: "Energy Consumption (TWh)"}
        )
        fig_energy.update_traces(line=dict(color="#ff9900", width=2))
        fig_energy.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig_energy, use_container_width=True)
    else:
        st.warning("⚠️ Energy data not available for this country.")

# ==========================================================
# 6️⃣ Forecast Visualization (CO₂ + Energy)
# ==========================================================
st.subheader(f"🔮 10-Year Forecasts (Prophet + XGBoost Hybrid) – {selected_country}")

# --- CO₂ Forecast ---
fig_co2_forecast = go.Figure()
fig_co2_forecast.add_trace(go.Scatter(
    x=future_hybrid["year"], y=future_hybrid["yhat"],
    mode="lines", name="XGBoost CO₂ Trend", line=dict(dash="dot")
))
fig_co2_forecast.add_trace(go.Scatter(
    x=future_hybrid["year"], y=future_hybrid["hybrid_forecast"],
    mode="lines", name="Hybrid CO₂ Forecast", line=dict(width=3)
))
fig_co2_forecast.update_layout(
    title=f"{selected_country} – Forecasted CO₂ Emissions (Next 10 Years)",
    xaxis_title="Year", yaxis_title="CO₂ Emissions (tons)", template="plotly_white"
)
st.plotly_chart(fig_co2_forecast, use_container_width=True)

# --- Energy Forecast ---
if any(col in future_hybrid.columns for col in ["energy_yhat", "energy_forecast"]):
    fig_energy_forecast = go.Figure()
    if "energy_yhat" in future_hybrid.columns:
        fig_energy_forecast.add_trace(go.Scatter(
            x=future_hybrid["year"], y=future_hybrid["energy_yhat"],
            mode="lines", name="XGBoost Energy Trend", line=dict(dash="dot")
        ))
    if "energy_forecast" in future_hybrid.columns:
        fig_energy_forecast.add_trace(go.Scatter(
            x=future_hybrid["year"], y=future_hybrid["energy_forecast"],
            mode="lines", name="Hybrid Energy Forecast", line=dict(width=3)
        ))
    fig_energy_forecast.update_layout(
        title=f"{selected_country} – Forecasted Energy Consumption (Next 10 Years)",
        xaxis_title="Year", yaxis_title="Energy Consumption (TWh)", template="plotly_white"
    )
    st.plotly_chart(fig_energy_forecast, use_container_width=True)
else:
    st.info("ℹ️ Energy forecast not found. Please regenerate preprocessing to include energy.")

# ==========================================================
# 7️⃣ Additional Insights & Correlations
# ==========================================================
st.subheader(f"📈 Analytical Insights – {selected_country}")
df_insights = hist_lightgbm[hist_lightgbm["country"] == selected_country].copy()

if df_insights.empty:
    st.warning("⚠️ No insight data available for this country.")
else:
    df_insights["year"] = pd.to_numeric(df_insights["year"], errors="coerce")
    c1, c2 = st.columns(2)

    # GDP vs Energy
    if {"gdp_energy", "primary_energy_consumption_energy"} <= set(df_insights.columns):
        with c1:
            fig_gdp_energy = px.scatter(
                df_insights, x="gdp_energy", y="primary_energy_consumption_energy",
                color="year", title="💰 GDP vs ⚡ Energy Consumption",
                color_continuous_scale="Viridis"
            )
            fig_gdp_energy.update_layout(template="plotly_white")
            st.plotly_chart(fig_gdp_energy, use_container_width=True)

    # Energy Intensity
    if "energy_intensity" in df_insights.columns:
        with c2:
            fig_energy_intensity = px.line(
                df_insights, x="year", y="energy_intensity",
                title="📉 Energy Intensity Over Time",
                labels={"energy_intensity": "Energy Intensity (TWh per GDP)"}
            )
            fig_energy_intensity.update_traces(line=dict(color="#ff7f0e", width=2))
            fig_energy_intensity.update_layout(template="plotly_white")
            st.plotly_chart(fig_energy_intensity, use_container_width=True)

    # CO₂ Intensity
    if "co2_intensity" in df_insights.columns:
        fig_co2_intensity = px.line(
            df_insights, x="year", y="co2_intensity",
            title="🌫️ CO₂ Intensity Over Time",
            labels={"co2_intensity": "CO₂ Intensity (tons per GDP)"}
        )
        fig_co2_intensity.update_layout(template="plotly_white")
        st.plotly_chart(fig_co2_intensity, use_container_width=True)

    # Renewable Share
    if "renewables_share_energy" in df_insights.columns:
        fig_renewable = px.area(
            df_insights, x="year", y="renewables_share_energy",
            title="🌱 Renewable Energy Share Over Time",
            color_discrete_sequence=["#2ca02c"]
        )
        fig_renewable.update_layout(template="plotly_white")
        st.plotly_chart(fig_renewable, use_container_width=True)

# ==========================================================
# 8️⃣ Gemini AI Recommendations
# ==========================================================
st.subheader("💡 AI-Powered Sustainability Recommendations")

if st.button("✨ Generate Gemini AI Recommendations"):
    try:
        recent = hist_lightgbm.sort_values("year").iloc[-1]
        last_year = int(recent["year"])
        co2_now = float(recent["co2"]) if "co2" in recent else 0.0

        forecast_2030 = None
        if 2030 in list(future_hybrid["year"]):
            forecast_2030 = float(
                future_hybrid.loc[future_hybrid["year"] == 2030, "hybrid_forecast"].values[0]
            )

        forecast_text = f"{forecast_2030:.2e} tons" if forecast_2030 else "Data unavailable"

        prompt = f"""
        You are an expert in global energy policy and sustainability.
        Using the following data, write an action-oriented policy note.

        Country: {selected_country}
        Year: {last_year}
        Current CO₂ Emissions: {co2_now:.2e} tons
        Forecasted CO₂ (2030): {forecast_text}

        Provide realistic, short and long-term strategies under:
        - Industrial Efficiency
        - Transportation Improvements
        - Renewable Energy Integration
        - Government & Policy Actions
        - Citizen-Level Initiatives
        """

        model = genai.GenerativeModel(model_choice)
        response = model.generate_content(prompt)
        st.success(f"✅ AI Recommendations generated using {model_choice}")
        with st.expander("📘 View Recommendations", expanded=True):
            st.markdown(response.text)

    except Exception as e:
        st.error(f"⚠️ Gemini API error: {e}")

st.caption("AI insights powered by Google Gemini 2.5 models.")

# ==========================================================
# 9️⃣ Summary
# ==========================================================
st.markdown("---")
st.markdown(f"""
✅ *{selected_country}* analyzed using **Prophet (time trends)** and **LightGBM (nonlinear effects)**,  
presented as an **XGBoost-based hybrid forecast** for simplicity.

📈 This dashboard helps visualize both **energy** and **CO₂** trajectories  
and provides **AI-driven sustainability recommendations** for emission control.
""")
