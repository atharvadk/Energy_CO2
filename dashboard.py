import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
import os
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# 0️⃣  Gemini API Setup
# ==========================================================
# Ensure your key is set:
#   export GOOGLE_API_KEY="AIzaSyD0RPW26ACCcTFm362hIrQQg-XdsuJI_Iw"   (Linux/Mac)
#   setx GOOGLE_API_KEY "your_key"     (Windows)
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.warning("⚠️ GOOGLE_API_KEY not found. Please set it before running.")
else:
    genai.configure(api_key=api_key)

# ==========================================================
# 1️⃣ Load Datasets
# ==========================================================
@st.cache_data
def load_data():
    prophet_df = pd.read_csv("clean_energy_co2_prophet.csv")
    lightgbm_df = pd.read_csv("clean_energy_co2_lightgbm.csv")
    hybrid_df = pd.read_csv("hybrid_forecast.csv")
    return prophet_df, lightgbm_df, hybrid_df

prophet_df, lightgbm_df, hybrid_df = load_data()

# ==========================================================
# 2️⃣ Dashboard Title & Sidebar
# ==========================================================
st.set_page_config(page_title="🌍 Energy & CO₂ Insights Dashboard", layout="wide")
st.title("🌍 Global Energy & CO₂ Insights Dashboard")

st.markdown(
    """
    This interactive dashboard visualizes **historical** and **forecasted** trends in
    energy consumption and CO₂ emissions for top countries, using:
    - **Prophet** for long-term trend modeling  
    - **LightGBM** for nonlinear relationships  
    - **Gemini AI** for intelligent, contextual recommendations  
    """
)

country_list = sorted(list(set(hybrid_df["country"])))
selected_country = st.sidebar.selectbox("Select a Country", country_list)
st.sidebar.info("📊 Uses Prophet (trend) + LightGBM (nonlinear) + Gemini (AI recommendations)")

# ==========================================================
# 3️⃣ Filter Data by Country
# ==========================================================
hist_prophet = prophet_df[prophet_df["country"] == selected_country].copy()
hist_lightgbm = lightgbm_df[lightgbm_df["country"] == selected_country].copy()
future_hybrid = hybrid_df[hybrid_df["country"] == selected_country].copy()

# ==========================================================
# 4️⃣ Historical Trends (CO₂ + Energy)
# ==========================================================


st.subheader(f"📈 Historical Energy & CO₂ Trends – {selected_country}")

col1, col2 = st.columns(2)

# --- Historical CO₂ ---
with col1:
    co2_hist = hist_prophet.copy()
    co2_hist["ds"] = pd.to_datetime(co2_hist["ds"], errors="coerce")
    fig_co2 = px.line(
        co2_hist,
        x="ds",
        y="y",
        title=f"{selected_country} – Historical CO₂ Emissions (Prophet Source)",
        labels={"ds": "Year", "y": "CO₂ Emissions (tons)"},
    )

    if "co2" in hist_lightgbm.columns:
        fig_co2.add_scatter(
            x=hist_lightgbm["year"],
            y=hist_lightgbm["co2"],
            mode="lines",
            name="CO₂ (LightGBM Source)",
            line=dict(dash="dash"),
        )

    fig_co2.update_layout(template="plotly_white", showlegend=True)
    st.plotly_chart(fig_co2, use_container_width=True)

# --- Historical Energy ---
with col2:
    # Handle possible column naming variations
    possible_energy_cols = [
        "primary_energy_consumption_energy",
        "primary_energy_consumption"
    ]
    energy_col = next((col for col in possible_energy_cols if col in hist_lightgbm.columns), None)

    if energy_col:
        energy_hist = hist_lightgbm.copy()
        fig_energy = px.line(
            energy_hist,
            x="year",
            y=energy_col,
            title=f"{selected_country} – Historical Energy Consumption",
            labels={"year": "Year", energy_col: "Energy Consumption (TWh)"},
        )
        fig_energy.update_traces(line=dict(color="#ff9900", width=2))
        fig_energy.update_layout(
            template="plotly_white",
            showlegend=False,
            yaxis_title="Energy (TWh)",
            xaxis_title="Year",
        )
        st.plotly_chart(fig_energy, use_container_width=True)
    else:
        st.warning("⚠️ Energy data not available for this country in the loaded dataset.")



# ==========================================================
# 5️⃣ Forecast Visualization (CO₂ + Energy)
# ==========================================================
st.subheader(f"🔮 Forecasted CO₂ & Energy – {selected_country}")

# --- CO₂ Forecast ---
fig_co2_forecast = go.Figure()
fig_co2_forecast.add_trace(go.Scatter(
    x=future_hybrid["year"], y=future_hybrid["yhat"],
    mode="lines", name="Prophet CO₂ Trend", line=dict(dash="dot")
))
fig_co2_forecast.add_trace(go.Scatter(
    x=future_hybrid["year"], y=future_hybrid["lightgbm_pred"],
    mode="lines", name="LightGBM CO₂ Prediction", line=dict(dash="dash")
))
fig_co2_forecast.add_trace(go.Scatter(
    x=future_hybrid["year"], y=future_hybrid["hybrid_forecast"],
    mode="lines", name="Hybrid CO₂ Forecast", line=dict(width=3)
))
fig_co2_forecast.update_layout(
    title=f"{selected_country} – Forecasted CO₂ Emissions (Next 10 Years)",
    xaxis_title="Year",
    yaxis_title="CO₂ Emissions (tons)",
    template="plotly_white"
)
st.plotly_chart(fig_co2_forecast, use_container_width=True)

# --- Energy Forecast (if present) ---
if "energy_forecast" in future_hybrid.columns or "energy_yhat" in future_hybrid.columns:
    fig_energy_forecast = go.Figure()
    if "energy_yhat" in future_hybrid.columns:
        fig_energy_forecast.add_trace(go.Scatter(
            x=future_hybrid["year"], y=future_hybrid["energy_yhat"],
            mode="lines", name="Prophet Energy Trend", line=dict(dash="dot")
        ))
    if "energy_lightgbm_pred" in future_hybrid.columns:
        fig_energy_forecast.add_trace(go.Scatter(
            x=future_hybrid["year"], y=future_hybrid["energy_lightgbm_pred"],
            mode="lines", name="LightGBM Energy Prediction", line=dict(dash="dash")
        ))
    if "energy_forecast" in future_hybrid.columns:
        fig_energy_forecast.add_trace(go.Scatter(
            x=future_hybrid["year"], y=future_hybrid["energy_forecast"],
            mode="lines", name="Hybrid Energy Forecast", line=dict(width=3)
        ))

    fig_energy_forecast.update_layout(
        title=f"{selected_country} – Forecasted Energy Consumption (Next 10 Years)",
        xaxis_title="Year",
        yaxis_title="Energy Consumption (TWh)",
        template="plotly_white"
    )
    st.plotly_chart(fig_energy_forecast, use_container_width=True)
else:
    st.info("ℹ️ Energy forecast columns not found. Run preprocessing to include them.")


st.subheader(f"📊 Additional Insights & Correlations – {selected_country}")

# Filter the lightgbm dataframe for the selected country
df_insights = hist_lightgbm[hist_lightgbm["country"] == selected_country].copy()
if df_insights.empty:
    st.warning("⚠️ No data available for insights for this country.")
else:
    # Convert year column to numeric if needed
    df_insights["year"] = pd.to_numeric(df_insights["year"], errors="coerce")

    # Create columns for side-by-side layout
    c1, c2 = st.columns(2)

    # ---------------------- GDP vs Energy ---------------------- #
    if "gdp_energy" in df_insights.columns and "primary_energy_consumption_energy" in df_insights.columns:
        with c1:
            fig_gdp_energy = px.scatter(
                df_insights,
                x="gdp_energy",
                y="primary_energy_consumption_energy",
                color="year",
                title="💰 GDP vs ⚡ Energy Consumption",
                labels={"gdp_energy": "GDP", "primary_energy_consumption_energy": "Energy (TWh)"},
                color_continuous_scale="Viridis"
            )
            fig_gdp_energy.update_layout(template="plotly_white")
            st.plotly_chart(fig_gdp_energy, use_container_width=True)

    # ---------------------- Energy Intensity ---------------------- #
    if "energy_intensity" in df_insights.columns:
        with c2:
            fig_energy_intensity = px.line(
                df_insights,
                x="year",
                y="energy_intensity",
                title="📉 Energy Intensity Over Time",
                labels={"year": "Year", "energy_intensity": "Energy Intensity (TWh per GDP)"},
            )
            fig_energy_intensity.update_traces(line=dict(color="#ff7f0e", width=2))
            fig_energy_intensity.update_layout(template="plotly_white")
            st.plotly_chart(fig_energy_intensity, use_container_width=True)

    # ---------------------- CO₂ Intensity ---------------------- #
    if "co2_intensity" in df_insights.columns:
        fig_co2_intensity = px.line(
            df_insights,
            x="year",
            y="co2_intensity",
            title="🌫️ CO₂ Intensity Over Time",
            labels={"year": "Year", "co2_intensity": "CO₂ Intensity (tons per GDP)"},
        )
        fig_co2_intensity.update_traces(line=dict(color="#636EFA", width=2))
        fig_co2_intensity.update_layout(template="plotly_white")
        st.plotly_chart(fig_co2_intensity, use_container_width=True)

    # ---------------------- Renewable Share ---------------------- #
    if "renewables_share_energy" in df_insights.columns:
        fig_renewable = px.area(
            df_insights,
            x="year",
            y="renewables_share_energy",
            title="🌱 Renewable Energy Share Over Time",
            labels={"year": "Year", "renewables_share_energy": "Renewables Share (%)"},
            color_discrete_sequence=["#2ca02c"]
        )
        fig_renewable.update_layout(template="plotly_white")
        st.plotly_chart(fig_renewable, use_container_width=True)

    # ---------------------- Energy per Capita ---------------------- #
    if "energy_per_capita_energy" in df_insights.columns:
        fig_energy_capita = px.line(
            df_insights,
            x="year",
            y="energy_per_capita_energy",
            title="👥 Energy Consumption per Capita",
            labels={"year": "Year", "energy_per_capita_energy": "Energy per Person (TWh)"},
        )
        fig_energy_capita.update_traces(line=dict(color="#d62728", width=2))
        fig_energy_capita.update_layout(template="plotly_white")
        st.plotly_chart(fig_energy_capita, use_container_width=True)


# ==========================================================
# 6️⃣ Gemini AI: Suggestive Measures
# ==========================================================
st.subheader("💡 AI-Generated Strategies to Reduce Energy Use & CO₂ Emissions")

# --- Gemini model selector in sidebar ---
model_choice = st.sidebar.selectbox(
    "🤖 Choose Gemini Model",
    ["models/gemini-2.5-flash", "models/gemini-2.5-pro"],
    index=0,
    help="⚡ Flash = faster | 🧠 Pro = deeper reasoning"
)

if st.button("✨ Generate Gemini AI Recommendations"):
    try:
        # --- Gather latest data context ---
        recent = hist_lightgbm.sort_values("year").iloc[-1]
        last_year = int(recent["year"])
        co2_now = float(recent["co2"]) if "co2" in recent else 0.0

        # Estimate forecasted 2030 value (if exists)
        forecast_2030 = None
        if 2030 in list(future_hybrid["year"]):
            forecast_2030 = float(
                future_hybrid.loc[future_hybrid["year"] == 2030, "hybrid_forecast"].values[0]
            )

        # Handle missing values gracefully
        forecast_text = (
            f"{forecast_2030:.2e} tons" if forecast_2030 is not None else "Data not available"
        )

        # --- Build Gemini prompt ---
        prompt = f"""
        You are an expert energy and climate policy analyst.
        Based on the following data, write a structured policy note.

        Country: {selected_country}
        Current year: {last_year}
        Estimated CO₂ emissions: {co2_now:.2e} tons
        Forecasted CO₂ emissions by 2030: {forecast_text}

        Provide detailed and actionable strategies for this country to:
        1. Reduce overall energy consumption.
        2. Lower CO₂ emissions.
        3. Accelerate renewable energy adoption.
        4. Promote sustainable industrial, transport, and citizen practices.

        Present the output as:
        - **Industrial Measures**
        - **Transportation Measures**
        - **Energy Sector Policies**
        - **Government & Policy Reforms**
        - **Citizen & Lifestyle Changes**

        Each bullet should be short, realistic, regionally relevant, and feasible.
        """

        # --- Gemini API call ---
        model = genai.GenerativeModel(model_choice)
        response = model.generate_content(prompt)

        # --- Display results ---
        st.success(f"✅ AI Recommendations generated successfully using {model_choice}")
        with st.expander("📘 View Detailed AI Recommendations", expanded=True):
            st.markdown(response.text)

    except Exception as e:
        st.error(f"⚠️ Gemini API error: {e}")

st.caption("AI insights powered by Google Gemini 2.5 models.")


# ==========================================================
# 7️⃣ Summary
# ==========================================================
st.markdown("---")
st.markdown(
    f"""
    ✅ *{selected_country}* modeled using **Prophet for temporal trends** and **LightGBM for nonlinear effects**.  
    The **hybrid forecast** represents the projected CO₂ emissions for the next decade.  
    Use the Gemini-powered recommendations to explore region-specific sustainability actions.
    """
)
