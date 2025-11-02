# 🌍 Global Energy and CO₂ Analysis Dashboard

An interactive data analytics and forecasting dashboard built using statistical inference techniques to analyze and visualize global energy consumption, production, and CO₂ emissions trends.  
This project combines Our World in Data (OWID) energy and emissions datasets to explore relationships between economic growth, energy use, and environmental impact across countries and years.

---

## Overview

This repository provides an interactive environment to:

- Visualize global and country-level energy and CO₂ trends
- Analyze relationships between GDP, energy intensity, and renewable energy share
- Forecast energy demand and CO₂ emissions using statistical time-series models
- Perform comparative analysis across countries and metrics
- Support policy insights and sustainable development research

All visualizations and analyses are rendered using Plotly (Dash) and Streamlit for an intuitive interactive experience.

---

## Features

### Data Integration
- Merges OWID Energy and OWID CO₂ datasets
- Supports filtering by country, region, and time period
- Cleans and harmonizes units and missing values

### Interactive Visualizations
- Dynamic Plotly charts for energy mix, renewables share, and CO₂ emissions
- Comparative and correlation plots (e.g., GDP vs Energy, Energy Intensity)
- Forecast visualizations from time-series models

### Statistical Modeling
- Forecasting with ARIMA and Prophet (extendable)
- Energy intensity and emission factor estimation
- Correlation and regression analyses for hypothesis validation

### Exploratory Dashboard
- Multi-page interface (Dash + Streamlit)
- Country and metric selectors with auto-updating insights

---

## Methodology

Key analytical methods used:
- Time-series decomposition and trend analysis
- Correlation and regression analysis
- Forecasting with ARIMA and Prophet
- Ratio- and index-based comparative metrics
- Visualization-driven hypothesis testing

---

## Datasets

Source: Our World in Data (https://ourworldindata.org/)

| Dataset | File | Description |
|---------|------|-------------|
| OWID Energy Data | `owid-energy-data.csv` | Energy consumption, production, renewables, and sources |
| OWID CO₂ Data | `owid-co2-data.csv` | Historical CO₂ emissions, per-capita metrics, emissions by fuel |

Both datasets are preprocessed and merged for analytical consistency.

---

## Tech Stack

| Category | Tools / Libraries |
|----------|-------------------|
| Programming | Python |
| Visualization | Plotly, Plotly Express |
| Dashboard | Dash, Streamlit |
| Data Handling | pandas, NumPy |
| Modeling | statsmodels, Prophet, scikit-learn |
| Environment | Jupyter, VS Code |

---

## Installation & Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/global-energy-co2-dashboard.git
cd global-energy-co2-dashboard
```

2. (Recommended) Create and activate a virtual environment
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```



4. Run the Streamlit app
```bash
streamlit run dashboard.py
# then open: http://localhost:8501/
```

Basic troubleshooting:
- Ensure Python 3.8+ is installed
- Reinstall dependencies if import errors occur
- Check dataset paths in `data/` or update config variables

---

## Learning Outcomes

This project demonstrates:
- Practical application of statistical inference in sustainability analytics
- Integration of interactive visualization with predictive modeling
- Use of Dash and Streamlit for data storytelling in environmental research
- Fundamentals of energy economics and CO₂ forecasting for data-driven policy insights

---

## Author

Atharva Kavade  
LinkedIn | GitHub

---

## Future Enhancements

- Integration with real-time energy APIs
- Advanced ML models (XGBoost, LSTM, Gradient Boosting) for improved forecasting
- Deployment on Streamlit Cloud, Render, or Heroku
- Automated PDF export of country-wise comparisons and reports

---