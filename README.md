# ⚡ Energy Data Science Demos

Interactive demos showcasing forecasting models and analysis pipelines built across energy-sector projects.

Built by [Denylson Fuentes](https://denylson-fuentes.netlify.app) · [Live Demo](https://app-demos-fz5uvifcahsqphom6xhslp.streamlit.app)

---

## What's Inside

Three demos, one app. Use the sidebar to switch between them.

| Demo | What It Does | Model |
|------|--------------|-------|
| ☀️ **Solar Forecasting** | 6-hour-ahead solar irradiance forecasts for Palm Springs with P10/P50/P90 uncertainty bands, benchmarked against a clear-sky baseline. | Random Forest (26 features) |
| 🔋 **Battery Optimization** | Interactive sensitivity analysis for commercial battery dispatch. Adjust battery size, cycles, price spread, and see the impact on annual revenue and ROI. | Price-threshold strategy (visualization) |
| 🚗 **EV Charging Loads** | Grid-node-level hourly energy demand predictions from NREL workplace charging data. Select any node and date with recorded activity. | Random Forest (9 features) |

---

## Architecture

streamlit-demos/
├── app.py # Entry point — st.navigation routes to pages
├── demos/
│ ├── home.py # Landing page
│ ├── solar_forecast.py # Solar demo
│ ├── battery_optimization.py
│ └── ev_charging.py
├── data/
│ ├── palm_springs_features.csv
│ └── ev_grid_features.csv
├── models/
│ ├── solar_p50_v2.pkl
│ ├── solar_scaler_v2.pkl
│ └── ev_grid_rf.pkl
└── requirements.txt


**Multi-page pattern:** Streamlit's `st.navigation` lets you define pages programmatically. One app, one URL, multiple demos accessible from the sidebar.

---

## Model Details

### Solar Forecasting

- **Input:** 26 engineered features — solar position (`cos_zenith`, `solar_elevation`), cyclical time (`hour_sin/cos`, `doy_sin/cos`), GHI lags (1h, 2h, 3h, 6h), rolling statistics, cloud type (one-hot), and Clear Sky Index lags.
- **Target:** GHI 6 hours ahead (`ghi.shift(-12)` for 30-minute data)
- **Performance:** Train MAE 15.17 W/m², **Test MAE 39.37 W/m²** on a chronological 80/20 split.
- **Data:** NSRDB 2023, Palm Springs, CA. 13,168 readings → 13,156 after target shifting.

### EV Charging Loads

- **Input:** 9 features — cyclical hour/day encoding, calendar features (`day_of_week`, `month`, `week_of_year`), active chargers per hour, unique vehicles per hour.
- **Target:** Total energy delivered (kWh) per grid-node per hour.
- **Performance:** Train MAE 4.30 kWh, **Test MAE 7.18 kWh** on a chronological 80/20 split.
- **Data:** NREL Workplace Charging Dataset. 14,588 sessions → 3,216 grid-node-day aggregates across 10 grid nodes.

### Battery Optimization

- **Type:** Sensitivity analysis, not a trained ML model.
- **What it shows:** How annual revenue, payback period, and ROI scale with battery capacity (100–2,000 kWh), daily cycles (0.5–3), price spread ($20–200/MWh), and solar array size (0–500 kW).
- **Assumptions:** $400/kWh battery cost, $1,000/kW solar cost, 85% usable depth of discharge. Does **not** include degradation, maintenance, financing, or forecast error — real ROI is typically 20-40% lower.

---

## Running Locally

```bash
git clone https://github.com/Denylson-Fuentes/streamlit-demos.git
cd streamlit-demos

pip install -r requirements.txt

streamlit run app.py