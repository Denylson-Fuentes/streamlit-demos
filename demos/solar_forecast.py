import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

st.title("☀️ Probabilistic Solar Irradiance Forecast")
st.markdown(
    """
    A **Random Forest quantile regressor** trained on NSRDB solar data
    forecasts solar irradiance (GHI) for Palm Springs, CA **6 hours ahead**.

    Pick a date to compare the model's predictions against actual measurements.
    """
)

# --- Load assets (cached) ---
@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)
    model = joblib.load(os.path.join(base, "..", "models", "solar_p50_v2.pkl"))
    scaler = joblib.load(os.path.join(base, "..", "models", "solar_scaler_v2.pkl"))
    return model, scaler

@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    df = pd.read_csv(
        os.path.join(base, "..", "data", "palm_springs_features.csv"),
        parse_dates=['timestamp'],
    )
    return df

model, scaler = load_model()
df = load_data()

FEATURES = [
    'cos_zenith', 'solar_elevation', 'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos',
    'total_cloud_cover', 'lag_1h', 'lag_2h', 'lag_3h', 'lag_6h',
    'ghi_rolling_mean_1h', 'ghi_rolling_std_1h', 'csi_lag_1h', 'csi_lag_2h',
    'cloud_type', 'cloud_0', 'cloud_1', 'cloud_2', 'cloud_3', 'cloud_4',
    'cloud_5', 'cloud_6', 'cloud_7', 'cloud_8', 'cloud_9',
]

# Create the target column matching training (GHI 6 hours = 12 rows ahead)
HORIZON = 12
df['target'] = df['ghi'].shift(-HORIZON)
df = df.dropna(subset=['target']).reset_index(drop=True)

# --- Sidebar controls ---
min_date = df['timestamp'].min().date()
max_date = df['timestamp'].max().date()

with st.sidebar:
    st.subheader("Demo Controls")
    selected_date = st.date_input(
        "Select a date",
        value=min_date + pd.Timedelta(days=200),
        min_value=min_date,
        max_value=max_date,
    )
    st.caption(
        f"Data covers {min_date} to {max_date}. "
        "Model forecasts 6 hours ahead."
    )

# --- Filter to selected day ---
day_df = df[df['timestamp'].dt.date == selected_date].copy()

if len(day_df) == 0:
    st.warning("No data for that date. Please pick another.")
    st.stop()

# --- Run predictions ---
X = scaler.transform(day_df[FEATURES])
day_df['predicted'] = model.predict(X)
day_df['error'] = day_df['predicted'] - day_df['target']

# --- Metrics ---
mae = np.mean(np.abs(day_df['error']))
peak_actual = day_df['target'].max()

col1, col2, col3 = st.columns(3)
col1.metric("Mean Absolute Error", f"{mae:.1f} W/m²")
col2.metric("Peak Actual GHI (t+6h)", f"{peak_actual:.0f} W/m²")
col3.metric("Forecast Points", f"{len(day_df)}")

# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(day_df['timestamp'], day_df['target'], 'o-',
        color='#1B4332', linewidth=2, markersize=4,
        label='Actual GHI (6h ahead)')
ax.plot(day_df['timestamp'], day_df['predicted'], 's--',
        color='#F4A261', linewidth=2, markersize=4,
        label='Predicted GHI')
ax.set_xlabel('Time of day')
ax.set_ylabel('GHI (W/m²)')
title_start = day_df['timestamp'].min().strftime('%Y-%m-%d %H:%M')
title_end = day_df['timestamp'].max().strftime('%H:%M')
ax.set_title(f'6-Hour-Ahead Solar Forecast — {title_start} to {title_end}')
ax.legend()
ax.grid(True, alpha=0.3)

# Format x-axis as time only
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
plt.tight_layout()
st.pyplot(fig)

# --- Data table ---
with st.expander("View hourly breakdown"):
    display_df = day_df[['timestamp', 'target', 'predicted', 'error']].copy()
    display_df.columns = ['Timestamp', 'Actual (t+6h)', 'Predicted', 'Error']
    display_df['Timestamp'] = display_df['Timestamp'].dt.strftime('%H:%M')
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# --- Model notes ---
st.markdown("---")
st.caption(
    "**Model:** Random Forest (n=100) trained on NSRDB data for Palm Springs, CA. "
    "Features: solar position, cloud cover, GHI lags (1h, 2h, 3h, 6h), "
    "rolling statistics, and Clear Sky Index lags. "
    "**Test MAE: 39.4 W/m²** on a chronological 80/20 split (Oct–Dec 2023). "
    "This is a **6-hour-ahead** forecast, not a nowcast."
)