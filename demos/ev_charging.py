import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
import os

st.title("🚗 EV Charging Load Prediction")
st.markdown(
    """
    A **Random Forest model** trained on NREL workplace charging data
    predicts hourly energy demand at the grid-node level.

    Select a grid node to see which days have charging activity,
    then pick one to see the model's performance.
    """
)

@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)
    return joblib.load(os.path.join(base, "..", "models", "ev_grid_rf.pkl"))

@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    return pd.read_csv(
        os.path.join(base, "..", "data", "ev_grid_features.csv"),
        parse_dates=['timestamp'],
    )

model = load_model()
df = load_data()

FEATURES = [
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'day_of_week', 'month', 'week_of_year',
    'active_chargers', 'unique_vehicles',
]

# --- Sidebar ---
grid_nodes = sorted(df['grid_node'].unique())

with st.sidebar:
    st.subheader("Demo Controls")
    selected_node = st.selectbox("Grid node", grid_nodes)

# --- Show node stats ---
node_df = df[df['grid_node'] == selected_node].copy()
node_df['date'] = node_df['timestamp'].dt.date
available_dates = sorted(node_df['date'].unique())

# Node summary metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Charging Hours", f"{len(node_df)}")
col2.metric("Days with Activity", f"{len(available_dates)}")
col3.metric("Total Energy", f"{node_df['total_energy_kwh'].sum():.0f} kWh")

st.markdown("---")

# --- Date selector from available dates only ---
st.subheader(f"Pick a date with data for {selected_node}")
selected_date = st.select_slider(
    "Available dates",
    options=available_dates,
    value=available_dates[len(available_dates) // 2],
)

# --- Filter ---
day_df = node_df[node_df['date'] == selected_date].copy()

if len(day_df) == 0:
    st.warning("No data for that date.")
    st.stop()

# --- Predict ---
day_df['predicted'] = model.predict(day_df[FEATURES].values)
day_df['error'] = day_df['predicted'] - day_df['total_energy_kwh']

# --- Metrics ---
mae = np.mean(np.abs(day_df['error']))
total_actual = day_df['total_energy_kwh'].sum()
total_pred = day_df['predicted'].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE", f"{mae:.2f} kWh")
col2.metric("Total Actual", f"{total_actual:.1f} kWh")
col3.metric("Total Predicted", f"{total_pred:.1f} kWh")
col4.metric("Charging Hours", f"{len(day_df)}")
# After the metrics row:
if len(day_df) < 4:
    st.caption(
        f"⚠️ This day only has {len(day_df)} hour(s) of charging activity for {selected_node}. "
        f"The full dataset spans {len(available_dates)} days with activity for this node."
    )

# --- Chart ---
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(day_df['timestamp'], day_df['total_energy_kwh'],
       width=0.02, color='#1B4332', alpha=0.7, label='Actual')
ax.plot(day_df['timestamp'], day_df['predicted'], 'o--',
        color='#F4A261', linewidth=2, markersize=8, label='Predicted')
ax.set_xlabel('Time of day')
ax.set_ylabel('Energy Delivered (kWh)')
ax.set_title(f'{selected_node} — {selected_date}')
ax.legend()
ax.grid(True, alpha=0.3)

# Only format time ticks if the day has more than 1 data point
if len(day_df) > 1:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=0)

plt.tight_layout()
st.pyplot(fig)

# --- Table ---
with st.expander("View hourly breakdown"):
    table_df = day_df[['timestamp', 'active_chargers', 'unique_vehicles',
                       'total_energy_kwh', 'predicted', 'error']].copy()
    table_df['timestamp'] = table_df['timestamp'].dt.strftime('%H:%M')
    table_df.columns = ['Time', 'Active Chargers', 'Unique Vehicles',
                        'Actual (kWh)', 'Predicted (kWh)', 'Error (kWh)']
    st.dataframe(
        table_df.round(2),
        use_container_width=True,
        hide_index=True,
    )

st.markdown("---")
st.caption(
    "**Model:** Random Forest (n=100) trained on aggregated NREL workplace "
    "charging sessions (2019-2021). Features: hour/day cyclical encoding, "
    "calendar features, active chargers per hour, and unique vehicles per hour. "
    "**Overall MAE: 4.83 kWh** across 8,565 grid-node-hourly samples. "
    "Data is sparse by design — most grid nodes have activity only on selected days."
)