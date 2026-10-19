"""
Aggregate NREL charging sessions to grid-node-hourly features.

This reproduces the exact feature pipeline used to train rf_grid_load.pkl.
Output: data/ev_grid_features.csv
"""
import pandas as pd
import numpy as np

# Load raw session data
print("Loading raw EV sessions...")
df = pd.read_csv(
    'D:/Projects/End-to-End Practice/Data Science/Energy/RT EV Charging Load Prediction & Grid Impact/data/processed/NREL_Workplace_Charging_Cleaned.csv',
    parse_dates=['start_date_time', 'end_date_time']
)
print(f"Loaded {len(df)} sessions")

# --- Grid node mapping (from your original code) ---
def map_station_to_grid(station):
    if station.startswith('LV22'):
        return 'GRID_LV22'
    elif station.startswith('LV31') or station.startswith('LV32'):
        return 'GRID_LV31'
    elif station.startswith('LV') and len(station) >= 4:
        return f"GRID_{station[:4]}"
    elif station.startswith('08'):
        return 'GRID_08A'
    elif station.startswith('05'):
        return 'GRID_05A'
    elif station.startswith('04'):
        return 'GRID_04A'
    elif station.startswith('07'):
        return 'GRID_07A'
    else:
        return 'GRID_OTHER'

df['grid_node'] = df['station'].apply(map_station_to_grid)
print(f"Grid nodes: {df['grid_node'].nunique()}")

# --- Create date + hour ---
df['date'] = df['start_date_time'].dt.date
df['hour'] = df['start_date_time'].dt.hour

# --- Aggregate to grid-node-hourly ---
print("\nAggregating to grid-node-hourly...")
grid_hourly = df.groupby(['grid_node', 'date', 'hour']).agg(
    total_energy_kwh=('energy_charged', 'sum'),
    active_chargers=('station', 'count'),
    unique_vehicles=('driverId', 'nunique'),
).reset_index()

print(f"Aggregated rows: {len(grid_hourly)}")

# --- Build time-based features ---
grid_hourly['date_dt'] = pd.to_datetime(grid_hourly['date'])
grid_hourly['day_of_week'] = grid_hourly['date_dt'].dt.dayofweek
grid_hourly['month'] = grid_hourly['date_dt'].dt.month
grid_hourly['week_of_year'] = grid_hourly['date_dt'].dt.isocalendar().week.astype(int)

grid_hourly['hour_sin'] = np.sin(2 * np.pi * grid_hourly['hour'] / 24)
grid_hourly['hour_cos'] = np.cos(2 * np.pi * grid_hourly['hour'] / 24)
grid_hourly['day_sin'] = np.sin(2 * np.pi * grid_hourly['day_of_week'] / 7)
grid_hourly['day_cos'] = np.cos(2 * np.pi * grid_hourly['day_of_week'] / 7)

# Add a clean timestamp
grid_hourly['timestamp'] = grid_hourly['date_dt'] + pd.to_timedelta(grid_hourly['hour'], unit='h')

# --- Keep only what we need ---
features = [
    'hour_sin', 'hour_cos',
    'day_sin', 'day_cos',
    'day_of_week', 'month', 'week_of_year',
    'active_chargers', 'unique_vehicles'
]

output = grid_hourly[['timestamp', 'grid_node'] + features + ['total_energy_kwh']].copy()
output = output.sort_values(['grid_node', 'timestamp']).reset_index(drop=True)
output = output.dropna()

print(f"\nFinal rows: {len(output)}")
print(f"Columns: {len(output.columns)}")
print(f"Date range: {output['timestamp'].min()} to {output['timestamp'].max()}")
print(f"Grid nodes: {output['grid_node'].nunique()}")
print("\nSample:")
print(output.head(10))

output.to_csv('data/ev_grid_features.csv', index=False)
print(f"\n✅ Saved to data/ev_grid_features.csv")