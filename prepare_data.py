"""
Precompute the 18 engineered features required by the solar forecast model.
Output: data/palm_springs_features.csv
"""
import pandas as pd

# Load the base processed data
df = pd.read_csv('data/palm_springs_processed.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# Detect timestep in hours
time_diff = df['timestamp'].diff().dt.total_seconds().dropna().mode()[0]
timestep_hours = time_diff / 3600
print(f"Detected timestep: {timestep_hours} hours")

# Compute lag steps (how many rows per hour)
steps_per_hour = int(1 / timestep_hours)

# Build lag features
df['lag_1h'] = df['ghi'].shift(1 * steps_per_hour)
df['lag_2h'] = df['ghi'].shift(2 * steps_per_hour)
df['lag_3h'] = df['ghi'].shift(3 * steps_per_hour)
df['lag_6h'] = df['ghi'].shift(6 * steps_per_hour)

# CSI lags
df['csi_lag_1h'] = df['csi'].shift(1 * steps_per_hour)
df['csi_lag_2h'] = df['csi'].shift(2 * steps_per_hour)

# Rolling stats (closed='left' matches original training code)
df['ghi_rolling_mean_1h'] = df['ghi'].rolling(window=steps_per_hour, closed='left').mean()
df['ghi_rolling_std_1h'] = df['ghi'].rolling(window=steps_per_hour, closed='left').std()

# Cloud type one-hot
for i in range(10):
    df[f'cloud_{i}'] = (df['cloud_type'] == i).astype(int)

# Drop rows with NaN in critical features (first few rows after shift)
required = [
    'cos_zenith', 'solar_elevation', 'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos',
    'total_cloud_cover',
    'lag_1h', 'lag_2h', 'lag_3h', 'lag_6h',
    'ghi_rolling_mean_1h', 'ghi_rolling_std_1h',
    'csi_lag_1h', 'csi_lag_2h', 'cloud_type',
    'cloud_0', 'cloud_1', 'cloud_2', 'cloud_3', 'cloud_4',
    'cloud_5', 'cloud_6', 'cloud_7', 'cloud_8', 'cloud_9',
    'ghi', 'csi'  # keep these for display
]
df = df.dropna(subset=required)
df = df.reset_index(drop=True)

print(f"Final rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Sample:")
print(df[['timestamp', 'ghi', 'lag_1h', 'csi_lag_1h', 'cloud_0', 'cloud_6']].head())

# Save
df.to_csv('data/palm_springs_features.csv', index=False)
print(f"\nSaved to data/palm_springs_features.csv")