"""
Retrain the solar forecasting model with clean preprocessing.
Retrain the solar forecasting model for a TRUE 6-hour-ahead forecast.

This script was created because the originally deployed model showed
high error (481 W/m² MAE) when applied to full-year data. The retraining
uses a chronological train/test split to properly evaluate generalization.
The previous model was predicting "now" (nowcasting), not forecasting.
This version shifts the target 12 rows forward (6 hours in 30-min data)
so the model predicts solar irradiance 6 hours into the future.


Outputs:
  - models/solar_p50_v2.pkl
  - models/solar_scaler_v2.pkl
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib
import os

# --- Configuration ---
DATA_PATH = 'data/palm_springs_features.csv'
MODEL_OUT = 'models/solar_p50_v2.pkl'
SCALER_OUT = 'models/solar_scaler_v2.pkl'
HORIZON_ROWS = 12  # 12 rows × 30 min = 6 hours ahead

FEATURES = [
    'cos_zenith', 'solar_elevation', 'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos',
    'total_cloud_cover', 'lag_1h', 'lag_2h', 'lag_3h', 'lag_6h',
    'ghi_rolling_mean_1h', 'ghi_rolling_std_1h', 'csi_lag_1h', 'csi_lag_2h',
    'cloud_type', 'cloud_0', 'cloud_1', 'cloud_2', 'cloud_3', 'cloud_4',
    'cloud_5', 'cloud_6', 'cloud_7', 'cloud_8', 'cloud_9',
]

# --- Load data ---
print(f"Loading {DATA_PATH}...")
df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
print(f"Loaded {len(df)} rows")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# --- Create the target: GHI 6 hours ahead ---
df['target'] = df['ghi'].shift(-HORIZON_ROWS)
df = df.dropna(subset=['target'] + FEATURES).reset_index(drop=True)
print(f"After target shift, {len(df)} rows remain")

# --- Chronological 80/20 split ---
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

print(f"\nTrain: {len(train)} rows ({train['timestamp'].min()} to {train['timestamp'].max()})")
print(f"Test:  {len(test)} rows ({test['timestamp'].min()} to {test['timestamp'].max()})")

# --- Scale ---
scaler = StandardScaler()
X_train = scaler.fit_transform(train[FEATURES])
X_test = scaler.transform(test[FEATURES])
y_train = train['target'].values
y_test = test['target'].values

# --- Train ---
print("\nTraining Random Forest for 6-hour-ahead forecast...")
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2,
)
model.fit(X_train, y_train)

# --- Evaluate ---
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

train_mae = mean_absolute_error(y_train, pred_train)
test_mae = mean_absolute_error(y_test, pred_test)

print(f"\nResults (6-hour-ahead):")
print(f"  Train MAE: {train_mae:.2f} W/m²")
print(f"  Test MAE:  {test_mae:.2f} W/m²")

# --- Sample predictions ---
print("\nSample predictions (target is 6h ahead):")
np.random.seed(42)
sample_idx = np.random.choice(len(test), 8, replace=False)
print(f"{'Predicted':<12} {'Actual (t+6h)':<15} {'Error':<12}")
print('-' * 42)
for i in sample_idx:
    pred = pred_test[i]
    actual = y_test[i]
    print(f"{pred:<12.1f} {actual:<15.1f} {abs(pred - actual):<12.1f}")

# --- Save ---
os.makedirs('models', exist_ok=True)
joblib.dump(model, MODEL_OUT)
joblib.dump(scaler, SCALER_OUT)
print(f"\n✅ Saved:")
print(f"   {MODEL_OUT}")
print(f"   {SCALER_OUT}")