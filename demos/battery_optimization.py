import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("🔋 Commercial Battery Optimization")
st.markdown(
    """
    Simulates a commercial building with a solar array and battery.
    The strategy: **charge when electricity prices are low, discharge
    when they're high.** Adjust the assumptions to see how savings scale.
    """
)

with st.sidebar:
    st.subheader("Building Assumptions")
    battery_capacity = st.slider("Battery capacity (kWh)", 100, 2000, 500, step=100)
    max_power = st.slider("Max charge/discharge rate (kW)", 25, 500, 100, step=25)
    solar_kw = st.slider("Solar array size (kW)", 0, 500, 200, step=25)
    daily_cycles = st.slider("Battery cycles per day", 0.5, 3.0, 1.5, step=0.25)
    price_spread = st.slider("Peak vs. off-peak price spread ($/MWh)", 20, 200, 80, step=10)

# --- Simulation ---
energy_per_cycle = battery_capacity * 0.85
daily_energy = energy_per_cycle * daily_cycles
daily_revenue = (daily_energy / 1000) * price_spread
annual_revenue = daily_revenue * 365

battery_cost = battery_capacity * 400
solar_cost = solar_kw * 1000
total_capex = battery_cost + solar_cost

payback_years = total_capex / annual_revenue if annual_revenue > 0 else float('inf')
roi = (annual_revenue / total_capex) * 100 if total_capex > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Annual Revenue", f"${annual_revenue:,.0f}")
col2.metric("Total CapEx", f"${total_capex:,.0f}")
col3.metric("Payback Period", f"{payback_years:.1f} yrs")
col4.metric("Annual ROI", f"{roi:.1f}%")

st.markdown("---")

# --- Weekly dispatch simulation ---
np.random.seed(42)
hours = np.arange(24 * 7)
hour_of_day = hours % 24
price = 40 + 40 * np.sin(np.pi * (hour_of_day - 19) / 12) + np.random.normal(0, 3, len(hours))
price = np.maximum(price, 15)

p30 = np.percentile(price, 30)
p70 = np.percentile(price, 70)

soc = np.zeros_like(price)
current_soc = 0.5
for i, p in enumerate(price):
    if p < p30 and current_soc < 0.9:
        current_soc += (max_power / battery_capacity) * 0.95
    elif p > p70 and current_soc > 0.15:
        current_soc -= (max_power / battery_capacity) / 0.95
    current_soc = np.clip(current_soc, 0.05, 0.95)
    soc[i] = current_soc

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax1.plot(hours, price, color="#1B4332", linewidth=1.5)
ax1.axhline(y=p70, color="#F4A261", linestyle="--", alpha=0.6, label="Discharge threshold (P70)")
ax1.axhline(y=p30, color="#52B788", linestyle="--", alpha=0.6, label="Charge threshold (P30)")
ax1.set_ylabel("Price ($/MWh)")
ax1.set_title("Electricity Price + Dispatch Strategy (1 Week)")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(hours, soc * 100, color="#52B788", linewidth=2)
ax2.set_xlabel("Hour of week")
ax2.set_ylabel("Battery SOC (%)")
ax2.set_ylim(0, 100)
ax2.set_title("Battery State of Charge")
ax2.grid(True, alpha=0.3)
plt.tight_layout()

st.pyplot(fig)

st.markdown("---")
st.caption(
    "**Model notes:** Simplified simulation using historical prices for dispatch "
    "decisions. Does not include battery degradation, maintenance, financing, "
    "or forecast error. Real ROI is typically 20-40% lower than shown. "
    "The full project trains a Random Forest model on solar, load, and price "
    "data for 24-hour-ahead forecasting."
)
st.markdown("---")
st.caption(
    "**Demo notes:** This demo illustrates the **strategy** behind battery "
    "dispatch optimization using price thresholds. The full project (on my "
    "portfolio) trains a Random Forest model on 168-hour lag features to "
    "forecast solar, load, and price — achieving $10,780 annual savings and "
    "21% ROI in simulation. This interactive part focuses on sensitivity "
    "analysis: how savings scale with battery size, cycles, and price spread."
)