import streamlit as st

st.title("⚡ Energy Data Science Demos")
st.markdown(
    """
    Interactive demos from my data science portfolio. Each demo
    illustrates a different piece of the energy forecasting and
    optimization stack.

    Use the sidebar to navigate between demos.
    """
)

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### ☀️ Solar Forecasting")
    st.markdown(
        "Probabilistic solar irradiance forecasting with P10/P50/P90 "
        "uncertainty bands, benchmarked against a clear-sky baseline."
    )

with col2:
    st.markdown("### 🔋 Battery Optimization")
    st.markdown(
        "Commercial battery dispatch optimization. Charge during low "
        "prices, discharge during high prices. See how much a business "
        "can save."
    )

with col3:
    st.markdown("### 🚗 EV Charging Loads")
    st.markdown(
        "Predicting EV charging demand at the grid level. Understand "
        "peak load timing and transformer capacity impact."
    )

st.markdown("---")
st.caption(
    "Full portfolio with project writeups and source code: "
    "[your-portfolio-url.netlify.app](https://your-portfolio-url.netlify.app)"
)