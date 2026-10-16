import streamlit as st

st.set_page_config(
    page_title="Energy Data Science Demos",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define navigation
pages = {
    "Overview": [
        st.Page("demos/home.py", title="Home", icon="🏠", default=True),
    ],
    "Interactive Demos": [
        st.Page("demos/solar_forecast.py", title="Solar Forecasting", icon="☀️"),
        st.Page("demos/battery_optimization.py", title="Battery Optimization", icon="🔋"),
        st.Page("demos/ev_charging.py", title="EV Charging Loads", icon="🚗"),
    ],
}

pg = st.navigation(pages)
pg.run()

# Persistent sidebar footer
with st.sidebar:
    st.markdown("---")
    st.markdown(
        """
        Built by **Denylson Fuentes**

        [Portfolio](https://denylson-fuentes.netlify.app/) ·
        [GitHub](https://github.com/Denylson-Fuentes)
        """
    )