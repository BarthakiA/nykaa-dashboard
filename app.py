import streamlit as st

st.set_page_config(
    page_title="Nykaa Customer Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Nykaa Customer Analytics Dashboard")
st.markdown("""
Welcome to Nykaa's data analytics platform.

Use the left sidebar to navigate to:
- Data Overview
- Customer Segmentation
- CLTV Prediction
- Churn Model
- A/B Testing Simulation

Each section includes interactive charts with filters and interpretations to support Nykaa's marketing strategy.
""")
