import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils.cltv_model import fit_bg_nbd, fit_gamma_gamma, calculate_cltv

st.title("Customer Lifetime Value (CLTV) Prediction")

st.markdown("""
This section uses the BG/NBD and Gamma-Gamma models to predict 
future purchases and revenue over 3 months.
""")

data = pd.read_csv("data/NYKA.csv")
summary = pd.read_csv("data/cltv_ready.csv")  # Pre-engineered RFM summary for lifetimes

bgf = fit_bg_nbd(summary, 'frequency', 'recency', 'T')
ggf = fit_gamma_gamma(summary, 'frequency', 'monetary_value')

summary['CLTV'] = calculate_cltv(bgf, ggf, summary)

st.subheader("CLTV Distribution")
fig, ax = plt.subplots()
summary['CLTV'].hist(bins=30, ax=ax)
st.pyplot(fig)

st.subheader("Top Customers by Predicted CLTV")
st.dataframe(summary.sort_values('CLTV', ascending=False).head(10))
