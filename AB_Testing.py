import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

st.title("A/B Testing Simulation")

st.markdown("""
This page demonstrates how Nykaa can evaluate marketing interventions 
using A/B testing to reduce churn or increase repeat purchases.
""")

n = 500
control = np.random.normal(0.2, 0.05, n)
treatment = np.random.normal(0.25, 0.05, n)

st.subheader("Retention Rate Distribution")
fig, ax = plt.subplots()
ax.hist(control, alpha=0.6, label='Control')
ax.hist(treatment, alpha=0.6, label='Treatment')
ax.legend()
st.pyplot(fig)

t_stat, p_val = ttest_ind(treatment, control)
st.subheader("T-Test Result")
st.write(f"T-statistic: {t_stat:.2f}, p-value: {p_val:.4f}")

if p_val < 0.05:
    st.success("Statistically significant uplift detected!")
else:
    st.warning("No significant difference detected.")
