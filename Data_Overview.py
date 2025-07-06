import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Data Overview")

st.markdown("""
This section shows the raw NYKA.csv data structure and key statistics 
to ensure data quality before modeling.
""")

data = pd.read_csv("data/NYKA.csv")
st.subheader("Raw Data Sample")
st.dataframe(data.head())

st.subheader("Summary Statistics")
st.write(data.describe())

st.subheader("Missing Values Heatmap")
fig, ax = plt.subplots()
sns.heatmap(data.isnull(), cbar=False)
st.pyplot(fig)

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
st.pyplot(fig)
