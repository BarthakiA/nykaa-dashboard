import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.rfm_analysis import calculate_rfm, perform_kmeans_clustering

st.title("Customer Segmentation with RFM and K-Means")

st.markdown("""
This section uses Recency, Frequency, and Monetary (RFM) analysis 
with K-Means clustering to segment Nykaa customers.
""")

data = pd.read_csv("data/NYKA.csv")

rfm = calculate_rfm(data, 'CustomerID', 'InvoiceDate', 'TotalAmount')
rfm = perform_kmeans_clustering(rfm)

st.subheader("RFM Table with Clusters")
st.dataframe(rfm.head())

st.subheader("Cluster Count")
st.bar_chart(rfm['Cluster'].value_counts())

st.subheader("Pairplot of RFM Segments")
fig = sns.pairplot(rfm, hue='Cluster', palette='tab10')
st.pyplot(fig)
