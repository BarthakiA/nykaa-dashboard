import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils.churn_model import train_churn_model, predict_churn, compute_roc
from sklearn.model_selection import train_test_split

st.title("Churn Prediction for First-Time Buyers")

st.markdown("""
We use classification to predict churn risk for new buyers based 
on early behavior, helping Nykaa reduce first-time buyer churn.
""")

data = pd.read_csv("data/churn_ready.csv")  # Pre-processed

X = data.drop('churn', axis=1)
y = data['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = train_churn_model(X_train, y_train)
preds = predict_churn(model, X_test)

st.subheader("Feature Importances")
importances = model.feature_importances_
fig, ax = plt.subplots()
ax.barh(X_train.columns, importances)
st.pyplot(fig)

st.subheader("ROC Curve")
fpr, tpr, roc_auc = compute_roc(y_test, preds)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax.plot([0,1],[0,1],'--')
ax.legend()
st.pyplot(fig)
