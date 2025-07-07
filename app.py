# app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load the trained model
with open("models/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🤖 Malicious Twitter Bot Detection")
st.write("Upload a Twitter user dataset and detect bots using a trained ML model.")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Uploaded Data")
    st.dataframe(df.head())

    # Check necessary columns
    required_cols = ['created_at', 'verified', 'statuses_count', 'followers_count', 'friends_count']
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must include the following columns: {', '.join(required_cols)}")
    else:
        # Preprocessing
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True)
        today = pd.to_datetime("today", utc=True)
        df['account_age_days'] = (today - df['created_at']).dt.days
        df['activity'] = df['statuses_count'] / (df['account_age_days'] + 1)
        df['anonymity'] = (~df['verified']).astype(int)
        df['amplification'] = df['followers_count'] / (df['friends_count'] + 1)

        features = df[['activity', 'anonymity', 'amplification']]
        features = features.fillna(0)

        # Prediction
        preds = model.predict(features)
        df['prediction'] = preds

        st.subheader("Prediction Results")
        st.dataframe(df[['screen_name', 'activity', 'anonymity', 'amplification', 'prediction']].head(10))

        bot_count = df['prediction'].sum()
        st.success(f"Detected {bot_count} bots out of {len(df)} users.")

        # Optional: ROC Curve
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(features)[:, 1]
            fpr, tpr, _ = roc_curve(df['prediction'], y_prob)
            roc_auc = auc(fpr, tpr)

            st.subheader("ROC Curve")
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='blue')
            ax.plot([0, 1], [0, 1], color='red', linestyle='--')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend()
            st.pyplot(fig)

        # Allow downloading results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", data=csv, file_name="bot_predictions.csv", mime='text/csv')