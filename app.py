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

# Set page config
st.set_page_config(page_title="Malicious Bot Detector", page_icon="🤖", layout="wide")

# Custom dark brown theme using your palette
st.markdown("""
    <style>
        body {
            background-color: #432818;
            color: #ffe6a7;
        }
        .main {
            background-color: #432818;
            color: #ffe6a7;
        }
        .stApp {
            background-color: #432818;
            color: #ffe6a7;
        }
        h1, h2, h3, h4 {
            color: #ffe6a7;
        }
        .stButton>button {
            background-color: #bb9457;
            color: white;
            border-radius: 10px;
            padding: 0.6em 1em;
            font-weight: bold;
        }
        .stDownloadButton>button {
            background-color: #99582a;
            color: white;
            border-radius: 10px;
            padding: 0.6em 1em;
            font-weight: bold;
        }
        .stDataFrame {
            background-color: #ffe6a7;
            color: #432818;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("🤖 Malicious Twitter Bot Detection")
st.markdown("Detect suspicious Twitter accounts using ML — powered by logistic regression.")

# File Upload
uploaded_file = st.file_uploader("📂 Upload Twitter User Dataset (.csv)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Raw Uploaded Data")
    st.dataframe(df.head(), use_container_width=True)

    # Check required columns
    required_cols = ['created_at', 'verified', 'statuses_count', 'followers_count', 'friends_count']
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must include the following columns: {', '.join(required_cols)}")
    else:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True)
        today = pd.to_datetime("today", utc=True)
        df['account_age_days'] = (today - df['created_at']).dt.days
        df['activity'] = df['statuses_count'] / (df['account_age_days'] + 1)
        df['anonymity'] = (~df['verified']).astype(int)
        df['amplification'] = df['followers_count'] / (df['friends_count'] + 1)

        features = df[['activity', 'anonymity', 'amplification']].fillna(0)
        preds = model.predict(features)
        df['prediction'] = preds

        st.subheader("🧠 Prediction Results")
        st.dataframe(df[['screen_name', 'activity', 'anonymity', 'amplification', 'prediction']].head(10), use_container_width=True)

        bot_count = int(df['prediction'].sum())
        st.success(f"🔍 Detected {bot_count} bots out of {len(df)} users.")

        # ROC Curve
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(features)[:, 1]
            fpr, tpr, _ = roc_curve(df['prediction'], y_prob)
            roc_auc = auc(fpr, tpr)

            st.subheader("📊 ROC Curve")
            fig, ax = plt.subplots()
            ax.set_facecolor("#ffe6a7")
            fig.patch.set_facecolor('#ffe6a7')
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='#6f1d1b')
            ax.plot([0, 1], [0, 1], color='#bb9457', linestyle='--')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve", color='#432818')
            ax.legend()
            st.pyplot(fig)

        # Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Predictions", data=csv, file_name="bot_predictions.csv", mime='text/csv')
