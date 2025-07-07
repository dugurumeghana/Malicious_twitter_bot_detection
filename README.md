# Malicious Twitter Bot Detection Using Machine Learning

This project demonstrates how machine learning can be used to identify malicious Twitter bot accounts based on publicly available metadata. The goal is to classify accounts as either bots or legitimate users by analyzing behavioral patterns such as tweet frequency, verification status, and follower ratios.

A simple and intuitive web app has been built using Streamlit, allowing users to upload a dataset and instantly get bot predictions, model performance metrics, and downloadable results.

---

## 🔍 Project Summary

Twitter bots are known for influencing conversations and spreading misinformation. Many of them leave behind patterns that make them detectable. This project focuses on:

- Extracting features from account metadata
- Training a logistic regression model to detect bots
- Deploying the model in a user-friendly web interface

The application is lightweight, fast, and easy to use — ideal for experimentation or showcasing as part of a portfolio.

---

## 📁 Folder Structure

Malicious-TwitterBot-Detection/
├── app.py # Streamlit frontend application
├── model_training.ipynb # Google Colab notebook used for training the model
├── /models/
│ └── logistic_model.pkl # Trained logistic regression model
├── /dataset/
│ └── kaggle_tweets.csv # Sample dataset for input
├── requirements.txt # List of Python dependencies
└── README.md # Project documentation



---

## ✅ Features

- Upload a CSV of Twitter user data
- Extracts custom features automatically
- Predicts which accounts are likely bots
- Shows prediction results in a table
- Displays ROC curve with AUC score
- Allows downloading prediction output as CSV

---

## 🧠 Model Details

- **Algorithm**: Logistic Regression
- **Accuracy**: 68.47%
- **Key Features Used**:
  - `activity`: Tweets per day (statuses_count divided by account age in days)
  - `anonymity`: Whether the account is verified or not
  - `amplification`: Ratio of followers to friends (follow-back behavior)

These features were chosen because they are commonly associated with automated account behavior.

---

## 📊 Dataset Requirements

To use this app, your dataset must include the following columns:

| Column Name        | Description                            |
|--------------------|----------------------------------------|
| `created_at`       | Date when the account was created      |
| `verified`         | Boolean flag indicating verification   |
| `statuses_count`   | Number of tweets posted by the user    |
| `followers_count`  | Number of followers                    |
| `friends_count`    | Number of accounts the user follows    |

Sample dataset: `dataset/kaggle_tweets.csv`

---

 💻 Getting Started

 1. Install the required packages:

pip install -r requirements.txt

2. Run the Streamlit app:
streamlit run app.py

3. Use the interface:
Upload a .csv file with the required columns

View prediction results

Check the ROC curve for performance

Download predictions as a new CSV

RESULTS:
![image](https://github.com/user-attachments/assets/cc234d76-5369-41f1-8a11-67144a17c733)


