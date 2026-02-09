import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix,
    classification_report
)

import joblib
import pickle

# ---------------------------------------------------------
# Helper functions to load models
# ---------------------------------------------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_joblib(path):
    return joblib.load(path)

# ---------------------------------------------------------
# Load saved scaler + extract expected feature names
# ---------------------------------------------------------
scaler = joblib.load("model/scaler.pkl")
scaler_features = list(scaler.feature_names_in_)

# ---------------------------------------------------------
# Streamlit App Title
# ---------------------------------------------------------
st.title("📊 Telco Customer Churn Prediction – ML App")

# ---------------------------------------------------------
# Student Information
# ---------------------------------------------------------
st.markdown("### 🧑‍🎓 Student Information")
st.write("**BITS ID:** 2025AA05571")
st.write("**Full Name:** SURESH BABU MUSIREDDY")

st.write("""
Upload your **test dataset**, choose a **model**, view **evaluation metrics**,  
and generate **confusion matrix / classification report**.
""")

# ---------------------------------------------------------
# Dataset Upload Section
# ---------------------------------------------------------
st.header("📁 Upload Test Dataset (CSV)")
uploaded_file = st.file_uploader("Upload your test CSV file", type=["csv"])

# ---------------------------------------------------------
# Load Main Training Dataset (from GitHub)
# ---------------------------------------------------------
@st.cache_data
def load_training_data():
    url = "https://raw.githubusercontent.com/smusireddy/datasetfiles/refs/heads/main/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    return df

df = load_training_data()

# ---------------------------------------------------------
# Preprocessing Function
# ---------------------------------------------------------
def preprocess_data(df):
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col])

    return df

df_processed = preprocess_data(df)

# ---------------------------------------------------------
# Train/Test Split
# ---------------------------------------------------------
with st.container():
    st.header("🔀 Train/Test Split")

    test_size = st.slider("Select Test Size", 0.1, 0.4, 0.2)
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df["Churn"]
    )
    
    st.success(f"Dataset successfully split for download only! Test size: {test_size*100:.0f}%")
    st.subheader("📥 Download Test Dataset")

    test_csv = test_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Test CSV",
        data=test_csv,
        file_name="test_dataset.csv",
        mime="text/csv",
        key="download_test_csv"
    )

X = df_processed.drop("Churn", axis=1)
y = df_processed["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# Align X_test with scaler's expected features
# ---------------------------------------------------------
for col in scaler_features:
    if col not in X_test.columns:
        X_test[col] = 0

X_test = X_test[scaler_features]
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# Load ML Models
# ---------------------------------------------------------
models = {
    "Logistic Regression": load_pickle("model/logistic_regression.pkl"),
    "Decision Tree Classifier": load_pickle("model/decision_tree.pkl"),
    "K-Nearest Neighbor Classifier": load_pickle("model/knn.pkl"),
    "Naive Bayes Classifier - Gaussian": load_pickle("model/naive_bayes.pkl"),
    "Ensemble Model - Random Forest (Optimized)": load_joblib("model/random_forest.pkl"),
    "Ensemble Model - XGBoost": load_pickle("model/xgboost.pkl")
}

# ---------------------------------------------------------
# Model Selection
# ---------------------------------------------------------
st.header("🤖 Select a Machine Learning Model")
model_choice = st.selectbox("Choose a model", list(models.keys()))
model = models[model_choice]

# ---------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------
st.header("📊 Evaluation Metrics")

y_pred = model.predict(X_test_scaled)

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "MCC Score"],
    "Score": [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        matthews_corrcoef(y_test, y_pred)
    ]
})

st.dataframe(metrics_df)

# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------
st.header("🧩 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# ---------------------------------------------------------
# Classification Report
# ---------------------------------------------------------
st.header("📄 Classification Report (Formatted)")
report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
st.dataframe(report_df)

# ---------------------------------------------------------
# Prediction on Uploaded Test Data
# ---------------------------------------------------------
if uploaded_file is not None:
    st.header("🔍 Predictions on Uploaded Test Data")

    test_df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Test Data Preview")
    st.dataframe(test_df.head())

    test_processed = preprocess_data(test_df)

    # Align with scaler features
    for col in scaler_features:
        if col not in test_processed.columns:
            test_processed[col] = 0

    test_processed = test_processed[scaler_features]
    test_scaled = scaler.transform(test_processed)

    predictions = model.predict(test_scaled)
    test_df["Churn Prediction"] = predictions

    st.write("### Predictions")
    st.dataframe(test_df)
