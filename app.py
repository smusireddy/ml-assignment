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

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ---------------------------------------------------------
# Streamlit App Title
# ---------------------------------------------------------
st.title("📊 Telco Customer Churn Prediction – ML App")

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

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # One-hot encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    return df

df_processed = preprocess_data(df)

# ---------------------------------------------------------
# Train/Test Split
# ---------------------------------------------------------
X = df_processed.drop("Churn", axis=1)
y = df_processed["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# Download Test Dataset Only
# ---------------------------------------------------------
st.subheader("📥 Download Test Dataset")

# Combine X_test and y_test into one DataFrame
test_df = pd.DataFrame(X_test)
test_df['target'] = y_test.values

# Convert to CSV
test_csv = test_df.to_csv(index=False).encode("utf-8")

# Download button
st.download_button(
    label="⬇️ Download Test CSV",
    data=test_csv,
    file_name="test_dataset.csv",
    mime="text/csv"
)


# ---------------------------------------------------------
# ML Models
# ---------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "K-Nearest Neighbor Classifier": KNeighborsClassifier(),
    "Naive Bayes Classifier - Gaussian": GaussianNB(),
    "Ensemble Model - Random Forest": RandomForestClassifier(),
    "Ensemble Model - XGBoost": XGBClassifier(eval_metric="logloss")
}

# ---------------------------------------------------------
# Model Selection
# ---------------------------------------------------------
st.header("🤖 Select a Machine Learning Model")
model_choice = st.selectbox("Choose a model", list(models.keys()))

model = models[model_choice]
model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------
st.header("📊 Evaluation Metrics")

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "MCC Score"],
    "Score": [accuracy, precision, recall, f1, mcc]
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
# Clean Classification Report (Table Format)
# ---------------------------------------------------------
st.header("📄 Classification Report (Formatted)")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)


# ---------------------------------------------------------
# Prediction on Uploaded Test Data
# ---------------------------------------------------------
if uploaded_file is not None:
    st.header("🔍 Predictions on Uploaded Test Data")

    test_df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Test Data Preview")
    st.dataframe(test_df.head())

    # Preprocess uploaded test data
    test_processed = preprocess_data(test_df)

    # Align columns with training data
    missing_cols = set(X_train.columns) - set(test_processed.columns)
    for col in missing_cols:
        test_processed[col] = 0

    test_processed = test_processed[X_train.columns]

    # Scale
    test_scaled = scaler.transform(test_processed)

    # Predict
    predictions = model.predict(test_scaled)
    test_df["Churn Prediction"] = predictions

    st.write("### Predictions")
    st.dataframe(test_df)
