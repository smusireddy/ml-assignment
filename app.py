import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ---------------------------------------------------------
# Streamlit Title
# ---------------------------------------------------------
st.title("📊 Telco Customer Churn Prediction – Model Comparison App")
st.write("This app trains six ML models and compares their performance on the Telco Churn dataset.")

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/smusireddy/datasetfiles/refs/heads/main/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(url)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X = pd.get_dummies(X, drop_first=True)

    return df, X, y

df, X, y = load_data()

# ---------------------------------------------------------
# Train-test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling for LR and KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# Initialize models
# ---------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "K-Nearest Neighbor Classifier": KNeighborsClassifier(),
    "Naive Bayes Classifier - Gaussian": GaussianNB(),
    "Ensemble Model - Random Forest": RandomForestClassifier(),
    "Ensemble Model - XGBoost": XGBClassifier(
        eval_metric="logloss",
        tree_method="hist",
        random_state=42
    )
}

# ---------------------------------------------------------
# Train & evaluate models
# ---------------------------------------------------------
results = []

for name, model in models.items():
    if name in ["Logistic Regression", "K-Nearest Neighbor Classifier"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC Score": matthews_corrcoef(y_test, y_pred)
    })

results_df = pd.DataFrame(results)

# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------
st.subheader("📈 Model Performance Comparison")
st.dataframe(results_df)

# ---------------------------------------------------------
# Model selection for prediction
# ---------------------------------------------------------
st.subheader("🔮 Predict Churn for a New Customer")

model_choice = st.selectbox("Choose a model for prediction", list(models.keys()))
selected_model = models[model_choice]

# Build input form dynamically
input_data = {}

for col in X.columns:
    if X[col].dtype == "uint8":
        input_data[col] = st.selectbox(col, [0, 1])
    else:
        input_data[col] = st.number_input(col, value=float(X[col].mean()))

input_df = pd.DataFrame([input_data])

# Scale if needed
if model_choice in ["Logistic Regression", "K-Nearest Neighbor Classifier"]:
    input_df = scaler.transform(input_df)

if st.button("Predict Churn"):
    prediction = selected_model.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠️ The customer is likely to CHURN")
    else:
        st.success("✅ The customer is NOT likely to churn")

