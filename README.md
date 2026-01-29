📊 Telco Customer Churn Prediction – Streamlit App
https://img.shields.io/badge/Status-Active-brightgreen
https://img.shields.io/badge/Python-3.10-blue
https://img.shields.io/badge/Streamlit-App-red
https://img.shields.io/badge/License-MIT-lightgrey

A machine learning web application that predicts customer churn using the Telco Customer Churn Dataset.
The app compares six ML models, evaluates their performance, and allows users to make predictions through an interactive Streamlit interface.

📌 Project Overview
Customer churn is a major challenge for telecom companies.
This project builds a complete ML pipeline to:

Load and preprocess the Telco dataset

Train six different machine learning models

Evaluate them using multiple metrics

Deploy an interactive Streamlit app for real‑time predictions

🚀 Features
🔍 Model Training & Evaluation
The app trains and compares the following models:

Logistic Regression

Decision Tree Classifier

K‑Nearest Neighbor Classifier

Naive Bayes Classifier (Gaussian)

Random Forest (Ensemble)

XGBoost (Ensemble)

📈 Evaluation Metrics
Each model is evaluated using:

Accuracy

Precision

Recall

F1 Score

MCC Score

🧮 Interactive Prediction
Users can:

Select a model

Enter customer details

Predict whether the customer will churn

🖼️ App Preview
(Optional: Add screenshots here once your app is running)

Code
📌 Example:
![App Screenshot](images/app_screenshot.png)
📁 Project Structure
Code
├── app.py               # Streamlit application
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
📦 Installation & Setup
1. Clone the repository
Code
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
2. Create and activate an Anaconda environment
Code
conda create -n churnapp python=3.10
conda activate churnapp
3. Install dependencies
Code
pip install -r requirements.txt
▶️ Run the Streamlit App
Code
streamlit run app.py
This will automatically open the app in your browser.

🌐 Deploying on Streamlit Cloud (Optional)
Push your project to GitHub

Go to: https://share.streamlit.io

Click “Deploy App”

Select your repo and choose app.py

Deploy

Streamlit Cloud will install dependencies and host your app online.

📊 Dataset Information
Dataset Source:

Code
https://raw.githubusercontent.com/smusireddy/datasetfiles/refs/heads/main/WA_Fn-UseC_-Telco-Customer-Churn.csv
Includes:

Customer demographics

Account information

Service usage

Churn labels

🧠 Modeling Approach
Missing values handled

Categorical variables one‑hot encoded

Numerical features scaled for LR & KNN

Train/test split with stratification

Ensemble models used for improved performance

👨‍💻 Author
Suresh  
Identity & Access Management Engineer | Machine Learning Enthusiast
Focused on building clean, reproducible ML workflows.

📜 License
This project is licensed under the MIT License.
