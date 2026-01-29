📊 Telco Customer Churn Prediction – Streamlit App
This project builds and compares six machine learning models to predict customer churn using the Telco Customer Churn dataset.
The app is built with Streamlit and deployed using Anaconda or Streamlit Cloud.

🚀 Features
Loads the Telco dataset directly from GitHub

Cleans and preprocesses the data

Trains six ML models:

Logistic Regression

Decision Tree Classifier

K‑Nearest Neighbor Classifier

Naive Bayes Classifier (Gaussian)

Random Forest (Ensemble)

XGBoost (Ensemble)

Computes evaluation metrics:

Accuracy

Precision

Recall

F1 Score

MCC Score

Displays a comparison table

Allows users to select a model

Accepts customer input and predicts churn

📁 Project Structure
Code
├── app.py               # Streamlit application
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
📦 Installation
1. Clone the repository
Code
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
2. Create/activate your Anaconda environment
Code
conda create -n churnapp python=3.10
conda activate churnapp
3. Install dependencies
Code
pip install -r requirements.txt
▶️ Running the App
Run the Streamlit app using:

Code
streamlit run app.py
This will open the app in your browser.

📊 Dataset
The dataset is loaded from GitHub:

Code
https://raw.githubusercontent.com/smusireddy/datasetfiles/refs/heads/main/WA_Fn-UseC_-Telco-Customer-Churn.csv
It includes customer demographics, service usage, and churn labels.

🧠 Model Evaluation
Each model is evaluated using:

Accuracy

Precision

Recall

F1 Score

MCC Score

A comparison table is displayed inside the Streamlit app.

🌐 Deployment (Optional)
You can deploy this app on Streamlit Cloud:

Push your repo to GitHub

Go to https://share.streamlit.io

Select your repo and choose app.py

Deploy

👨‍💻 Author
Suresh  
Machine Learning & IAM Engineer
Passionate about building clean, reproducible ML workflows.
