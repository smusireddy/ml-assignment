# 📊 Project Title  
**Telco Customer Churn Prediction – Streamlit App**

---

## 📝 Problem Statement
Telecommunication companies lose significant revenue when customers discontinue their services.  
The objective of this project is to build a machine learning solution that:

- Analyzes customer behavior  
- Predicts whether a customer will churn  
- Compares multiple ML models  
- Provides an interactive Streamlit interface for real‑time predictions  

---

## 📚 Dataset Description
The dataset contains customer demographics, account information, subscribed services, and churn labels.

### **Key Feature Categories**
- **Demographics:** gender, senior citizen, partner, dependents  
- **Account Info:** tenure, contract type, payment method, monthly charges, total charges  
- **Services:** phone service, internet service, online security, streaming services  
- **Target Variable:**  
  - **Churn = Yes (1)** → Customer left  
  - **Churn = No (0)** → Customer stayed  

**Dataset Source:**
https://raw.githubusercontent.com/smusireddy/datasetfiles/refs/heads/main/WA_Fn-UseC_-Telco-Customer-Churn.csv

**Copied From:**
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

### **🔹 Machine Learning Models Used**
- Logistic Regression  
- Decision Tree Classifier  
- K‑Nearest Neighbor  
- Naive Bayes (Gaussian)  
- Random Forest  
- XGBoost  

### **🔹 Evaluation Metrics**
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- MCC Score  


## 📊 Model Performance Comparison

| Model                           | Accuracy  | Precision | Recall    | F1 Score  | MCC Score |
|---------------------------------|-----------|-----------|-----------|-----------|-----------|
| **Logistic Regression**         | 0.784244  | 0.721519  | 0.304813  | 0.428571  | 0.367070  |
| **Decision Tree Classifier**    | 0.774308  | 0.589172  | 0.494652  | 0.537791  | 0.392603  |
| **KNN Classifier**              | 0.760823  | 0.573705  | 0.385027  | 0.460800  | 0.325025  |
| **Naive Bayes (Gaussian)**      | 0.614620  | 0.400236  | 0.906417  | 0.555283  | 0.374772  |
| **Random Forest (Ensemble)**    | 0.796309  | 0.662921  | 0.473262  | 0.552262  | 0.435259  |
| **XGBoost (Ensemble)**          | 0.784954  | 0.607903  | 0.534759  | 0.568990  | 0.428064  |


## 📌 Model Performance Analysis

| **ML Model**                        | **Observation About Performance** |
|------------------------------------|-----------------------------------|
| **Logistic Regression**            | Good overall accuracy; strong precision but low recall, meaning it predicts churners conservatively and misses many actual churn cases. |
| **Decision Tree Classifier**       | Balanced performance; better recall than Logistic Regression but slightly less stable due to overfitting tendencies. |
| **K‑Nearest Neighbor Classifier**  | Moderate accuracy; struggles with recall, indicating difficulty identifying churners in high‑dimensional encoded data. |
| **Naive Bayes (Gaussian)**         | Lowest accuracy but **highest recall**, meaning it catches most churners but produces many false positives. Useful when missing churners is costly. |
| **Random Forest (Ensemble)**       | **Best overall performer; highest accuracy and strong balance across metrics, showing robustness and good generalization.** |
| **XGBoost (Ensemble)**             | Competitive performance; strong F1 and MCC scores, indicating reliable predictions and good handling of complex patterns. |

