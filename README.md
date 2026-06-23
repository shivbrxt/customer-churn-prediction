# 📡 Telecom Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green?logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-ff4b4b?logo=streamlit)

---

# 📖 Project Overview

Customer churn is one of the biggest challenges faced by telecom companies. Losing existing customers directly impacts revenue and increases customer acquisition costs.

This project uses Machine Learning techniques to identify customers who are likely to leave the service, allowing businesses to take proactive retention actions.

The project includes:

✅ Data Analysis (DA)

✅ Exploratory Data Analysis (EDA)

✅ Feature Engineering

✅ Logistic Regression Model

✅ Random Forest Model

✅ XGBoost Model

✅ Model Comparison

✅ Business Recommendations

✅ Streamlit Deployment

---

# 🎯 Business Problem

Telecom providers lose a significant amount of revenue due to customer attrition.

The objective of this project is to:

- Predict customer churn
- Identify key churn drivers
- Generate actionable business insights
- Support customer retention strategies

---

# 📊 Dataset

Dataset: Telco Customer Churn Dataset

Number of Customers: ~7,000

Target Variable:

- Churn
  - Yes → Customer left
  - No → Customer retained

### Features

#### Customer Information

- Gender
- SeniorCitizen
- Partner
- Dependents

#### Service Information

- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies

#### Account Information

- Contract
- PaperlessBilling
- PaymentMethod
- Tenure
- MonthlyCharges
- TotalCharges

---

# 🛠️ Project Workflow

```text
Business Problem
        ↓
Data Collection
        ↓
Data Analysis
        ↓
Exploratory Data Analysis
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Data Preprocessing
        ↓
Model Building
        ↓
Model Evaluation
        ↓
Business Recommendations
        ↓
Deployment
```

---

# 📂 Repository Structure

```text
customer-churn-prediction/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── notebooks/
│   ├── 01_Data_Analysis.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Logistic_Regression.ipynb
│   ├── 04_Random_Forest.ipynb
│   └── 05_XGBoost.ipynb
│
├── models/
│   ├── telecom_customer_churn_model_LR.pkl
│   ├── telecom_customer_churn_model_RF.pkl
│   └── telecom_customer_churn_model_XGB.pkl
│
├── app/
│   └── churn_app.py
│
├── images/
│
├── requirements.txt
│
├── README.md
│
└── LICENSE
```

---

# 🔍 Exploratory Data Analysis

The EDA focused on:

### Customer Demographics

- Gender Distribution
- Senior Citizen Analysis
- Partner & Dependents Analysis

### Service Usage

- Internet Service Distribution
- Streaming Services Usage
- Security Services Usage

### Customer Retention Patterns

- Churn vs Contract Type
- Churn vs Payment Method
- Churn vs Tenure
- Churn vs Monthly Charges

### Correlation Analysis

- Pearson Correlation
- Spearman Correlation
- Kendall Correlation

---

# 🤖 Machine Learning Models

Three classification models were developed and compared.

## 1. Logistic Regression

Advantages:

- Interpretable
- Fast Training
- Strong Baseline Model

---

## 2. Random Forest

Advantages:

- Handles Non-linearity
- Robust to Noise
- Feature Importance Analysis

---

## 3. XGBoost

Advantages:

- High Predictive Performance
- Handles Class Imbalance
- State-of-the-Art Gradient Boosting

---

# 📈 Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Precision-Recall Curve
- Confusion Matrix

---

# 🏆 Best Model

### XGBoost Classifier

Reasons:

- Highest ROC-AUC
- Better Recall
- Better Churn Detection
- Strong Generalization

---

# 💡 Key Business Insights

### Insight 1

Customers with Month-to-Month contracts have significantly higher churn rates.

### Recommendation

Offer discounts for annual and two-year contracts.

---

### Insight 2

Customers with short tenure are more likely to churn.

### Recommendation

Implement onboarding and loyalty programs during the first year.

---

### Insight 3

Customers without Online Security and Tech Support churn more frequently.

### Recommendation

Bundle security and support services into premium plans.

---

### Insight 4

Customers with higher monthly charges are more likely to churn.

### Recommendation

Provide personalized retention offers and pricing incentives.

---

# 🚀 Streamlit Deployment

Interactive web application built using Streamlit.

### Features

- Customer churn prediction
- Real-time probability score
- Risk classification
- Business retention recommendations
- Interactive user interface

Run locally:

https://customer-churn-prediction-dex.streamlit.app/

---

# ⚙️ Installation

Clone Repository

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
```

Move to Project Folder

```bash
cd customer-churn-prediction
```

Create Virtual Environment

```bash
python -m venv venv
```

Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

Main libraries used:

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- Streamlit
- Joblib

---

# 📸 Screenshots

- EDA Visualizations
  
- Model Performance

- Churn Prediction Results

- Streamlit Dashboard
- 
<img width="904" height="434" alt="Screenshot 2026-06-24 023523" src="https://github.com/user-attachments/assets/18374261-8b5c-41b7-a57f-2c9e675f79f6" />


---

# 🔮 Future Improvements

- Hyperparameter Optimization
- SHAP Explainability
- Customer Segmentation
- Model Monitoring
- Cloud Deployment (AWS/GCP)
- CI/CD Pipeline

---

# 👨‍💻 Author

**Shivbrat Singh**

MBA (Marketing) | Data Analytics Enthusiast

Skills:

- Python
- SQL
- Power BI
- Machine Learning
- AWS
- Ms-Excel

GitHub:
https://github.com/shivbrxt

LinkedIn:
linkedin.com/in/shivbrat-singh-a475a1382

---

# ⭐ If you found this project useful, consider giving it a star!
