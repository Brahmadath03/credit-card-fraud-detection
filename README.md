## Credit Card Fraud Detection (Machine Learning Project)

# Project Overview

This project aims to detect fraudulent credit card transactions using machine learning techniques. The goal is to handle imbalanced data and build a model that effectively identifies fraud cases.

# Dataset

- Synthetic credit card fraud dataset (10,000 transactions)
- Target variable: is_fraud
- Includes features like:
- transaction amount
- transaction hour
- merchant category
- device trust score
- velocity (last 24 hours)

# ⚙️ Tech Stack

- Python
- Pandas
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib
- Seaborn
- XGBOOST

# Machine Learning Workflow

- Data Cleaning
- One-hot encoding for categorical features
- Feature scaling
- Train-test split (Stratified)
- Handling class imbalance using SMOTE

# Model Training:
- Logistic Regression
- Random Forest

# Model Evaluation:
- Confusion Matrix
- Classification Report
- ROC-AUC Score
- Hyperparameter tuning using GridSearchCV

# Model Performance

Evaluated using:

- Precision
- Recall
- F1-score
- ROC-AUC
- Random Forest performed better after hyperparameter tuning.

# Why Recall is Important?

In fraud detection, missing a fraudulent transaction (False Negative) is more costly than a false alarm. Therefore, recall was prioritized during model optimization.

# What I Learned?

Handling imbalanced datasets using SMOTE

Importance of evaluation metrics beyond accuracy

# Hyperparameter tuning

Difference between probability-based ROC-AUC and label-based metrics

# How to Run?

- pip install -r requirements.txt
- python main.py

## 📊 Model Evaluation
- ROC-AUC: 0.XX
- PR-AUC: 0.XX
- Threshold tuning implemented

# Future Improvements

- Add model deployment using Flask
- Add feature importance visualization
- Add real-time fraud prediction simulation

# 🚀 Features
- Imbalanced dataset handling
- Threshold tuning
- ROC-AUC & PR-AUC evaluation
- Model comparison
- SHAP explainability (Planned)

# 📁 Project Structure

credit-card-fraud-detection/
│
├── data/
├── notebooks/
├── main.py
├── predict.py
├── models/
├── README.md
├── requirements.txt
└── .gitignore



