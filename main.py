import matplotlib
matplotlib.use("Agg")
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve

def main():

# Load the dataset
    df = pd.read_csv("data/raw/credit_card_fraud_10k.csv")
    print(df.head())
    print(df['is_fraud'].value_counts())
    print("-----------------------------")

    #  Scale Amount
    print("Columns in the dataset:", df.columns)

        # Drop transaction_id (not useful for ML)
    df = df.drop("transaction_id", axis=1)

        # One-hot encode merchant_category (categorical feature)
    df = pd.get_dummies(df, columns=["merchant_category"], drop_first=True)

    # Basic Preprocessing
    scaler = StandardScaler()
    df['amount'] = scaler.fit_transform(df[['amount']])

    # Split the data into features and target
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']

    #Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training set shape:", X_train.shape)
    print("-----------------------------")
    print("Test set shape:", X_test.shape)
    print("-----------------------------")


    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print("After SMOTE shape:", X_train_res.shape, y_train_res.shape)
    print("-----------------------------")


    print("Before SMOTE:", y_train.value_counts())
    print("-----------------------------")
    print("After SMOTE:", y_train_res.value_counts())
    print("-----------------------------")

    # Train a Logistic Regression model
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_res, y_train_res)

    y_pred_lr = lr.predict(X_test)

    #evaluation
    print(classification_report(y_test, y_pred_lr))
    print("-----------------------------")
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    print("ROC-AUC:", roc_auc_score(y_test, y_prob_lr))
    print("-----------------------------")

    # Train a Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_res, y_train_res)

    y_pred_rf = rf.predict(X_test)

    # Evaluation
    print(classification_report(y_test, y_pred_rf))
    print("-----------------------------")
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))
    print("-----------------------------")

    # Confusion Matrix for Random Forest
    cm = confusion_matrix(y_test, y_pred_rf) # cm = confusion matrix

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Random Forest')
    plt.savefig("models/confusion_matrix.png")
    plt.close()

    # Hyperparameter Tuning for Random Forest(optimization)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring='recall',
        n_jobs=-1
    )

    grid.fit(X_train_res, y_train_res)

    print("Best Parameters:", grid.best_params_)
    print("-----------------------------")
    best_rf = grid.best_estimator_

    # ROC Curve for Best Random Forest
    y_prob_best = best_rf.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_prob_best)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])  # diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Random Forest")
    plt.savefig("models/roc_curve.png")
    plt.close()
    y_pred_best_rf = best_rf.predict(X_test)

    joblib.dump(rf, "fraud_model.pkl")
    joblib.dump(best_rf, "models/best_fraud_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    os.makedirs("models", exist_ok=True)
    print("Best model saved as best_fraud_model.pkl")
    print("File exists:", os.path.exists("models/best_fraud_model.pkl"))
    
    print("Saving model to:", os.getcwd())
    print("-----------------------------")

    print(classification_report(y_test, y_pred_best_rf))
    print("-----------------------------")
    y_prob_best = best_rf.predict_proba(X_test)[:, 1]
    print("ROC-AUC:", roc_auc_score(y_test, y_prob_best))

if __name__ == "__main__":
    main()




