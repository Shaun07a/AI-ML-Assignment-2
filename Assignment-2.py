# ==========================================
# AI-ML Assignment 2
# Customer Churn Prediction using Logistic Regression
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ----------------------------------------
# Task 1 : Data Understanding
# ----------------------------------------

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("First Five Records\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nNumerical Features:")
print(df.select_dtypes(include=["number"]).columns.tolist())

print("\nCategorical Features:")
print(df.select_dtypes(include=["object", "string"]).columns.tolist())

print("\nTarget Variable:")
print("Churn")

# ----------------------------------------
# Task 2 : Data Preprocessing
# ----------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# Remove customerID (not useful for prediction)
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values with median
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Encode all categorical columns
encoder = LabelEncoder()

categorical_cols = df.select_dtypes(include=["object", "string"]).columns

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col].astype(str))

print("\nData Types After Encoding:\n")
print(df.dtypes)

# Features and Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

# ----------------------------------------
# Task 3 : Model Development
# ----------------------------------------

model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ----------------------------------------
# Task 4 : Model Evaluation
# ----------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n========== Model Evaluation ==========\n")
print(f"Accuracy Score : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1-Score       : {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.show()