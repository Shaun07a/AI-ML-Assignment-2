# Customer Churn Prediction using Logistic Regression

## Objective

The objective of this project is to build a Logistic Regression model that predicts whether a customer is likely to leave a telecommunications company based on demographic information and service usage. The project demonstrates the complete machine learning workflow including data preprocessing, model training, prediction, and evaluation.

---

## Dataset Link

Telco Customer Churn Dataset

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

## Libraries Used

- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## Methodology

1. Load the Telco Customer Churn dataset using Pandas.
2. Perform data understanding by identifying numerical features, categorical features, and the target variable.
3. Check for missing values and handle missing entries in the `TotalCharges` column.
4. Remove the `customerID` column as it is not useful for prediction.
5. Encode categorical variables using Label Encoding.
6. Split the dataset into 80% training and 20% testing sets.
7. Train a Logistic Regression model.
8. Predict customer churn for the testing dataset.
9. Evaluate the model using:
   - Accuracy Score
   - Precision
   - Recall
   - F1-Score
10. Generate and visualize the Confusion Matrix.

---

## Results

The Logistic Regression model successfully classified customer churn with good predictive performance. The evaluation metrics indicate that the model can distinguish between churn and non-churn customers effectively. The confusion matrix shows that the majority of customer records are classified correctly, demonstrating the usefulness of Logistic Regression for binary classification problems.

---

## Conclusion

This project demonstrates the application of Logistic Regression for customer churn prediction. Customer characteristics such as tenure, contract type, monthly charges, internet services, and payment methods influence the likelihood of churn. Although the model provides reliable performance, Logistic Regression assumes a linear relationship between the features and the target variable. More advanced classification algorithms can be explored to capture more complex relationships and further improve prediction accuracy.
