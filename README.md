# task-9

AI & ML Internship – Task 9
Random Forest – Credit Card Fraud Detection

1. Introduction
Credit card fraud detection is a classic imbalanced classification problem, where fraudulent transactions are very rare compared to normal ones.
In this task, we use Random Forest, an ensemble learning technique, to detect fraudulent transactions and compare its performance with a Logistic Regression baseline model.
________________________________________
2. Tools Used
•	Python
•	Pandas
•	NumPy
•	Scikit-learn
•	Matplotlib
•	Joblib
________________________________________
3. Dataset
•	Primary Dataset: Kaggle Credit Card Fraud Dataset
•	Target column: Class
o	0 → Normal transaction
o	1 → Fraudulent transaction
________________________________________
4. Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import joblib
________________________________________
5. Load Dataset
df = pd.read_csv("creditcard.csv")
df.head()
________________________________________
6. Check Class Imbalance
df['Class'].value_counts()
Observation:
The dataset is highly imbalanced, with very few fraud cases compared to non-fraud cases.
________________________________________
7. Feature and Target Separation
X = df.drop('Class', axis=1)
y = df['Class']
________________________________________
8. Train-Test Split (Stratified Sampling)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
Reason:
Stratified sampling ensures that the fraud ratio remains the same in both training and testing datasets.
________________________________________
9. Baseline Model – Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
print(classification_report(y_test, y_pred_lr))
Evaluation Metrics Used:
•	Precision
•	Recall
•	F1-score
(Accuracy is avoided because it is misleading for imbalanced datasets.)
________________________________________
10. Random Forest Model
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))
________________________________________
11. Feature Importance
importances = rf.feature_importances_
features = X.columns

plt.figure(figsize=(10,6))
plt.barh(features, importances)
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance - Random Forest")
plt.show()
Purpose:
To identify which features contribute most to fraud detection.
________________________________________
12. Model Comparison
•	Logistic Regression serves as a baseline
•	Random Forest performs better due to:
o	Ensemble learning
o	Handling non-linearity
o	Better performance on imbalanced data
________________________________________
13. Save the Trained Model
joblib.dump(rf, "random_forest_fraud_model.pkl")
________________________________________
14. Final Outcome
•	Learned ensemble learning using Random Forest
•	Understood imbalanced data handling
•	Evaluated models using precision, recall, and F1-score
•	Identified key fraud indicators using feature importance

15. Interview Questions (Short Answers)
1. Why is accuracy misleading in fraud detection?
Because the dataset is imbalanced, a model can achieve high accuracy by predicting all transactions as non-fraud.
2. What is Random Forest?
An ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy.
3. What is ensemble learning?
A technique where multiple models are combined to produce better results than a single model.
4. What is n_estimators?
The number of decision trees used in the Random Forest.
5. What is SMOTE?
A technique used to handle imbalanced datasets by generating synthetic samples of the minority class.


