# Churn Prediction Using XGBoost

This project implements a customer churn prediction model using XGBoost. It handles imbalanced datasets with SMOTE oversampling and optimizes classification threshold for better F1-score performance.

---

## **Dataset**

- The dataset used contains customer information and churn status.
- Columns include both numeric and categorical features such as `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`, etc.
- The target variable is `Churn` (Yes/No).

---

## **Project Workflow**

1. **Load & Clean Data**
    - Missing values are replaced and numeric columns are converted properly.
    - The `customerID` column is removed as it is not useful for prediction.

2. **Feature Selection**
    - Numeric features with low correlation to the target (`Churn`) are dropped.
    - Categorical features are one-hot encoded.

3. **Train-Test Split**
    - Dataset is split into train and test sets (80:20) with stratification to maintain class balance.

4. **Handle Class Imbalance**
    - SMOTE (Synthetic Minority Oversampling Technique) is applied to balance the classes in the training set.

5. **Feature Scaling**
    - StandardScaler is used to scale numeric features.

6. **Model Training**
    - XGBoost classifier is trained with hyperparameters:
      ```python
      n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85
      ```

7. **Threshold Optimization**
    - Optimal probability threshold is determined to maximize F1-score.

8. **Evaluation**
    - Final model performance is measured using F1-score, ROC AUC, and confusion matrix.

---
