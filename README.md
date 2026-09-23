# 📊 Customer Churn Prediction Using Artificial Neural Network (ANN)

## 📌 Project Overview

Customer churn refers to the situation where a customer stops using a company's products or services.

This project uses an **Artificial Neural Network (ANN)** to predict whether a bank customer is likely to churn based on customer demographic, financial, and account-related information.

The project includes:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Data scaling
* ANN model development
* Model evaluation
* Model serialization using Joblib
* Streamlit web application
* Deployment using Streamlit Community Cloud

---

## 🎯 Project Objective

The main objective of this project is to develop a machine learning/deep learning model that can predict customer churn.

The model predicts:

```text
0 → Customer is likely to stay
1 → Customer is likely to churn
```

This type of prediction can help businesses identify customers who may leave and take appropriate retention actions.

---

## 📂 Dataset

The project uses a bank customer churn dataset.

### Important Features

| Feature         | Description                          |
| --------------- | ------------------------------------ |
| CreditScore     | Customer's credit score              |
| Geography       | Customer's country                   |
| Gender          | Customer's gender                    |
| Age             | Customer's age                       |
| Tenure          | Number of years with the bank        |
| Balance         | Customer's account balance           |
| NumOfProducts   | Number of bank products used         |
| HasCrCard       | Whether customer has a credit card   |
| IsActiveMember  | Whether customer is an active member |
| EstimatedSalary | Estimated customer salary            |
| Exited          | Churn/target variable                |

The following columns were not used as predictive features:

```text
RowNumber
CustomerId
Surname
```

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Removed unnecessary columns.
2. Checked for missing values.
3. Checked duplicate records.
4. Converted categorical variables into numerical form.
5. Applied one-hot encoding to `Geography`.
6. Applied binary encoding to `Gender`.
7. Separated independent and dependent variables.
8. Split the data into training and testing sets.
9. Applied `StandardScaler` to numerical input features.

### Final Model Features

```text
CreditScore
Age
Tenure
Balance
NumOfProducts
HasCrCard
IsActiveMember
EstimatedSalary
Geography_Germany
Geography_Spain
Gender_Male
```

---

## 🧠 Artificial Neural Network

The project uses `MLPClassifier` from Scikit-learn to implement an Artificial Neural Network.

###
