import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and preprocessing files
model = joblib.load("customer_churn_ann.pkl")
scaler = joblib.load("customer_churn_scaler.pkl")
features = joblib.load("customer_churn_features.pkl")

# Load model evaluation metrics
metrics = joblib.load("customer_churn_metrics.pkl")

# Load evaluation data
evaluation = joblib.load("customer_churn_evaluation.pkl")

y_test = evaluation["y_test"]
y_pred = evaluation["y_pred"]
y_prob = evaluation["y_prob"]

#
import streamlit as st
import pandas as pd
import joblib

# ==============================
# Load Model and Preprocessing
# ==============================

model = joblib.load("customer_churn_ann.pkl")
scaler = joblib.load("customer_churn_scaler.pkl")
features = joblib.load("customer_churn_features.pkl")


# ==============================
# Page Configuration
# ==============================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)


# ==============================
# Title
# ==============================

st.title("📊 Customer Churn Prediction")

st.write(
    "Predict whether a bank customer is likely to churn "
    "using an Artificial Neural Network (ANN)."
)

st.divider()


# ==============================
# Customer Information
# ==============================

st.subheader("👤 Customer Information")

col1, col2 = st.columns(2)

with col1:

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    tenure = st.number_input(
        "Tenure",
        min_value=0,
        max_value=10,
        value=5
    )

    balance = st.number_input(
        "Balance",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

with col2:

    num_products = st.number_input(
        "Number of Products",
        min_value=1,
        max_value=4,
        value=1
    )

    has_card = st.selectbox(
        "Has Credit Card?",
        ["Yes", "No"]
    )

    active_member = st.selectbox(
        "Is Active Member?",
        ["Yes", "No"]
    )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )


# ==============================
# Categorical Information
# ==============================

st.subheader("🌍 Customer Details")

col3, col4 = st.columns(2)

with col3:

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

with col4:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )


st.divider()


# ==============================
# Prediction
# ==============================

if st.button(
    "🔮 Predict Churn",
    use_container_width=True
):

    # --------------------------
    # Convert inputs
    # --------------------------

    has_card_value = 1 if has_card == "Yes" else 0

    active_member_value = 1 if active_member == "Yes" else 0

    gender_male = 1 if gender == "Male" else 0

    geography_germany = (
        1 if geography == "Germany" else 0
    )

    geography_spain = (
        1 if geography == "Spain" else 0
    )


    # --------------------------
    # Create input dataframe
    # --------------------------

    input_data = {

        "CreditScore": credit_score,

        "Age": age,

        "Tenure": tenure,

        "Balance": balance,

        "NumOfProducts": num_products,

        "HasCrCard": has_card_value,

        "IsActiveMember": active_member_value,

        "EstimatedSalary": estimated_salary,

        "Geography_Germany": geography_germany,

        "Geography_Spain": geography_spain,

        "Gender_Male": gender_male
    }


    input_df = pd.DataFrame([input_data])


    # --------------------------
    # Arrange columns correctly
    # --------------------------

    input_df = input_df[features]


    # --------------------------
    # Scale data
    # --------------------------

    input_scaled = scaler.transform(input_df)


    # --------------------------
    # Prediction
    # --------------------------

    prediction = model.predict(input_scaled)[0]

    probabilities = model.predict_proba(input_scaled)[0]

    churn_probability = probabilities[1] * 100


    # ==========================
    # Display Result
    # ==========================

    st.subheader("📈 Prediction Result")

    if prediction == 1:

        st.error(
            "⚠️ Customer is likely to churn."
        )

    else:

        st.success(
            "✅ Customer is unlikely to churn."
        )


    st.metric(
        "Churn Probability",
        f"{churn_probability:.2f}%"
    )


    # --------------------------
    # Probability bar
    # --------------------------

    st.progress(
        int(churn_probability)
    )

    st.caption(
        "Probability represents the ANN model's estimated likelihood "
        "that the customer will churn."
    )
    st.divider()

st.subheader("📊 Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Accuracy",
    f"{metrics['Accuracy'] * 100:.2f}%"
)

col2.metric(
    "Precision",
    f"{metrics['Precision'] * 100:.2f}%"
)

col3.metric(
    "Recall",
    f"{metrics['Recall'] * 100:.2f}%"
)

col4.metric(
    "F1 Score",
    f"{metrics['F1 Score'] * 100:.2f}%"
)

col5.metric(
    "ROC-AUC",
    f"{metrics['ROC-AUC']:.3f}"
)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

st.divider()

st.subheader("📌 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

ax.imshow(cm)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("ANN Confusion Matrix")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["No Churn", "Churn"])
ax.set_yticklabels(["No Churn", "Churn"])

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

st.pyplot(fig)

from sklearn.metrics import roc_curve, auc

st.subheader("📈 ROC Curve")

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

roc_auc = auc(fpr, tpr)

fig2, ax2 = plt.subplots()

ax2.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.3f}"
)

ax2.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve - ANN")
ax2.legend()

st.pyplot(fig2)