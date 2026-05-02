import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==========================================
# 1. LOAD ALL PRE-TRAINED MODELS
# ==========================================
@st.cache_resource
def load_model():
    with open('loan_model.pkl', 'rb') as file:
        nb, knn, lin_reg, scaler_clf, scaler_reg, le_property = pickle.load(file)
    return nb, knn, lin_reg, scaler_clf, scaler_reg, le_property

nb, knn, lin_reg, scaler_clf, scaler_reg, le_property = load_model()

# ==========================================
# 2. BUILD THE STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Loan Dashboard", page_icon="🏦", layout="centered")

st.title("🏦 Loan Comparison Dashboard")
st.markdown("This dashboard uses **Naive Bayes** and **KNN** to predict approval, and **Linear Regression** to suggest the ideal loan amount.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics & Education")
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    st.subheader("Financial Profile")
    applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0, value=0)
    loan_amount = st.number_input("Requested Loan (in thousands)", min_value=1.0, max_value=5000.0, value=150.0)
    loan_term = st.selectbox("Loan Term (Days)", [360.0, 180.0, 120.0, 84.0, 60.0])
    credit_history = st.selectbox("Credit History", ["1.0 (Good/Existing)", "0.0 (Bad/None)"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.divider()

# ==========================================
# 3. PREDICTION LOGIC
# ==========================================
if st.button("Run Model Analysis", type="primary", use_container_width=True):
    
    # 1. Translate UI text into numbers
    input_gender = 1 if gender == "Male" else 0
    input_married = 1 if married == "Yes" else 0
    input_dependents = 3 if dependents == "3+" else int(dependents)
    input_education = 1 if education == "Graduate" else 0
    input_self_employed = 1 if self_employed == "Yes" else 0
    input_credit = 1.0 if "1.0" in credit_history else 0.0
    input_property = le_property.transform([property_area])[0]

    total_income = applicant_income + coapplicant_income
    loan_income_ratio = (loan_amount * 1000) / total_income if total_income > 0 else 999999.0

    # 2. Build the DataFrame for Classification (NB & KNN)
    user_data_clf = pd.DataFrame([[
        input_credit, input_dependents, input_education, input_gender, 
        loan_term, loan_amount, input_married, input_property, 
        input_self_employed, applicant_income, coapplicant_income,
        total_income, loan_income_ratio
    ]], columns=[
        "Credit_History", "Dependents", "Education", "Gender", "Loan_Amount_Term", 
        "LoanAmount", "Married", "Property_Area", "Self_Employed", 
        "ApplicantIncome", "CoapplicantIncome", "Total_Income", "Loan_Income_Ratio"
    ])

    # 3. Build the DataFrame for Regression (Linear) - Excludes LoanAmount
    user_data_reg = pd.DataFrame([[
        input_credit, input_dependents, input_education, input_gender, 
        loan_term, input_married, input_property, input_self_employed, 
        applicant_income, coapplicant_income, total_income
    ]], columns=[
        "Credit_History", "Dependents", "Education", "Gender", "Loan_Amount_Term", 
        "Married", "Property_Area", "Self_Employed", "ApplicantIncome", 
        "CoapplicantIncome", "Total_Income"
    ])

    # 4. Scale and Predict
    clf_scaled = scaler_clf.transform(user_data_clf)
    reg_scaled = scaler_reg.transform(user_data_reg)

    pred_nb = nb.predict(clf_scaled)[0]
    prob_nb = nb.predict_proba(clf_scaled)[0][1]

    pred_knn = knn.predict(clf_scaled)[0]
    prob_knn = knn.predict_proba(clf_scaled)[0][1]

    predicted_amount = lin_reg.predict(reg_scaled)[0]

    # 5. Display the Multi-Model Output
    st.subheader("Model Comparison Results:")
    
    res_col1, res_col2 = st.columns(2)
    
    # Naive Bayes Output
    with res_col1:
        st.write("### 🧮 Naive Bayes")
        if pred_nb == 1:
            st.success(f"**APPROVED** ({prob_nb * 100:.1f}%)")
        else:
            st.error(f"**DENIED** ({(1 - prob_nb) * 100:.1f}%)")
            
    # KNN Output
    with res_col2:
        st.write("### 📍 K-Nearest Neighbors")
        if pred_knn == 1:
            st.success(f"**APPROVED** ({prob_knn * 100:.1f}%)")
        else:
            st.error(f"**DENIED** ({(1 - prob_knn) * 100:.1f}%)")

    st.divider()
    
    # Linear Regression Output
    st.write("### 📈 Linear Regression Insight")
    st.info(f"Based on historical data for similar financial profiles, the algorithm suggests a safe loan amount of **${predicted_amount * 1000:,.2f}**.")
    
    if (loan_amount * 1000) > (predicted_amount * 1000):
        st.warning(f"⚠️ The requested amount (${loan_amount * 1000:,.0f}) is higher than the mathematically suggested amount.")