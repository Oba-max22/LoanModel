import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the model
model = joblib.load('loan_model.pkl')

# 2. App Header
st.title("Smart Loan Predictor")
st.write("This app uses Machine Learning to evaluate loan eligibility based on financial ratios.")

# 3. Sidebar for Personal Info
with st.sidebar:
    st.header("Applicant Details")
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])

# 4. Main Panel for Financial Info
st.header("Financial Information")
col1, col2 = st.columns(2)

with col1:
    applicant_income = st.number_input("Applicant Income ($)", value=5000)
    coapplicant_income = st.number_input("Co-Applicant Income ($)", value=0)
    loan_amount = st.number_input("Loan Amount ($)", value=100)

with col2:
    loan_term = st.selectbox("Loan Term (Months)", [360, 180, 120, 60, 480])
    credit_history = st.selectbox("Credit History", ["Good (1.0)", "Bad (0.0)"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# 5. Preprocessing (Connecting inputs to the Model's brain)
if st.button("Predict Status"):
    
    # A. Feature Engineering: Calculate the Critical Ratio
    total_income = applicant_income + coapplicant_income
    if total_income == 0:
        debt_ratio = 0
    else:
        debt_ratio = loan_amount / total_income
        
    st.info(f"Calculated Debt-to-Income Ratio: {debt_ratio:.4f}")

    # B. Encoding (Must match your training data EXACTLY)
    # 1=Male, 0=Female | 1=Yes, 0=No | Graduate=0, Not=1
    row = {
        'Gender': 1 if gender == "Male" else 0,
        'Married': 1 if married == "Yes" else 0,
        'Dependents': 3 if dependents == "3+" else int(dependents),
        'Education': 0 if education == "Graduate" else 1,
        'Self_Employed': 1 if self_employed == "Yes" else 0,
        'Loan_Amount_Term': loan_term,
        'Credit_History': 1.0 if "Good" in credit_history else 0.0,
        'Property_Area': 2 if property_area == "Urban" else (1 if property_area == "Semiurban" else 0),
        'Debt_Income_Ratio': debt_ratio  # <--- The new secret weapon
    }

    # C. Create DataFrame
    df_input = pd.DataFrame([row])

    # D. Predict
    prediction = model.predict(df_input)[0]

    # E. Display Result
    if prediction == 1:
        st.success("**APPROVED**: You meet the criteria for this loan.")
        st.balloons()
    else:
        st.error("**REJECTED**: The model has flagged this as high risk.")
