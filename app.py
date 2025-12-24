import gradio as gr
import joblib
import pandas as pd
import numpy as np

# 1. Load the model
# Ensure 'loan_model.pkl' is in the same folder as this script
model = joblib.load('loan_model.pkl')

# 2. Define the Prediction Function
def predict_loan(gender, married, dependents, education, self_employed, 
                 applicant_income, coapplicant_income, loan_amount, 
                 loan_term, credit_history, property_area):
    
    # --- A. Feature Engineering: Calculate Ratio ---
    total_income = applicant_income + coapplicant_income
    if total_income == 0:
        debt_ratio = 0
    else:
        debt_ratio = loan_amount / total_income
    
    # --- B. Create Data Row (Must match training columns) ---
    # Note: I included all columns (ApplicantIncome, etc.) just in case your 
    # model expects them, even if it primarily uses the Ratio.
    row = {
        'Gender': 1 if gender == "Male" else 0,
        'Married': 1 if married == "Yes" else 0,
        'Dependents': 3 if dependents == "3+" else int(dependents),
        'Education': 0 if education == "Graduate" else 1,
        'Self_Employed': 1 if self_employed == "Yes" else 0,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': float(loan_term),
        'Credit_History': 1.0 if "Good" in credit_history else 0.0,
        'Property_Area': 2 if property_area == "Urban" else (1 if property_area == "Semiurban" else 0),
        'TotalIncome': total_income,
        'Debt_Income_Ratio': debt_ratio 
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([row])
    
    # --- C. Predict ---
    try:
        prediction = model.predict(df)[0]
        
        # Format the output message
        result_msg = "APPROVED: You meet the criteria." if prediction == 1 else "REJECTED: High risk flagged."
        return result_msg
        
    except Exception as e:
        return f"Error: {str(e)}\n(Check if your model expects fewer columns!)"

# 3. Define the Interface Inputs
inputs = [
    gr.Dropdown(["Male", "Female"], label="Gender"),
    gr.Dropdown(["Yes", "No"], label="Married"),
    gr.Dropdown(["0", "1", "2", "3+"], label="Dependents"),
    gr.Dropdown(["Graduate", "Not Graduate"], label="Education"),
    gr.Dropdown(["Yes", "No"], label="Self Employed"),
    gr.Number(label="Applicant Income ($)", value=5000),
    gr.Number(label="Co-Applicant Income ($)", value=0),
    gr.Number(label="Loan Amount ($)", value=100),
    gr.Dropdown([360, 180, 120, 60, 480], label="Loan Term (Months)", value=360),
    gr.Dropdown(["Good (1.0)", "Bad (0.0)"], label="Credit History"),
    gr.Dropdown(["Urban", "Semiurban", "Rural"], label="Property Area")
]

# 4. Launch
iface = gr.Interface(
    fn=predict_loan,
    inputs=inputs,
    outputs="text",
    title="Loan Eligibility Checker",
    description="This app uses Machine Learning to evaluate loan eligibility based on financial ratios."
)

iface.launch()
