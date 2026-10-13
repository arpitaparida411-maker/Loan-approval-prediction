import streamlit as st
import joblib
import numpy as np

model = joblib.load('loan_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Loan Approval Prediction')
st.write('Enter applicant details to predict loan approval status.')
gender = st.selectbox('Gender', ['Male', 'Female'])
married = st.selectbox('Married', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['0', '1', '2', '3'])
education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
applicant_income = st.number_input('Applicant Income', min_value=0, value=5000)
coapplicant_income = st.number_input('Coapplicant Income', min_value=0, value=0)
loan_amount = st.number_input('Loan Amount (in thousands)', min_value=0, value=120)
loan_term = st.selectbox('Loan Amount Term (days)', [360, 180, 120, 84, 60, 36])
credit_history = st.selectbox('Credit History', ['Has credit history (1)', 'No credit history (0)'])
property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])
if st.button('Predict'):
    gender_val = 1 if gender == 'Male' else 0
    married_val = 1 if married == 'Yes' else 0
    education_val = 1 if education == 'Graduate' else 0
    self_employed_val = 1 if self_employed == 'Yes' else 0
    credit_history_val = 1 if credit_history == 'Has credit history (1)' else 0
    dependents_val = int(dependents)

    property_semiurban = 1 if property_area == 'Semiurban' else 0
    property_urban = 1 if property_area == 'Urban' else 0

    total_income = applicant_income + coapplicant_income
    total_income_log = np.log1p(total_income)
    loan_amount_log = np.log1p(loan_amount)
    input_data = np.array([[gender_val, married_val, dependents_val, education_val,
                             self_employed_val, applicant_income, coapplicant_income,
                             loan_amount, loan_term, credit_history_val,
                             property_semiurban, property_urban,
                             total_income, total_income_log, loan_amount_log]])
    

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success('Loan Approved ✅')
    else:
        st.error('Loan Rejected ❌')