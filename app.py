import streamlit as st
import pandas as pd
import joblib

st.title("Credit Risk Prediction")
st.write(
    "Estimate the probability of default on a consumer loan, "
    "based solely on data known at the time of the application (no variables"
    "assigned by the bank following an assessment that has already taken place)."
)

model = joblib.load("models/best_credit_risk_model.pkl")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual Income", min_value=0, value=50000, step=1000)
    home_ownership = st.selectbox("Type of property", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    emp_length = st.number_input("Years of work experience", min_value=0, max_value=60, value=5)
    default_on_file = st.selectbox("Previous defaults recorded in the register?", ["N", "Y"])

with col2:
    loan_intent = st.selectbox(
        "Scopo del prestito",
        ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
    )
    loan_amnt = st.number_input("Amount:", min_value=500, value=10000, step=500)
    cred_hist_length = st.number_input("Years of credit history", min_value=0, max_value=60, value=5)

# derived features — these must replicate src/data_prep.py::engineer_features exactly
loan_percent_income = loan_amnt / income if income > 0 else 0
credit_hist_to_age_ratio = cred_hist_length / age if age > 0 else 0
income_after_loan = income - loan_amnt
income_per_emp_year = income / (emp_length + 1)

input_df = pd.DataFrame([{
    "person_age": age,
    "person_income": income,
    "person_home_ownership": home_ownership,
    "person_emp_length": emp_length,
    "loan_intent": loan_intent,
    "loan_amnt": loan_amnt,
    "loan_percent_income": loan_percent_income,
    "cb_person_default_on_file": default_on_file,
    "cb_person_cred_hist_length": cred_hist_length,
    "credit_hist_to_age_ratio": credit_hist_to_age_ratio,
    "income_after_loan": income_after_loan,
    "income_per_emp_year": income_per_emp_year,
}])

if st.button("Assess the risk"):
    proba_default = model.predict_proba(input_df)[0, 1]

    st.metric("Estimated probability of default", f"{proba_default:.1%}")

    if proba_default < 0.15:
        st.success("Low Risk")
    elif proba_default < 0.40:
        st.warning("Medium Risk")
    else:
        st.error("High Risk")

st.caption(
    "Model trained without loan_grade/loan_int_rate (variables assigned "
    "by the bank following its own risk assessment) — see notebooks/credit_risk_analysis.ipynb"
)
