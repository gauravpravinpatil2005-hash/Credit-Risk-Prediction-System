import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("outputs/models/best_model.joblib")

st.title("Credit Risk Prediction System")

st.write("Enter customer details:")

RevolvingUtilizationOfUnsecuredLines = st.number_input(
    "Revolving Utilization Of Unsecured Lines", value=0.5
)

age = st.number_input("Age", value=35)

NumberOfTime30_59DaysPastDueNotWorse = st.number_input(
    "30-59 Days Past Due", value=0
)

DebtRatio = st.number_input("Debt Ratio", value=0.5)

MonthlyIncome = st.number_input("Monthly Income", value=5000)

NumberOfOpenCreditLinesAndLoans = st.number_input(
    "Open Credit Lines And Loans", value=5
)

NumberOfTimes90DaysLate = st.number_input(
    "90 Days Late", value=0
)

NumberRealEstateLoansOrLines = st.number_input(
    "Real Estate Loans Or Lines", value=1
)

NumberOfTime60_89DaysPastDueNotWorse = st.number_input(
    "60-89 Days Past Due", value=0
)

NumberOfDependents = st.number_input(
    "Number Of Dependents", value=1
)

if st.button("Predict Risk"):

    data = np.array([[
        RevolvingUtilizationOfUnsecuredLines,
        age,
        NumberOfTime30_59DaysPastDueNotWorse,
        DebtRatio,
        MonthlyIncome,
        NumberOfOpenCreditLinesAndLoans,
        NumberOfTimes90DaysLate,
        NumberRealEstateLoansOrLines,
        NumberOfTime60_89DaysPastDueNotWorse,
        NumberOfDependents
    ]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("High Credit Risk")
    else:
        st.success("Low Credit Risk")