import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ✅ Load the trained model
model_path = 'churn_model.pkl'

if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found! Please place 'churn_model.pkl' in the same folder as app.py.")
else:
    model = pickle.load(open(model_path, 'rb'))

    st.title("📊 Customer Churn Prediction App")
    st.write("Use this app to predict whether a customer will churn or not based on input details.")

    # Input fields
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0)
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0)

    # Encoding categorical variables
    gender = 1 if gender == "Male" else 0
    Partner = 1 if Partner == "Yes" else 0
    Dependents = 1 if Dependents == "Yes" else 0

    # Create dataframe for prediction
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges]
    })

    # Prediction button
    if st.button("Predict"):
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.error("❌ Customer is likely to Churn.")
        else:
            st.success("✅ Customer is likely to Stay.")



# Save model
with open('churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)