import streamlit as st
import pandas as pd
import joblib
from src.make_prediction import process_input_for_prediction, predict_buses  # Import prediction logic

# Load the trained models and scaler
lin_reg = joblib.load('models/linear_regression_model.pkl')
ridge_reg = joblib.load('models/ridge_regression_model.pkl')
scaler = joblib.load('models/scaler.pkl')

def main():
    st.title("Bus Demand Prediction App")

    # Date and time input fields
    date_input = st.date_input("Select a date:")
    time_input = st.time_input("Select a time:")

    # Button to predict
    if st.button("Predict"):
        if date_input and time_input:
            date_str = date_input.strftime('%Y-%m-%d')
            time_str = time_input.strftime('%H:%M')

            # Process the input and make predictions
            feature_array = process_input_for_prediction(date_str, time_str)
            lin_prediction, ridge_prediction = predict_buses(feature_array)

            # Display the predictions
            st.subheader("Predicted Number of Buses Required")
            st.write(f"Linear Regression: {lin_prediction:.2f} buses")
            st.write(f"Ridge Regression: {ridge_prediction:.2f} buses")
        else:
            st.error("Please select both date and time.")

if __name__ == '__main__':
    main()
