import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

lin_reg = joblib.load('models/linear_regression_model.pkl')
ridge_reg = joblib.load('models/ridge_regression_model.pkl')
scaler = joblib.load('models/scaler.pkl')

bus_schedules = pd.read_csv('data/processed_bus_schedules.csv')
holidays = pd.read_csv('data/holiday.csv')

holidays['Date'] = pd.to_datetime(holidays['Date'], format='%m/%d/%Y', errors='coerce')

def is_holiday(date):
    return date in holidays['Date'].values

def calculate_total_passengers(start_hour, end_hour):
    if start_hour < end_hour:
        relevant_buses = bus_schedules[
            (pd.to_datetime(bus_schedules['StartTime'], format='%H:%M:%S', errors='coerce').dt.hour >= start_hour) & 
            (pd.to_datetime(bus_schedules['EndTime'], format='%H:%M:%S', errors='coerce').dt.hour <= end_hour)
        ]
    else:
        relevant_buses = bus_schedules[
            (pd.to_datetime(bus_schedules['StartTime'], format='%H:%M:%S', errors='coerce').dt.hour >= start_hour) |
            (pd.to_datetime(bus_schedules['EndTime'], format='%H:%M:%S', errors='coerce').dt.hour <= end_hour)
        ]
    
    total_passengers = relevant_buses['Passengerscount'].sum()
    return total_passengers

def process_input_for_prediction(date_str, time_str):
    date = pd.to_datetime(date_str, format='%Y-%m-%d')
    time = pd.to_datetime(time_str, format='%H:%M').time()

    start_hour = time.hour
    end_hour = (time.hour + 1) % 24
    friday = 1 if date.weekday() == 4 else 0  
    holiday_effect = 1 if is_holiday(date) else 0 
    
    total_passengers = calculate_total_passengers(start_hour, end_hour)

    feature_array = [[start_hour, end_hour, friday, holiday_effect, total_passengers]]
    
    return feature_array

def predict_buses(feature_array):
    feature_names = ['StartHour', 'EndHour', 'Friday', 'HolidayEffect', 'TotalPassengers']

    feature_df = pd.DataFrame(feature_array, columns=feature_names)

    feature_array_scaled = scaler.transform(feature_df)
    
    lin_pred = lin_reg.predict(feature_array_scaled)
    
    ridge_pred = ridge_reg.predict(feature_array_scaled)
    
    return lin_pred[0], ridge_pred[0]

if __name__ == '__main__':
    date_input = input("Enter date (YYYY-MM-DD): ")
    time_input = input("Enter time (HH:MM): ")
    feature_array = process_input_for_prediction(date_input, time_input)
    lin_prediction, ridge_prediction = predict_buses(feature_array)
    print(f"Predicted number of buses required:")
    print(f"Linear Regression: {lin_prediction:.2f} buses")
    print(f"Ridge Regression: {ridge_prediction:.2f} buses")
