import pandas as pd
import numpy as np

bus_schedules = pd.read_csv('data/bus_schedules.csv')
demographics = pd.read_csv('data/demographics.csv')
holidays = pd.read_csv('data/holiday.csv', encoding='ISO-8859-1')

def handle_missing_values(df):
    return df.fillna(0)

bus_schedules = handle_missing_values(bus_schedules)
demographics = handle_missing_values(demographics)
holidays = handle_missing_values(holidays)

bus_schedules['StartTime'] = pd.to_datetime(bus_schedules['StartTime'], format='%H:%M', errors='coerce').dt.time
bus_schedules['EndTime'] = pd.to_datetime(bus_schedules['EndTime'], format='%H:%M', errors='coerce').dt.time

demographics['hostelers'] = demographics['school_population'] * 0.30
demographics['public_transport_students'] = demographics['school_population'] * (1 - 0.45 - 0.30)

def adjust_passenger_count_based_on_day(bus_schedules):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day in days:
        bus_schedules['Passengerscount'] *= np.random.uniform(0.9, 1.1) 
    return bus_schedules

bus_schedules = adjust_passenger_count_based_on_day(bus_schedules)

bus_schedules.to_csv('data/processed_bus_schedules.csv', index=False)
demographics.to_csv('data/processed_demographics.csv', index=False)

print("Data preprocessing completed and saved to 'data/processed_bus_schedules.csv' and 'data/processed_demographics.csv'.")
