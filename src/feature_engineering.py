import pandas as pd
import numpy as np
from datetime import timedelta

bus_schedules = pd.read_csv('data/processed_bus_schedules.csv')
demographics = pd.read_csv('data/processed_demographics.csv')
holidays = pd.read_csv('data/holiday.csv')

time_slots = [
    (6, 10), (10, 14), (14, 18),
    (18, 22), (22, 2), (2, 6)
]
bus_capacity = 60

dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')

holidays['Date'] = pd.to_datetime(holidays['Date'], format='%m/%d/%Y', errors='coerce')

def calculate_required_buses(total_passengers):
    return np.ceil(total_passengers / bus_capacity)

engineered_data = []

for date in dates:
    is_friday = date.weekday() == 4  
    is_holiday_period = any(
        (holiday - timedelta(days=2)) <= date <= (holiday + timedelta(days=2))
        for holiday in holidays['Date'] if pd.notna(holiday)
    )
    
    for start_hour, end_hour in time_slots:
        
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
        
        if is_holiday_period:
            total_passengers *= 1.25
        if is_friday:
            total_passengers *= 1.10        
        
        if not relevant_buses.empty:
            
            max_passengers = relevant_buses['Passengerscount'].max()
            min_passengers = relevant_buses['Passengerscount'].min()
            avg_passengers = relevant_buses['Passengerscount'].mean()            
            
            fluctuation_factor = np.random.uniform(0.9, 1.1) 
            max_passengers *= fluctuation_factor
            min_passengers *= fluctuation_factor
            avg_passengers *= fluctuation_factor
        else:
            max_passengers = min_passengers = avg_passengers = 0         
        
        required_buses = calculate_required_buses(total_passengers)
        
        engineered_data.append({
            'Date': date,
            'StartHour': start_hour,
            'EndHour': end_hour,
            'Friday': is_friday,
            'HolidayEffect': is_holiday_period,
            'TotalPassengers': total_passengers,
            'RequiredBuses': required_buses,
            'MaxPassengers': round(max_passengers, 2),
            'MinPassengers': round(min_passengers, 2),
            'AvgPassengers': round(avg_passengers, 2)
        })

engineered_df = pd.DataFrame(engineered_data)

engineered_df.to_csv('data/feature_engineered_bus_demand.csv', index=False)

print("Feature engineering completed and saved to 'data/feature_engineered_bus_demand.csv'.")
