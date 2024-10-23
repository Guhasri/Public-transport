import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
feature_engineered_data = pd.read_csv('data/feature_engineered_bus_demand.csv')
processed_bus_schedule_data = pd.read_csv('data/processed_bus_schedules.csv')

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.scatterplot(data=feature_engineered_data, x='TotalPassengers', y='RequiredBuses', color='blue', label='Feature Engineered Data')
plt.title('Total Passengers vs Required Buses (Feature Engineered Data)')
plt.xlabel('Total Passengers')
plt.ylabel('Required Buses')
plt.legend()
plt.grid(True)
plt.savefig('visualizations/total_passengers_vs_required_buses.png') 
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=processed_bus_schedule_data, x='Passengerscount', color='orange')
plt.title('Passengers Count Distribution (Processed Bus Schedule Data)')
plt.xlabel('Passengers Count')
plt.grid(True)
plt.savefig('visualizations/passengers_count_distribution.png') 
plt.show()


plt.figure(figsize=(12, 8))

corr_matrix = feature_engineered_data.drop(columns=['Date']).corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix of Feature Engineered Bus Demand Data (Date Excluded)')
plt.savefig('visualizations/correlation_matrix_feature_engineered_excluding_date.png') 
plt.show()


plt.figure(figsize=(12, 6))
sns.histplot(feature_engineered_data['RequiredBuses'], bins=20, kde=True, color='purple')
plt.title('Distribution of Required Buses (Feature Engineered Data)')
plt.xlabel('Required Buses')
plt.grid(True)
plt.savefig('visualizations/distribution_of_required_buses.png') 
plt.show()
