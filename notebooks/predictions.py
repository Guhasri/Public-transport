import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data/feature_engineered_bus_demand.csv')

plt.figure(figsize=(16, 12))

columns_to_plot = ['StartHour', 'EndHour', 'TotalPassengers', 'MaxPassengers', 'MinPassengers', 'AvgPassengers', 'RequiredBuses']

for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(3, 3, i)
    sns.histplot(data[column], kde=True, bins=20, color='blue')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
