python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'Public_Transportation_User_Data_2026.csv'
data = pd.read_csv(file_path)

# Data Cleaning and Preparation
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

# Analyzing passenger trends by transportation type
transport_type_analysis = data.groupby(['Transport_Type', 'Month'])['Passenger_Count'].sum().unstack()

# Plot the trends
transport_type_analysis.T.plot(kind='line', figsize=(12, 8), title='Monthly Passenger Trends by Transport Type')
plt.xlabel('Month')
plt.ylabel('Passenger Count')
plt.legend(title='Transport Type')
plt.grid()
plt.show()

# Predictive Analytics for Future Passenger Counts
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Aggregating data by month
monthly_data = data.groupby(['Year', 'Month'])['Passenger_Count'].sum().reset_index()
monthly_data['TimeIndex'] = monthly_data.index

# Fitting the model
model = ExponentialSmoothing(monthly_data['Passenger_Count'], seasonal='add', seasonal_periods=12).fit()

# Forecasting for the next 12 months
forecast = model.forecast(steps=12)

# Visualization
plt.figure(figsize=(14, 7))
plt.plot(monthly_data['Passenger_Count'], label='Historical Data')
plt.plot(range(len(monthly_data), len(monthly_data) + len(forecast)), forecast, label='Forecast', color='red')
plt.title('Passenger Count Forecast for Next 12 Months')
plt.xlabel('Month')
plt.ylabel('Passenger Count')
plt.legend()
plt.grid()
plt.show()

# Exporting Cleaned Data
cleaned_file_path = 'Cleaned_Public_Transportation_User_Data_2026.csv'
data.to_csv(cleaned_file_path, index=False)
print(f"Cleaned data saved to {cleaned_file_path}")
