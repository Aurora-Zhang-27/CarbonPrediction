import pandas as pd

# Replace with your file path
file_path = '/Users/mac/Desktop/CAISO2/hourly_emission_rates.csv'

# Load the data
historical_emissions = pd.read_csv(file_path)

# Ensure 'datetime_beginning_utc' column is in datetime format with specified format
historical_emissions['datetime_beginning_utc'] = pd.to_datetime(
    historical_emissions['datetime_beginning_utc'], format="%m/%d/%Y %I:%M:%S %p"
)

# Group by each hour and calculate the sum of 'total_sum_co2' column
hourly_total_emissions = historical_emissions.groupby('datetime_beginning_utc')['total_sum_co2'].sum().reset_index()

# Divide the total emissions by 1000
hourly_total_emissions['total_hourly_emission_co2'] = hourly_total_emissions['total_sum_co2'] / 1000

# Rename columns for clarity
hourly_total_emissions.rename(columns={'datetime_beginning_utc': 'datetime_utc', 'total_hourly_emission_co2': 'total_hourly_emission_co2_kg'}, inplace=True)

# Save to a new file if needed
output_path = '/Users/mac/Desktop/CAISO2/new_hourly_total_emissions.csv'
hourly_total_emissions.to_csv(output_path, index=False)
print(f"Hourly total emissions saved to {output_path}")