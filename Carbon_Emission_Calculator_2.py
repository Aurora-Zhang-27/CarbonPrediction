import pandas as pd
import numpy as np

carbonRateLifecycle = {
    "coal": 820, "biomass": 230, "nat_gas": 490, "geothermal": 38, "hydro": 24,
    "nuclear": 12, "oil": 650, "solar": 45, "unknown": 700, "other": 700, "wind": 11
}

def initialize(file_path):
    """
    Initializes the dataset by loading the energy data from the provided file.
    """
    dataset = pd.read_csv(file_path)
    dataset['date'] = pd.to_datetime(dataset['date'] + ' ' + dataset['time'])
    dataset.set_index('date', inplace=True)
    dataset.drop(columns=['time'], inplace=True)
    
    dataset.replace(np.nan, 0, inplace=True)  # Replace NaN values with 0
    num = dataset._get_numeric_data()
    num[num < 0] = 0  # Set any negative values to 0

    return dataset

def calculateCarbonIntensity(dataset, carbonRate):
    """
    Calculates the carbon intensity based on the energy sources in the dataset and carbon emission factors.
    """
    carbonCol = []
    sources = ['fuel_mix.coal', 'fuel_mix.biomass', 'fuel_mix.natural_gas', 'fuel_mix.large_hydro', 
               'fuel_mix.imports', 'fuel_mix.other', 'fuel_mix.biogas']

    for _, row in dataset.iterrows():
        carbonIntensity = 0
        rowSum = sum(row[source] for source in sources if source in row)
        
        for source in sources:
            if rowSum > 0 and source in row:
                sourceContribFrac = row[source] / rowSum
                carbonIntensity += sourceContribFrac * carbonRate.get(source.split('.')[-1], 0)
        
        carbonCol.append(round(carbonIntensity, 2))

    dataset['carbon_intensity'] = carbonCol
    return dataset

def convertIntensityToEmissions(dataset):
    """
    Converts carbon intensity to carbon emissions using the energy consumption data
    and ensures proper unit conversions to mTCO₂/h.
    """
    emissions = []
    for _, row in dataset.iterrows():
        carbon_intensity = row['carbon_intensity']  # Carbon intensity in gCO₂/kWh
        energy_consumption = row['net_load'] * 1_000_000  # Convert net_load from GW to kWh
        
        # Calculate carbon emissions in grams and adjust by multiplying by 100
        carbon_emissions_grams = carbon_intensity * energy_consumption * 100  # Result in gCO₂, adjusted

        # Convert carbon emissions from grams to mTCO₂/h
        carbon_emissions_mTCO2_per_hour = carbon_emissions_grams / 1_000_000_000  # Convert from g to mTCO₂/h
        emissions.append(round(carbon_emissions_mTCO2_per_hour, 6))

    dataset['carbon_emissions_mTCO2_per_hour'] = emissions
    return dataset

def runCarbonIntensityCalculation(file_path, output_path, isLifecycle=True):
    """
    Runs the carbon intensity calculation for the provided dataset using either lifecycle or direct emission factors.
    Generates a new CSV file with the calculated carbon emissions.
    """
    dataset = initialize(file_path)
    
    if isLifecycle:
        dataset = calculateCarbonIntensity(dataset, carbonRateLifecycle)
    else:
        dataset = calculateCarbonIntensity(dataset, carbonRateDirect)
    
    # Convert carbon intensity to carbon emissions in mTCO₂/h
    dataset = convertIntensityToEmissions(dataset)

    # Display all data points for the range of July 15 to July 30
    full_data = dataset.loc['2024-07-15':'2024-07-30']
    print(full_data.to_string(index=True))

    # Save the updated dataset with carbon emissions to a new CSV file
    dataset.to_csv(output_path)
    print(f"Updated dataset with carbon emissions saved to {output_path}")

    return dataset

# document path
input_file_path = '/Users/mac/Desktop/CAISO/CAISO 5 minute standardized data_2024-07-15T00_00_00-07_00_2024-07-29T23_59_59.999000-07_00.csv'
output_file_path = '/Users/mac/Desktop/CAISO/combined_energy_data_with_emissions.csv'

result_dataset = runCarbonIntensityCalculation(input_file_path, output_file_path, isLifecycle=True)










