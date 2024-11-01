# Carbon-Emissions-Prediction
A Python-based Long Short-Term Memory (LSTM) model system, which is capable of accurately predicting hourly carbon emissions for the next 24 hours(can be longer)

Version: 1.0  
Authors: Yin Zhang
# 1. Regions covered  
If energy and weather data could be collected for a specific area, this program could theoretically predict any area.   
Currently, only the California region is available for information purposes only. 
# 2. Data Sources
CA(US) region  
 Energy data collected from [CAISO](https://www.gridstatus.io/graph/fuel-mix?iso=caiso&date=2024-07-15to2024-07-29](https://www.gridstatus.io/graph/fuel-mix?iso=caiso&date=2024-07-15to2024-07-29)) Weather data collected from [VisualCrossing](https://www.visualcrossing.com/weather/weather-data-services)
# 3. Background to the calculation of carbon emissions
### 3.1 Carbon Intensity and Carbon Emissions  
This code provides a flow of calculations from energy carbon intensity to carbon emissions. It first calculates the carbon intensity per hour based on the carbon emission factors and contribution ratios of each energy source, and then converts it to carbon emissions.  
Carbon Intensity Calculation: The hourly contribution ratios of the different energy sources are multiplied by their corresponding carbon emission factors to arrive at the carbon emissions per unit of electricity (gCO₂/kWh).  
Carbon emissions conversion: Carbon intensity is multiplied by net_load (i.e., total load per hour) and converted by units to yield carbon emissions, expressed in mTCO₂/h.
### 3.2 Usage  
The carbonEmissionCalculator.py file provides data loading, cleaning and calculation functions for calculating total carbon emissions from different energy combinations.  
