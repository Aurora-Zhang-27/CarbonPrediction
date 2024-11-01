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

We use the following formula for calculating avg carbon intensity:
![Carbon Intensity Formula](image/Carbon%20Intensity%20Formula.png)
 , where

CIavg = Average carbon intensity (real-time or forecast) of a region  
Ei = Electricity produced by source i.  
CEFi = Carbon emission factor (lifecycle/direct) of source i.

Then，we use Carbon Emission Formula:
![Carbon Emission Formula](image/Carbon%20Emission%20Formula.png)  
Finally, apply Unit Conversion Formula:
![Unit Conversion Formula](image/Unit%20Conversion%20Formula.png)  


### 3.2 Usage  
The carbonEmissionCalculator.py file provides data loading, cleaning and calculation functions for calculating total carbon emissions from different energy combinations.  
