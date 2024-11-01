# CarbonPrediction
A Python-based Long Short-Term Memory (LSTM) model system, which is capable of accurately predicting hourly carbon emissions for the next 24 hours(can be longer)

Version: 1.0  
Authors: Yin Zhang

# 1. Regions covered  
If energy and weather data could be collected for a specific area, this program could theoretically predict any area.   
Currently, only the California region is available for information purposes only as an example. 

# 2. Data Sources
CA(US) region  
 Energy data collected from [CAISO](https://www.gridstatus.io/graph/fuel-mix?iso=caiso&date=2024-07-15to2024-07-29](https://www.gridstatus.io/graph/fuel-mix?iso=caiso&date=2024-07-15to2024-07-29)) Weather data collected from [VisualCrossing](https://www.visualcrossing.com/weather/weather-data-services)
 
# 3. Background to the calculation of carbon emissions
### 3.1 Carbon Intensity and Carbon Emissions  
This code provides a flow of calculations from energy carbon intensity to carbon emissions. It first calculates the carbon intensity per hour based on the carbon emission factors and contribution ratios of each energy source, and then converts it to carbon emissions.

First, we use the following formula for calculating avg carbon intensity:
![Carbon Intensity Formula](image/Carbon%20Intensity%20Formula.png)
 , where

CIavg = Average carbon intensity (real-time or forecast) of a region  
Ei = Electricity produced by source i.  
CEFi = Carbon emission factor (lifecycle/direct) of source i.

Then，we use Carbon Emission Formula for calculating carbon emission coresponding to energy data:
![Carbon Emission Formula](image/Carbon%20Emission%20Formula.png)  

Finally, apply Unit Conversion Formula to get the accurate data with correct unit:
![Unit Conversion Formula](image/Unit%20Conversion%20Formula.png)  


### 3.2 Usage  
The carbonEmissionCalculator.py file provides data loading, cleaning and calculation functions for calculating total carbon emissions from different energy combinations.  

# 4. Run Carbon Emission Calculator and Main code with existing datasets and models
### 4.1 Installing dependencies

Carbon Emission Calculator requires Python3.  
Other required packages:

- Required python modules are listed in `requirements.txt`.  
  Run `pip install -U -r requirements.txt` for installing the dependencies.

### 4.2 Running CarbonPrediction using saved data:
The saved data needs to be inputed into the - [Carbon_Emission_Calculator_2.py](src/Carbon_Emission_Calculator_2.py).  
From the example, the inputed data should be - [Energy data(CAISO 5 minute standardized data_2024-07-15T00_00_00-07_00_2024-07-29T23_59_59.999000-07_00.csv)](data/CAISO%205%20minute%20standardized%20data_2024-07-15T00_00_00-07_00_2024-07-29T23_59_59.999000-07_00.csv), and the outputed data should be - [Energy + Emission combined (combined_energy_data_with_emission_2.csv)](data/combined_energy_data_with_emission_2.csv).  
You can directly download the data using the link before. No need to modify the contents of the document.  
After getting the output, input it to the - [Main_Code for prediction](src/Main_code%20for%20prediction.py) to get the result and the image of the prediction line

