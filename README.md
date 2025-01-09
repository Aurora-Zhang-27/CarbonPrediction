# CarbonPrediction
A Python-based Long Short-Term Memory (LSTM) model system, which is capable of accurately predicting hourly carbon emissions for the next 24 hours(can be longer)

Version: 1.2.0  
Authors: Yin Zhang
### Highlights in Version 1.2.0
- Enhanced **multi-step forecasting**.
- Improved **model flexibility**, supporting advanced LSTM and GRU architectures.
- Optimized **data preprocessing**, enabling dynamic feature selection and robust scaling.

## 1. Regions covered  
If energy and weather data could be collected for a specific area, this program could theoretically predict any area.   
Current example:  
1. California: CA(US)   
2. Delaware, Illinois, Indiana, Kentucky, Maryland, Michigan, New Jersey, North Carolina, Ohio, Pennsylvania, Tennessee, Virginia, West Virginia and the District of Columbia: PJM(US)

## 2. Data Sources
CA(US) region  
 Energy data collected from [gridstatus](https://www.gridstatus.io/graph/fuel-mix?iso=caiso&date=2024-07-15to2024-07-29](https://www.gridstatus.io/graph/fuel-mix?iso=caiso&date=2024-07-15to2024-07-29)) Weather data collected from [VisualCrossing](https://www.visualcrossing.com/weather/weather-data-services)   
PJM(US) region  
 Energy data collected from [gridstatus](https://www.gridstatus.io/graph/fuel-mix?iso=pjm&date=2023-10-31to2024-10-31) Weather data collected from [VisualCrossing](https://www.visualcrossing.com/weather/weather-data-services/new%20york/metric/2023-11-01/2024-10-31) Carbon Emission Data collected from [PJM](https://www.pjm.com/)
 
## 3. Background to the calculation of carbon emissions
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
The [Carbon Emission Calculator](src/Carbon_Emission_Calculator_2.py) program provides data loading, cleaning and calculation functions for calculating total carbon emissions from different energy combinations.  

The [CarbonPrediction](src) program use combines LSTM, GRU and CNN network architectures for time series prediction tasks based on historical carbon emissions and weather data. By predicting carbon emissions for the next 24 hours, the code supports real-time monitoring and forecasting of regional carbon emissions.  

The [Unit Conversion and Summation Calculator](src/Unit%20Conversion%20and%20Summation%20Calculator.py) program can change the historical carbon emission data from [PJM](https://dataminer2.pjm.com/feed/hourly_emission_rates.) into correct form and unit.


## 4. Run Carbon Emission Calculator and CarbonPrediction with existing datasets and models
### 4.1 Installing dependencies

[Carbon Emission Calculator](src/Carbon_Emission_Calculator_2.py) requires Python3.  
Other required packages:

- Required python modules are listed in `requirements.txt`.  
  Run `pip install -U -r requirements.txt` for installing the dependencies.

### 4.2 Running CarbonPrediction_CA(US) using saved data:
The saved data in [CA(US)](data/CA(US)) needs to be inputed into the [Carbon Emission Calculator](src/Carbon_Emission_Calculator_2.py).  
From the example, the inputed data should be [Energy data](data/CA(US)/CAISO%205%20minute%20standardized%20data_2024-09-30T00_00_00-07_00_2024-10-30T23_59_59.999000-07_00.csv), and the outputed data should be [Energy + Emission combined data](data/CA(US)/combined_energy_data_with_emissions.csv).  
You can directly download the data using the link before. No need to modify the contents of the document.  
After getting the output, input it to the [CarbonPrediction_CA(US)](src/CarbonPrediction_CA(US).py) to get the result and the image of the prediction line.

### 4.3 Running CarbonPrediction_PJM(US) using saved data:  
The saved data in [PJM(US)](data/PJM(US)) which are [Energy data](data/PJM(US)/PJM%205%20minute%20standardized%20data_2024-09-30T00_00_00-04_00_2024-10-30T23_59_59.999000-04_00.csv), [Weather data](data/PJM(US)/new%20york%202024-09-30%20to%202024-10-30.csv), [Processed historical carbon emissions data](data/PJM(US)/new_hourly_emission_rates.csv) needs to be inputed into the [CarbonPrediction_PJM(US)](src/CarbonPrediction_PJM(US).py) directly, and you can have the result.  
Note that if you want to use the [Unprocessed historical carbon emissions data](data/PJM(US)/hourly_emission_rates.csv) you should first input it into [Unit Conversion and Summation Calculator](src/Unit%20Conversion%20and%20Summation%20Calculator.py) which you can change the data into correct form and unit. Then, repeat the above.

## 5. Running CarbonPrediction from scratch
To run CarbonPrediction from scratch (with new data/for new regions etc.), first install the dependencies mentioned in Section 4.1.  
### 5.1 Getting weather/energy data     
You can download any weather data you want from the Internet. Then, make sure that the interval between each row of data is one hour. Similarly, Repeat the above for your own energy data. 
### 5.2 Calculating Carbon Emission corresopnding to energy data       
After having two data files, make sure that they are csv files and each column corresponds to a different energy and weather type. Next, open [Carbon Emission Calculator](src/Carbon_Emission_Calculator_2.py) and change the path of the code here:
```python
input_file_path = '/Users/mac/Desktop/CAISO/CAISO 5 minute standardized data_2024-07-15T00_00_00-07_00_2024-07-29T23_59_59.999000-07_00.csv'
output_file_path = '/Users/mac/Desktop/CAISO/combined_energy_data_with_emissions.csv'
```
 Note that, you should make sure that all of the files have to be in one floder.  
### 5.3 Getting carbon emission forecasts using CarbonPrediction  
Input the new csv file which outputed from the [Carbon Emission Calculator](src/Carbon_Emission_Calculator_2.py) to [CarbonPrediction_CA(US)](src/CarbonPrediction_CA(US).py) by changing file path here:
```python
# Load data
energy_data_path = 'combined_energy_data_with_emissions.csv'
weather_data_path = 'weather_data_of_california 2024-09-30 to 2024-10-30.csv'
```  
and input your own weather data by changing the path in line 11.  
If you need data which is more than 24 hours, you can change the `sequence_length`. Note that the larger the sequence_length, the lower the precision rate afterward.  

[CarbonPrediction_PJM(US)_(alpha)](src/CarbonPrediction_PJM(US)_(alpha).py) allows customization of model architecture:
1. **Number of Layers:** You can adjust the depth of the LSTM or GRU network by modifying the `n_layers` parameter.  
2. **Units per Layer:** Increase the number of neurons with the `rnn_units` parameter for better long-term dependencies.  
3. **Prediction Length:** Update the `sequence_length` for input length and `OUTPUT_WINDOW` for output prediction length.  

### 5.4 If you are able to collect the historical carbon emission data  
No need to use the calculator. Please Note that the interval between each row of data of carbon emission should be an hour as well. Then, since you don't have to use the calculator, you should use [CarbonPrediction_PJM(US)](src/CarbonPrediction_PJM(US).py) to make prediction.  
In the code, you should replace the file paths here:
```python
# 1. Load the energy, weather, and historical carbon emission data
energy_data_path = '/Users/mac/Desktop/CAISO2/PJM 5 minute standardized data_2024-09-30T00_00_00-04_00_2024-10-30T23_59_59.999000-04_00.csv'
weather_data_path = '/Users/mac/Desktop/CAISO2/new york 2024-09-30 to 2024-10-30.csv'
historical_emissions_path = '/Users/mac/Desktop/CAISO2/new_hourly_emission_rates.csv'
```
Also, you may have to change the datatime name to your own datatime name of your files here:
```python
# 2. Preprocess data
energy_data['datetime'] = pd.to_datetime(energy_data['interval_start_utc']).dt.tz_localize(None)
weather_data['datetime'] = pd.to_datetime(weather_data['datetime']).dt.tz_localize(None)
historical_emissions['datetime'] = pd.to_datetime(historical_emissions['datetime_utc']).dt.tz_localize(None)
```
As for [CarbonPrediction_PJM(US)_(alpha)](src/CarbonPrediction_PJM(US)_(alpha).py), the operation is the same.  

### Note：  
If your data comes from [PJM](https://www.pjm.com/)(recommand) or has different types of energy carbon emissions at one point in time like [hourly_emission_rates.csv](data/PJM(US)/hourly_emission_rates.csv), and you need to sum them before you can continue, [Unit Conversion and Summation Calculator](src/Unit%20Conversion%20and%20Summation%20Calculator.py) can help you.  
Replace your file name here:
```python
file_path = '/Users/mac/Desktop/CAISO2/hourly_emission_rates.csv'
output_path = '/Users/mac/Desktop/CAISO2/new_hourly_total_emissions.csv'
```
and it will automatically generate a file. You should put it in the folder where the origin file is located. This is your available carbon emissions data. Input it to [CarbonPrediction_PJM(US)](src/CarbonPrediction_PJM(US).py) with your energy and weather data, you will get the result.  

### 5.5 Multi-step Forecasting with [CarbonPrediction_PJM(US)_(alpha)](src/CarbonPrediction_PJM(US)_(alpha).py)
To enable multi-step forecasting (e.g., predicting the next 24 hours or more):
1. For mode changing:  
```python
# ========== Mode Selection: Single-step or Multi-step ==========
IS_MULTISTEP = True   # True => Multi-step output, False => Single-step output
OUTPUT_WINDOW = 24    # Number of steps to predict in multi-step mode
```  
2. Modify the `OUTPUT_WINDOW` parameter in the code (default: 24).
3. Use the rolling forecast functionality to generate predictions for the next desired time frame:
   ```python
   forecast_list = []
   for _ in range(OUTPUT_WINDOW):
       X_pred = last_sequence.reshape((1, sequence_length, -1))
       pred = model.predict(X_pred)
       forecast_list.append(pred[0, 0])
       last_sequence = np.append(last_sequence[1:], pred[0, 0])

## 6. Citing CarbonPrediction
## 7. Acknowledgements





