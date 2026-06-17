# CarbonPrediction

CarbonPrediction is an early machine-learning project for short-term carbon-emissions forecasting with recurrent neural networks.

The project combines grid generation data, weather data, carbon-intensity accounting, and LSTM/GRU-based time-series models to forecast hourly carbon emissions. It started as a first exploration of carbon and energy forecasting, and later informed my work on carbon-intensity forecasting and structured fuel-mix surrogate modeling.

**Version:** 1.2.0  
**Author:** Yin Zhang

## Highlights in Version 1.2.0

The later PJM alpha workflow added:

- multi-step forecasting support
- LSTM and GRU architecture options
- dynamic feature selection
- improved preprocessing and robust scaling
- configurable input sequence length and forecast horizon

## Regions Covered

The workflow is designed to be adaptable to any region where energy, weather, and emissions data can be collected and aligned.

Current examples:

1. **CAISO**: California
2. **PJM**: Delaware, Illinois, Indiana, Kentucky, Maryland, Michigan, New Jersey, North Carolina, Ohio, Pennsylvania, Tennessee, Virginia, West Virginia, and the District of Columbia

## Data Sources

### CAISO

- Energy data: [GridStatus](https://www.gridstatus.io/)
- Weather data: [Visual Crossing](https://www.visualcrossing.com/weather/weather-data-services)

### PJM

- Energy / fuel-mix data: [GridStatus](https://www.gridstatus.io/)
- Weather data: [Visual Crossing](https://www.visualcrossing.com/weather/weather-data-services)
- Historical carbon-emissions data: [PJM](https://www.pjm.com/)

## Carbon-Emissions Accounting

The project includes a carbon-emissions calculation workflow that converts energy-source generation into average carbon intensity and total emissions.

Average carbon intensity is calculated from the generation contribution and emissions factor of each energy source:

```text
average carbon intensity = sum(generation_i * emissions_factor_i) / total_generation
```

The resulting carbon intensity can then be combined with generation or demand to estimate hourly carbon emissions.

The main utility scripts include:

- `Carbon_Emission_Calculator_2.py`: calculates emissions from energy-source data
- `Unit Conversion and Summation Calculator.py`: converts and aggregates PJM emissions data into a usable hourly format
- `CarbonPrediction_*.py`: trains and evaluates forecasting models

## Repository Structure

```text
src/        Carbon accounting, preprocessing, and forecasting scripts
data/       Example regional datasets
image/      Formula images and result figures
README.md   Project documentation
```

## Installation

Install Python dependencies:

```bash
pip install -U -r requirements.txt
```

Python 3 is required.

## Running with Saved Data

### CAISO Example

1. Place CAISO energy data under `data/CA(US)/`.
2. Run `Carbon_Emission_Calculator_2.py` to generate combined energy and emissions data.
3. Use the output file as input to `CarbonPrediction_CA(US).py`.
4. Train the model and generate prediction plots.

### PJM Example

1. Place PJM energy, weather, and historical emissions data under `data/PJM(US)/`.
2. If raw PJM emissions data need unit conversion or aggregation, run `Unit Conversion and Summation Calculator.py`.
3. Use the processed emissions file with `CarbonPrediction_PJM(US).py`.
4. Train the model and evaluate the forecast.

## Running with New Data

To adapt the project to a new region:

1. Collect energy/fuel-mix data and weather data.
2. Align all time series to an hourly interval.
3. If direct emissions data are unavailable, calculate emissions from energy-source mix and emissions factors.
4. Update file paths inside the forecasting script.
5. Train the forecasting model.

Example path fields to update:

```python
energy_data_path = "path/to/energy_data.csv"
weather_data_path = "path/to/weather_data.csv"
historical_emissions_path = "path/to/hourly_emissions.csv"
```

If datetime column names differ across files, update the preprocessing block accordingly:

```python
energy_data["datetime"] = pd.to_datetime(energy_data["interval_start_utc"]).dt.tz_localize(None)
weather_data["datetime"] = pd.to_datetime(weather_data["datetime"]).dt.tz_localize(None)
historical_emissions["datetime"] = pd.to_datetime(historical_emissions["datetime_utc"]).dt.tz_localize(None)
```

## Multi-Step Forecasting

The PJM alpha version supports multi-step forecasting. Key parameters include:

- `IS_MULTISTEP`: switch between single-step and multi-step forecasting
- `OUTPUT_WINDOW`: number of future hours to predict
- `sequence_length`: historical input length
- `n_layers`: number of recurrent layers
- `rnn_units`: hidden units per recurrent layer

Example:

```python
IS_MULTISTEP = True
OUTPUT_WINDOW = 24
sequence_length = 24
```

## Notes

- Forecasting quality depends strongly on data quality, feature alignment, and region-specific reporting conventions.
- For regions with direct historical carbon-emissions data, the calculator step can be skipped.
- This project is preserved as an early-stage carbon forecasting workflow and is not the final version of my current research pipeline.

## Tools

`Python` `pandas` `NumPy` `scikit-learn` `matplotlib` `LSTM` `GRU` `time-series forecasting`

