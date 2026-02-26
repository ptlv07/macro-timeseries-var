# macro-timeseries-var
Econometric pipeline modeling lead-lag relationships between WTI crude oil prices and 10-Year US Treasury yields using VAR, Granger causality, and impulse response analysis.

## Setup
pip install pandas numpy statsmodels matplotlib fredapi

## Usage
Run with synthetic data out of the box:
python macro_timeseries_analysis.py

To use real FRED data, set USE_LIVE_API = True and add your free 
API key from fred.stlouisfed.org

## Output
![Analysis Results](var_analysis_results.png)
