import pandas as pd
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import seasonal_decompose


# def naive_2_forecast(series, m):
#     # Perform the 90% autocorrelation test
#     acf_values = acf(series, nlags=m)
#     if acf_values[m] > 1.645 / np.sqrt(len(series)):  # 1.645 is the z-value for 90% confidence
#         # Data is seasonal, apply multiplicative decomposition
#         decomposition = seasonal_decompose(series, model='multiplicative', period=m)
#         seasonal = decomposition.seasonal
#         adjusted_series = series / seasonal
#         forecast = adjusted_series.iloc[-m:] * seasonal.iloc[-m:]
#         forecast = forecast.iloc[-1]  # Use the last seasonal value for forecasting
#     else:
#         # Data is not seasonal, use last value
#         forecast = series.iloc[-1]
    
#     return forecast


def naive_2_forecast(series, m):
    """
    Implements Naive 2 forecasting method for the M4 competition.
    
    series: time series data
    m: the seasonal period (e.g., 12 for monthly data)
    """
    # Perform the 90% autocorrelation test to check for seasonality
    acf_values = acf(series, nlags=m)
    
    # Check if the absolute autocorrelation at lag m indicates seasonality
    if abs(acf_values[m]) > 0.9:
        # Data is seasonal, apply multiplicative decomposition
        decomposition = seasonal_decompose(series, model='multiplicative', period=m)
        seasonal = decomposition.seasonal.dropna()
        
        # Seasonally adjust the series
        seasonally_adjusted = series / seasonal
        
        # Use the last seasonally adjusted value and multiply by the last seasonal component
        forecast = seasonally_adjusted.iloc[-1] * seasonal.iloc[-m]  # Use the seasonal value from the same season
    else:
        # Data is not seasonal, use last value
        forecast = series.iloc[-1]
    
    return forecast

def evaluate_naive_2(train, test, m):
    """
    Evaluate Naive 2 method on train and test sets.
    
    train: training dataset (time series data for all series)
    test: testing dataset (time series data for all series)
    m: the seasonal period (e.g., 12 for monthly data)
    """
    results = []
    
    for i in range(len(train)):
        series_id = train.iloc[i, 0]  
        # train_series = train.iloc[i, 1:].dropna().astype(float)  
        # test_series = test.iloc[i, 1:].dropna().astype(float)  
        train_series = train.iloc[i, 1:].astype(float).ffill()
        test_series = test.iloc[i, 1:].astype(float).ffill()
        
        if len(train_series) > 0 and len(test_series) > 0:
            forecast = naive_2_forecast(train_series, m)
            
            # Generate forecast for each step in test series
            forecast_values = [forecast] * len(test_series)
            for j, (actual, forecast_value) in enumerate(zip(test_series, forecast_values)):
                error = forecast_value - actual
                results.append({
                    'Series': series_id,
                    'Forecast': float(forecast_value),  
                    'Actual': float(actual),  
                    'Error': float(error),  
                    'TrainSeries': train_series.to_dict(),  
                    'm': m
                })
    
    # Return results as a DataFrame
    results_df = pd.DataFrame(results)
    return results_df
