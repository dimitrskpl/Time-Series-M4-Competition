import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose


def ses_forecast(series, alpha):
    smoothed_series = np.zeros(len(series))
    
    # Initialize S_0 with the first observation)
    smoothed_series[0] = series.iloc[0]
    
    # Compute smoothed values for the rest of the series
    for t in range(1, len(series)):
        smoothed_series[t] = alpha * series.iloc[t] + (1 - alpha) * smoothed_series[t-1]
    
    forecast = smoothed_series[-1]
    
    return forecast, pd.Series(smoothed_series, index=series.index)

def ses(series, m, alpha):
    decomposition = seasonal_decompose(series, model='multiplicative', period=m)
    seasonal = decomposition.seasonal.dropna()
    
    seasonally_adjusted = series / seasonal
    
    forecast, smoothed_series = ses_forecast(seasonally_adjusted.dropna(), alpha)
    
    forecast = forecast * seasonal.iloc[-m] 
    
    return forecast


def evaluate_ses(train, test, m, alpha):
    results = []
    
    for i in range(len(train)):
        series_id = train.iloc[i, 0] 
        train_series = train.iloc[i, 1:].dropna().astype(float) 
        test_series = test.iloc[i, 1:].dropna().astype(float)  
        
        if len(train_series) > 0 and len(test_series) > 0:
            forecast = ses(train_series, m, alpha)
            
            forecast_values = [forecast] * len(test_series)
            for j, (actual, forecast_value) in enumerate(zip(test_series, forecast_values)):
                error = forecast_value - actual
                results.append({
                    'Series': series_id,
                    'Forecast': float(forecast_value),
                    'Actual': float(actual),
                    'Error': float(error),
                    'TrainSeries': train_series.to_dict(),
                    'm': m,
                    'Alpha': alpha
                })
    
    results_df = pd.DataFrame(results)
    return results_df



from statsmodels.tsa.holtwinters import SimpleExpSmoothing

def evaluate_ses_statsmodels(train, test, m, alpha):
    
    results = []
    
    for i in range(len(train)):
        series_id = train.iloc[i, 0]  # Assuming the first column is the series ID
        train_series = train.iloc[i, 1:].dropna().astype(float)  # The remaining columns are the time series
        test_series = test.iloc[i, 1:].dropna().astype(float)  # Test series
        
        if len(train_series) > 0 and len(test_series) > 0:
            # Use statsmodels SimpleExpSmoothing for SES
            model = SimpleExpSmoothing(train_series)
            
            # Fit the model with provided alpha
            fit = model.fit(smoothing_level=alpha)
            
            # Generate the forecast
            forecast = fit.forecast(steps=len(test_series))

            for j, (actual, forecast_value) in enumerate(zip(test_series, forecast)):
                error = forecast_value - actual
                results.append({
                    'Series': series_id,
                    'Forecast': float(forecast_value),
                    'Actual': float(actual),
                    'Error': float(error),
                    'TrainSeries': train_series.to_dict(),
                    'm': m,  # Even though m is not used in SES, we include it for consistency
                    'Alpha': alpha
                })
    
    results_df = pd.DataFrame(results)
    return results_df
