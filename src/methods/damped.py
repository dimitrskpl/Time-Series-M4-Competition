import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def damped(series, alpha, beta, phi, m, n_forecasts=1):

    decomposition = seasonal_decompose(series, model='multiplicative', period=m)
    seasonal = decomposition.seasonal.dropna()
    
    seasonally_adjusted = series / seasonal
    
    # Initialize the level and trend
    l = seasonally_adjusted.iloc[0]
    b = seasonally_adjusted.iloc[1] - seasonally_adjusted.iloc[0]

    # Arrays to store the smoothed values
    level = np.zeros(len(seasonally_adjusted))
    trend = np.zeros(len(seasonally_adjusted))
    level[0] = l
    trend[0] = b

    # Step 2: Compute the level, trend, and forecast recursively for each time step
    for t in range(1, len(seasonally_adjusted)):
        # Update level and trend with damping factor
        l_new = alpha * seasonally_adjusted.iloc[t] + (1 - alpha) * (level[t-1] + phi * trend[t-1])
        b_new = beta * (l_new - level[t-1]) + (1 - beta) * phi * trend[t-1]
        
        level[t] = l_new
        trend[t] = b_new
    
    # Step 3: Forecast for the next period (damped trend extrapolation)
    forecasts = []
    for k in range(1, n_forecasts + 1):
        # By using (len(seasonal) - m + k) % m, we ensure that the seasonal 
        # component for the forecast k periods ahead corresponds to the 
        # correct period in the seasonal cycle
        forecast = (level[-1] + phi * trend[-1] * np.sum([phi ** i for i in range(1, k + 1)])) * seasonal.iloc[(len(seasonal) - m + k) % m]
        #forecast = (level[-1] + phi * trend[-1] * np.sum([phi ** i for i in range(1, k + 1)])) * seasonal.iloc[-m + k % m]
        forecasts.append(forecast)
    
    return forecasts[0], pd.Series(level, index=series.index), pd.Series(trend, index=series.index)


def evaluate_damped(train, test, m, alpha, beta, phi):
    """
    alpha (float): The smoothing factor for the level (between 0 and 1).
    beta (float): The smoothing factor for the trend (between 0 and 1).
    phi (float): The damping factor (between 0 and 1).
    """
    
    results = []
    
    for i in range(len(train)):
        series_id = train.iloc[i, 0] 
        train_series = train.iloc[i, 1:].dropna().astype(float)  
        test_series = test.iloc[i, 1:].dropna().astype(float)  
        
        if len(train_series) > 0 and len(test_series) > 0:
            forecast, _, _ = damped(train_series, alpha, beta, phi, m)
            
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
                    'Alpha': alpha,
                    'Beta': beta,
                    'Phi': phi
                })
    
    results_df = pd.DataFrame(results)
    return results_df


def evaluate_damped_statsmodels(train, test, m, alpha, beta, phi):
    """
    Evaluate Damped Exponential Smoothing using statsmodels on train and test sets.
    
    If m=1 (e.g., yearly, weekly, daily), the seasonal component is disabled.
    
    Parameters:
    train (pd.DataFrame): Training dataset (time series data for all series).
    test (pd.DataFrame): Testing dataset (time series data for all series).
    m (int): The seasonal period (e.g., 12 for monthly data, 4 for quarterly data, or 1 for no seasonality).
    alpha (float): The smoothing factor for the level (between 0 and 1).
    beta (float): The smoothing factor for the trend (between 0 and 1).
    phi (float): The damping factor (between 0 and 1).

    Returns:
    pd.DataFrame: A DataFrame with forecast, actual, and error for each series.
    """
    
    results = []
    
    for i in range(len(train)):
        series_id = train.iloc[i, 0]  # Assuming the first column is the series ID
        train_series = train.iloc[i, 1:].dropna().astype(float)  # The remaining columns are the time series
        test_series = test.iloc[i, 1:].dropna().astype(float)  # Test series
        
        if len(train_series) > 0 and len(test_series) > 0:
            # Handle case where m=1 (no seasonality)
            if m == 1:
                # No seasonality
                model = ExponentialSmoothing(
                    train_series, 
                    trend='add', 
                    damped_trend=True, 
                    seasonal=None  # Disable seasonal component
                )
            else:
                # Use seasonal component if m > 1
                model = ExponentialSmoothing(
                    train_series, 
                    trend='add', 
                    damped_trend=True, 
                    seasonal='multiplicative', 
                    seasonal_periods=m
                )
                
            # Fit the model with provided parameters
            fit = model.fit(smoothing_level=alpha, smoothing_slope=beta, damping_slope=phi)
            
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
                    'm': m,
                    'Alpha': alpha,
                    'Beta': beta,
                    'Phi': phi
                })
    
    results_df = pd.DataFrame(results)
    return results_df
