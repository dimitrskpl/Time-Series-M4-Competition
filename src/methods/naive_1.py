import pandas as pd


# Function to implement Naïve 1 forecasting
def naive_1_forecast(series):
    return series.iloc[-1]
    
# Evaluate Naïve 1 method on the dataset
def evaluate_naive_1(train, test, m):
    results = []
    for i in range(len(train)):
        series_id = train.iloc[i, 0]
        # train_series = train.iloc[i, 1:].dropna().astype(float)
        # test_series = test.iloc[i, 1:].dropna().astype(float)  # Use the entire test set
        train_series = train.iloc[i, 1:].astype(float).ffill()
        test_series = test.iloc[i, 1:].astype(float).ffill()

        if len(train_series) > 0 and len(test_series) > 0:
            forecast = naive_1_forecast(train_series)
            forecasts = [forecast] * len(test_series)  # Extend the last observed value
            
            for j in range(len(test_series)):
                actual = test_series.iloc[j]
                error = forecasts[j] - actual
                results.append({
                    'Series': series_id,
                    'Forecast': forecasts[j],
                    'Actual': actual,
                    'Error': error,
                    'TrainSeries': train_series,
                    'm': m
                })
    
    results_df = pd.DataFrame(results)
    return results_df
