"""SARIMA/ARIMA price forecasting model."""

import warnings

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from baseline import train_test_split_time_series
from config import SARIMA_ORDER, SARIMA_SEASONAL_ORDER, TEST_SIZE
from evaluate import evaluate_price_predictions


def run_sarima_price_model(
    df,
    price_col="Close",
    test_size=TEST_SIZE,
    order=SARIMA_ORDER,
    seasonal_order=SARIMA_SEASONAL_ORDER,
):
    train, test = train_test_split_time_series(df[[price_col]], test_size=test_size)
    train_series = train[price_col].astype(float)
    test_series = test[price_col].astype(float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            train_series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)

    forecast = fitted.get_forecast(steps=len(test_series)).predicted_mean
    forecast = pd.Series(forecast.values, index=test_series.index, name="predicted_close")

    metrics = evaluate_price_predictions(test_series, forecast)
    return {
        "name": "SARIMA",
        "model": fitted,
        "metrics": metrics,
        "y_true": test_series,
        "y_pred": forecast,
        "order": order,
        "seasonal_order": seasonal_order,
    }

