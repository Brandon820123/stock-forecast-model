"""SARIMA/ARIMA price forecasting model."""

import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from baseline import naive_price_baseline_from_split, train_test_split_time_series
from config import (
    SARIMA_D_VALUES,
    SARIMA_ORDER,
    SARIMA_P_VALUES,
    SARIMA_Q_VALUES,
    SARIMA_SEASONAL_ORDER,
    TEST_SIZE,
)
from evaluate import evaluate_price_predictions


def _fit_sarimax(series, order, seasonal_order):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False, maxiter=80)


def search_sarima_order(
    train_series,
    p_values=SARIMA_P_VALUES,
    d_values=SARIMA_D_VALUES,
    q_values=SARIMA_Q_VALUES,
    seasonal_order=SARIMA_SEASONAL_ORDER,
):
    """Search p/d/q by AIC on the training set."""
    series = pd.Series(train_series).dropna().astype(float)
    best_order = SARIMA_ORDER
    best_aic = float("inf")
    rows = []

    for order in itertools.product(p_values, d_values, q_values):
        try:
            fitted = _fit_sarimax(series, order=order, seasonal_order=seasonal_order)
            aic = float(fitted.aic)
            rows.append({"order": order, "seasonal_order": seasonal_order, "aic": aic})
            if aic < best_aic:
                best_order = order
                best_aic = aic
        except Exception:
            continue

    return best_order, best_aic, sorted(rows, key=lambda row: row["aic"])


def rolling_sarima_close_forecast(train_series, test_series, order, seasonal_order):
    """Walk-forward close-price forecast.

    Each step forecasts only the next day, then appends the observed true value
    before the following step. Parameters are fitted on the training window and
    state is updated with each observed test value for practical runtime.
    """
    train_series = pd.Series(train_series).dropna().astype(float)
    test_series = pd.Series(test_series).dropna().astype(float)
    predictions = []

    try:
        fitted = _fit_sarimax(train_series, order=order, seasonal_order=seasonal_order)
    except Exception:
        fitted = None

    last_observed = float(train_series.iloc[-1])
    for date, actual in test_series.items():
        if fitted is not None:
            pred = float(np.asarray(fitted.forecast(steps=1))[0])
        else:
            pred = last_observed

        predictions.append(pred)
        actual = float(actual)
        last_observed = actual

        if fitted is not None:
            try:
                fitted = fitted.append(pd.Series([actual], index=[date]), refit=False)
            except Exception:
                fitted = None

    return pd.Series(predictions, index=test_series.index, name="sarima_close_predicted")


def rolling_sarima_log_return_forecast(train_series, test_series, order, seasonal_order):
    """Forecast log return and restore it to price in a walk-forward loop."""
    train_series = pd.Series(train_series).dropna().astype(float)
    test_series = pd.Series(test_series).dropna().astype(float)
    log_return_history = np.log(train_series / train_series.shift(1)).dropna()
    last_close = float(train_series.iloc[-1])
    predictions = []

    try:
        fitted = _fit_sarimax(log_return_history, order=order, seasonal_order=seasonal_order)
    except Exception:
        fitted = None

    for date, actual_close in test_series.items():
        if fitted is not None:
            predicted_log_return = float(np.asarray(fitted.forecast(steps=1))[0])
        else:
            predicted_log_return = 0.0

        predicted_close = last_close * np.exp(predicted_log_return)
        predictions.append(float(predicted_close))

        actual_log_return = np.log(float(actual_close) / last_close)
        last_close = float(actual_close)

        if fitted is not None:
            try:
                fitted = fitted.append(pd.Series([actual_log_return], index=[date]), refit=False)
            except Exception:
                fitted = None

    return pd.Series(predictions, index=test_series.index, name="sarima_log_return_predicted")


def _result(name, y_true, y_pred, order=None, seasonal_order=None, aic=None, search_results=None):
    return {
        "name": name,
        "metrics": evaluate_price_predictions(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
        "order": order,
        "seasonal_order": seasonal_order,
        "aic": aic,
        "search_results": search_results or [],
    }


def run_sarima_price_model(
    df,
    price_col="Close",
    test_size=TEST_SIZE,
    seasonal_order=SARIMA_SEASONAL_ORDER,
):
    """Run Naive, SARIMA Close, and SARIMA Log Return on the same test set."""
    train, test = train_test_split_time_series(df[[price_col]], test_size=test_size)
    train_series = train[price_col].astype(float)
    test_series = test[price_col].astype(float)

    naive_result = naive_price_baseline_from_split(df[price_col].astype(float), test_series.index)

    close_order, close_aic, close_search = search_sarima_order(
        train_series,
        seasonal_order=seasonal_order,
    )
    close_pred = rolling_sarima_close_forecast(
        train_series,
        test_series,
        order=close_order,
        seasonal_order=seasonal_order,
    )
    close_result = _result(
        "SARIMA Close",
        test_series,
        close_pred,
        order=close_order,
        seasonal_order=seasonal_order,
        aic=close_aic,
        search_results=close_search,
    )

    train_log_returns = np.log(train_series / train_series.shift(1)).dropna()
    log_order, log_aic, log_search = search_sarima_order(
        train_log_returns,
        seasonal_order=seasonal_order,
    )
    log_return_pred = rolling_sarima_log_return_forecast(
        train_series,
        test_series,
        order=log_order,
        seasonal_order=seasonal_order,
    )
    log_return_result = _result(
        "SARIMA Log Return",
        test_series,
        log_return_pred,
        order=log_order,
        seasonal_order=seasonal_order,
        aic=log_aic,
        search_results=log_search,
    )

    return {
        "train": train,
        "test": test,
        "results": [naive_result, close_result, log_return_result],
        "naive": naive_result,
        "sarima_close": close_result,
        "sarima_log_return": log_return_result,
    }
