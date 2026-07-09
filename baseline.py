"""Baseline models for price and direction prediction."""

import numpy as np
import pandas as pd

from evaluate import evaluate_classification, evaluate_price_predictions


def train_test_split_time_series(data, test_size=0.2):
    split_idx = int(len(data) * (1 - test_size))
    if split_idx <= 0 or split_idx >= len(data):
        raise ValueError("Not enough rows for time-series train/test split.")
    return data.iloc[:split_idx].copy(), data.iloc[split_idx:].copy()


def naive_price_baseline(price_series, test_size=0.2):
    split_idx = int(len(price_series) * (1 - test_size))
    test = price_series.iloc[split_idx:].copy()
    predictions = price_series.shift(1).iloc[split_idx:]

    valid = predictions.notna()
    y_true = test.loc[valid]
    y_pred = predictions.loc[valid]

    metrics = evaluate_price_predictions(y_true, y_pred)
    return {
        "name": "Naive Price Baseline",
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def naive_price_baseline_from_split(full_price_series, test_index):
    """Naive baseline aligned to an existing test index."""
    predictions = full_price_series.shift(1).reindex(test_index)
    y_true = full_price_series.reindex(test_index)
    valid = predictions.notna() & y_true.notna()
    y_true = y_true.loc[valid]
    y_pred = predictions.loc[valid]

    return {
        "name": "Naive Price Baseline",
        "metrics": evaluate_price_predictions(y_true, y_pred),
        "y_true": y_true,
        "y_pred": pd.Series(y_pred.values, index=y_true.index, name="naive_predicted_close"),
    }


def majority_class_baseline(train_labels, test_labels):
    majority_class = int(train_labels.mode().iloc[0])
    predictions = np.full(shape=len(test_labels), fill_value=majority_class)
    metrics = evaluate_classification(test_labels, predictions)
    return {
        "name": "Majority Class Baseline",
        "majority_class": majority_class,
        "metrics": metrics,
        "y_true": test_labels,
        "y_pred": predictions,
    }
