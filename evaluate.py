"""Evaluation metrics for price regression and direction classification."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_price_predictions(y_true, y_pred):
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }


def evaluate_classification(y_true, y_pred):
    y_pred = np.asarray(y_pred)
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-score": float(f1_score(y_true, y_pred, zero_division=0)),
        "Confusion Matrix": confusion_matrix(y_true, y_pred),
        "predicted_up_count": int(np.sum(y_pred == 1)),
        "predicted_down_count": int(np.sum(y_pred == 0)),
    }


def baseline_improvement(model_value, baseline_value, higher_is_better=False):
    """Return percentage improvement of model over baseline."""
    if baseline_value == 0:
        return float("nan")
    if higher_is_better:
        return float((model_value - baseline_value) / abs(baseline_value) * 100)
    return float((baseline_value - model_value) / abs(baseline_value) * 100)


def comparison_message(model_name, baseline_name, improvement_pct):
    if np.isnan(improvement_pct):
        return f"{model_name}: cannot compare against {baseline_name} because baseline metric is zero."

    if improvement_pct > 0:
        message = f"{model_name} outperformed {baseline_name} by {improvement_pct:.2f}%."
    else:
        message = f"{model_name} did not outperform {baseline_name}."

    if 0 < improvement_pct < 2:
        message += " Improvement is small and may not be meaningful."
    return message
