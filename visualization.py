"""Visualization utilities for model outputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import OUTPUT_DIR


def _ensure_output_dir(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_price_prediction(price_results, output_dir=OUTPUT_DIR, filename="price_prediction.png"):
    output_dir = _ensure_output_dir(output_dir)
    path = output_dir / filename

    plt.figure(figsize=(12, 6))
    actual = price_results[0]["y_true"]
    plt.plot(actual.index, actual.values, label="Actual Close", linewidth=2)
    for result in price_results:
        predicted = result["y_pred"]
        plt.plot(predicted.index, predicted.values, label=result["name"], linewidth=1.6)
    plt.title("Actual Close vs Price Forecast Models")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_confusion_matrix(cm, output_dir=OUTPUT_DIR, filename="confusion_matrix.png", title="Confusion Matrix"):
    output_dir = _ensure_output_dir(output_dir)
    path = output_dir / filename

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred Down/Flat", "Pred Up"],
        yticklabels=["Actual Down/Flat", "Actual Up"],
    )
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_feature_importance(importances, output_dir=OUTPUT_DIR, filename="feature_importance.png"):
    output_dir = _ensure_output_dir(output_dir)
    path = output_dir / filename

    plt.figure(figsize=(10, 6))
    importances.sort_values().plot(kind="barh", color="#2a9d8f")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_model_comparison(price_metrics, classification_metrics, output_dir=OUTPUT_DIR, filename="model_comparison.png"):
    output_dir = _ensure_output_dir(output_dir)
    path = output_dir / filename

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    price_df = pd.DataFrame(price_metrics).set_index("Model")
    price_df[["MAE", "RMSE", "MAPE"]].plot(kind="bar", ax=axes[0])
    axes[0].set_title("Price Prediction Metrics")
    axes[0].set_ylabel("Error")
    axes[0].set_xlabel("Model")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].legend(title="Metric")

    class_df = pd.DataFrame(classification_metrics).set_index("Model")
    class_df[["Accuracy", "F1-score"]].plot(kind="bar", ax=axes[1], color=["#457b9d", "#e76f51"])
    axes[1].set_title("Direction Classification Performance")
    axes[1].set_ylabel("Score")
    axes[1].set_xlabel("Model")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend(title="Metric")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_prediction_distribution(classification_results, output_dir=OUTPUT_DIR, filename="prediction_distribution.png"):
    output_dir = _ensure_output_dir(output_dir)
    path = output_dir / filename

    rows = []
    for result in classification_results:
        metrics = result["metrics"]
        rows.append(
            {
                "Model": result["name"],
                "Predicted Up": metrics["predicted_up_count"],
                "Predicted Down": metrics["predicted_down_count"],
            }
        )
    dist_df = pd.DataFrame(rows).set_index("Model")

    plt.figure(figsize=(10, 6))
    dist_df.plot(kind="bar", color=["#2a9d8f", "#e76f51"])
    plt.title("Prediction Distribution: Up vs Down")
    plt.xlabel("Model")
    plt.ylabel("Prediction Count")
    plt.xticks(rotation=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path
