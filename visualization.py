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


def plot_price_prediction(actual, predicted, output_dir=OUTPUT_DIR, filename="price_prediction.png"):
    output_dir = _ensure_output_dir(output_dir)
    path = output_dir / filename

    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual.values, label="Actual Close", linewidth=2)
    plt.plot(predicted.index, predicted.values, label="Predicted Close", linewidth=2)
    plt.title("Actual Close vs Predicted Close")
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    price_df = pd.DataFrame(price_metrics).set_index("Model")
    price_df[["MAE", "RMSE"]].plot(kind="bar", ax=axes[0])
    axes[0].set_title("Price Prediction Error")
    axes[0].set_ylabel("Error")
    axes[0].tick_params(axis="x", rotation=20)

    class_df = pd.DataFrame(classification_metrics).set_index("Model")
    class_df[["Accuracy", "F1-score"]].plot(kind="bar", ax=axes[1], color=["#457b9d", "#e76f51"])
    axes[1].set_title("Direction Classification Performance")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path

