"""Run the complete stock forecasting workflow."""

import numpy as np

from baseline import majority_class_baseline, naive_price_baseline
from config import END_DATE, OUTPUT_DIR, PRICE_COLUMN, START_DATE, TEST_SIZE, TICKER
from data_loader import load_stock_data
from features import build_features
from ml_model import get_random_forest_feature_importance, run_classification_models
from sarima_model import run_sarima_price_model
from visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    plot_price_prediction,
)


def print_price_metrics(name, metrics):
    print(f"\n[{name}] Price prediction metrics")
    print(f"MAE : {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAPE: {metrics['MAPE']:.4f}%")


def print_classification_metrics(name, metrics):
    print(f"\n[{name}] Direction classification metrics")
    print(f"Accuracy : {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall   : {metrics['Recall']:.4f}")
    print(f"F1-score : {metrics['F1-score']:.4f}")
    print("Confusion Matrix:")
    print(metrics["Confusion Matrix"])


def main():
    print("=== Stock Forecast Model ===")
    print(f"Ticker: {TICKER}")
    print(f"Date range: {START_DATE} to {END_DATE}")

    df, data_source = load_stock_data(TICKER, START_DATE, END_DATE)
    print(f"Rows loaded: {len(df)}")
    print(f"Confirmed data source: {data_source}")

    feature_df = build_features(df)
    print(f"Rows after feature engineering: {len(feature_df)}")

    naive_result = naive_price_baseline(df[PRICE_COLUMN], test_size=TEST_SIZE)
    print_price_metrics(naive_result["name"], naive_result["metrics"])

    sarima_result = run_sarima_price_model(df, price_col=PRICE_COLUMN, test_size=TEST_SIZE)
    print_price_metrics(sarima_result["name"], sarima_result["metrics"])

    ml_outputs = run_classification_models(feature_df, test_size=TEST_SIZE)
    majority_result = majority_class_baseline(ml_outputs["y_train"], ml_outputs["y_test"])
    print_classification_metrics(majority_result["name"], majority_result["metrics"])

    for result in ml_outputs["results"]:
        print_classification_metrics(result["name"], result["metrics"])

    random_forest_result = next(result for result in ml_outputs["results"] if result["name"] == "Random Forest")
    importances = get_random_forest_feature_importance(random_forest_result)

    output_paths = []
    output_paths.append(plot_price_prediction(sarima_result["y_true"], sarima_result["y_pred"]))
    output_paths.append(plot_confusion_matrix(random_forest_result["metrics"]["Confusion Matrix"]))
    output_paths.append(plot_feature_importance(importances))

    price_metrics = [
        {"Model": naive_result["name"], **naive_result["metrics"]},
        {"Model": sarima_result["name"], **sarima_result["metrics"]},
    ]
    classification_metrics = [
        {
            "Model": majority_result["name"],
            **{k: v for k, v in majority_result["metrics"].items() if k != "Confusion Matrix"},
        }
    ]
    for result in ml_outputs["results"]:
        classification_metrics.append(
            {
                "Model": result["name"],
                **{k: v for k, v in result["metrics"].items() if k != "Confusion Matrix"},
            }
        )

    output_paths.append(plot_model_comparison(price_metrics, classification_metrics))

    print("\n=== Output files ===")
    for path in output_paths:
        print(path.resolve())

    if data_source == "sample data":
        print("\nWARNING: sample data was used. It is only for testing the code flow, not real investment analysis.")

    print("\nDone.")


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main()

