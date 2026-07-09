"""Run the complete stock forecasting workflow."""

import numpy as np

from baseline import majority_class_baseline
from config import (
    CLASSIFICATION_THRESHOLD,
    END_DATE,
    FUTURE_DIRECTION_DAYS,
    OUTPUT_DIR,
    PRICE_COLUMN,
    START_DATE,
    TEST_SIZE,
    TICKER,
)
from data_loader import load_stock_data
from evaluate import baseline_improvement, comparison_message
from features import FEATURE_COLUMNS, build_features
from ml_model import get_random_forest_feature_importance, run_classification_models
from sarima_model import run_sarima_price_model
from visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    plot_price_prediction,
    plot_prediction_distribution,
)


def format_table(rows, columns):
    widths = {
        column: max(len(str(column)), *(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }
    header = " | ".join(str(column).ljust(widths[column]) for column in columns)
    separator = "-+-".join("-" * widths[column] for column in columns)
    body = [
        " | ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns)
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def metric_row(model_name, metrics, extra=None):
    row = {
        "Model": model_name,
        "MAE": f"{metrics['MAE']:.4f}" if "MAE" in metrics else "",
        "RMSE": f"{metrics['RMSE']:.4f}" if "RMSE" in metrics else "",
        "MAPE": f"{metrics['MAPE']:.4f}%" if "MAPE" in metrics else "",
        "Accuracy": f"{metrics['Accuracy']:.4f}" if "Accuracy" in metrics else "",
        "Precision": f"{metrics['Precision']:.4f}" if "Precision" in metrics else "",
        "Recall": f"{metrics['Recall']:.4f}" if "Recall" in metrics else "",
        "F1-score": f"{metrics['F1-score']:.4f}" if "F1-score" in metrics else "",
        "Pred Up": metrics.get("predicted_up_count", ""),
        "Pred Down": metrics.get("predicted_down_count", ""),
    }
    if extra:
        row.update(extra)
    return row


def print_classification_metrics(name, metrics):
    print(f"\n[{name}] Direction classification metrics")
    print(f"Accuracy : {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall   : {metrics['Recall']:.4f}")
    print(f"F1-score : {metrics['F1-score']:.4f}")
    print(f"predicted up  : {metrics['predicted_up_count']}")
    print(f"predicted down: {metrics['predicted_down_count']}")
    print("Confusion Matrix:")
    print(metrics["Confusion Matrix"])


def main():
    print("=== Stock Forecast Model ===")
    print(f"Ticker: {TICKER}")
    print(f"Date range: {START_DATE} to {END_DATE}")

    df, data_source = load_stock_data(TICKER, START_DATE, END_DATE)
    print(f"Data source: {data_source}")
    print(f"Samples loaded: {len(df)}")

    feature_df = build_features(df)
    print(f"Rows after feature engineering: {len(feature_df)}")
    print(f"Feature count: {len(FEATURE_COLUMNS)}")
    print(f"Classification threshold: {CLASSIFICATION_THRESHOLD}")

    price_output = run_sarima_price_model(df, price_col=PRICE_COLUMN, test_size=TEST_SIZE)
    price_results = price_output["results"]
    naive_result = price_output["naive"]

    print("\n=== Price prediction model comparison ===")
    price_rows = []
    for result in price_results:
        extra = {}
        if result.get("order") is not None:
            extra["Best order"] = result["order"]
            extra["AIC"] = f"{result['aic']:.2f}" if np.isfinite(result.get("aic", np.nan)) else ""
        else:
            extra["Best order"] = ""
            extra["AIC"] = ""
        price_rows.append(metric_row(result["name"], result["metrics"], extra=extra))
    print(format_table(price_rows, ["Model", "MAE", "RMSE", "MAPE", "Best order", "AIC"]))

    print("\n=== SARIMA parameter search top results ===")
    for result in [price_output["sarima_close"], price_output["sarima_log_return"]]:
        top_rows = [
            {
                "Model": result["name"],
                "Order": row["order"],
                "AIC": f"{row['aic']:.2f}",
            }
            for row in result.get("search_results", [])[:5]
        ]
        if top_rows:
            print(f"\n{result['name']}")
            print(format_table(top_rows, ["Model", "Order", "AIC"]))

    classification_runs = []
    classification_rows = []
    target_specs = [
        ("Next-day direction", "label_next_day"),
        (f"{FUTURE_DIRECTION_DAYS}-day future direction", "label_5d"),
    ]

    for target_name, target_col in target_specs:
        ml_outputs = run_classification_models(
            feature_df,
            test_size=TEST_SIZE,
            target_col=target_col,
            threshold=CLASSIFICATION_THRESHOLD,
        )
        majority_result = majority_class_baseline(ml_outputs["y_train"], ml_outputs["y_test"])
        majority_result["target_name"] = target_name
        majority_result["target_col"] = target_col
        all_results = [majority_result, *ml_outputs["results"]]
        classification_runs.append(
            {
                "target_name": target_name,
                "target_col": target_col,
                "ml_outputs": ml_outputs,
                "results": all_results,
                "majority": majority_result,
            }
        )

        for result in all_results:
            classification_rows.append(
                metric_row(
                    result["name"],
                    result["metrics"],
                    extra={"Target": target_name},
                )
            )

    print("\n=== Classification model comparison ===")
    print(
        format_table(
            classification_rows,
            [
                "Target",
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1-score",
                "Pred Up",
                "Pred Down",
            ],
        )
    )

    print("\n=== Classification confusion matrices ===")
    for run in classification_runs:
        print(f"\nTarget: {run['target_name']}")
        for result in run["results"]:
            print_classification_metrics(result["name"], result["metrics"])

    default_run = classification_runs[-1]
    random_forest_result = next(
        result for result in default_run["results"] if result["name"] == "Random Forest"
    )
    logistic_result = next(
        result for result in default_run["results"] if result["name"] == "Logistic Regression"
    )
    importances = get_random_forest_feature_importance(random_forest_result)

    output_paths = []
    output_paths.append(plot_price_prediction(price_results))
    output_paths.append(
        plot_confusion_matrix(
            logistic_result["metrics"]["Confusion Matrix"],
            filename="confusion_matrix_logistic.png",
            title=f"Logistic Regression Confusion Matrix ({default_run['target_name']})",
        )
    )
    output_paths.append(
        plot_confusion_matrix(
            random_forest_result["metrics"]["Confusion Matrix"],
            filename="confusion_matrix_random_forest.png",
            title=f"Random Forest Confusion Matrix ({default_run['target_name']})",
        )
    )
    output_paths.append(plot_feature_importance(importances))

    price_metrics = [
        {"Model": result["name"], **result["metrics"]}
        for result in price_results
    ]
    classification_metrics = []
    for result in default_run["results"]:
        row = {
            "Model": result["name"],
            **{k: v for k, v in result["metrics"].items() if k != "Confusion Matrix"},
        }
        classification_metrics.append(row)

    output_paths.append(plot_model_comparison(price_metrics, classification_metrics))
    output_paths.append(plot_prediction_distribution(default_run["results"]))

    print("\n=== Baseline comparison conclusions ===")
    sarima_improvements = []
    for result in [price_output["sarima_close"], price_output["sarima_log_return"]]:
        improvement = baseline_improvement(
            result["metrics"]["MAE"],
            naive_result["metrics"]["MAE"],
            higher_is_better=False,
        )
        sarima_improvements.append(improvement)
        print(comparison_message(result["name"], "Naive Price Baseline", improvement))

    if all(improvement <= 0 for improvement in sarima_improvements):
        print("SARIMA did not outperform the naive baseline on this dataset.")

    for run in classification_runs:
        majority = run["majority"]
        for result in run["results"]:
            if result["name"] == "Majority Class Baseline":
                continue
            improvement = baseline_improvement(
                result["metrics"]["Accuracy"],
                majority["metrics"]["Accuracy"],
                higher_is_better=True,
            )
            print(
                f"{run['target_name']} - "
                + comparison_message(result["name"], "Majority Class Baseline", improvement)
            )

    print("\n=== Output files ===")
    for path in output_paths:
        print(path.resolve())

    if data_source == "sample random-walk data":
        print(
            "\nWARNING: sample data was used. Results are only for testing workflow, "
            "not real investment analysis."
        )

    print("\nDone.")


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
