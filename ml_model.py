"""Machine learning models for next trading day direction classification."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from baseline import train_test_split_time_series
from config import CLASSIFICATION_THRESHOLD, RANDOM_STATE, TEST_SIZE
from evaluate import evaluate_classification
from features import FEATURE_COLUMNS


def _predict_with_threshold(model, X_test, threshold):
    if not hasattr(model, "predict_proba"):
        predictions = model.predict(X_test)
        return predictions, None

    probabilities = model.predict_proba(X_test)
    classes = list(model.classes_)
    if 1 not in classes:
        up_probability = np.zeros(len(X_test))
    else:
        up_probability = probabilities[:, classes.index(1)]
    predictions = (up_probability > threshold).astype(int)
    return predictions, up_probability


def _fit_predict_model(name, model, X_train, y_train, X_test, y_test, threshold):
    model.fit(X_train, y_train)
    predictions, up_probability = _predict_with_threshold(model, X_test, threshold)
    metrics = evaluate_classification(y_test, predictions)
    return {
        "name": name,
        "model": model,
        "metrics": metrics,
        "y_true": y_test,
        "y_pred": predictions,
        "up_probability": up_probability,
        "threshold": threshold,
    }


def run_classification_models(
    feature_df,
    test_size=TEST_SIZE,
    target_col="label_5d",
    threshold=CLASSIFICATION_THRESHOLD,
):
    train, test = train_test_split_time_series(feature_df, test_size=test_size)

    X_train = train[FEATURE_COLUMNS]
    y_train = train[target_col].astype(int)
    X_test = test[FEATURE_COLUMNS]
    y_test = test[target_col].astype(int)

    logistic = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    random_forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    results = [
        _fit_predict_model("Logistic Regression", logistic, X_train, y_train, X_test, y_test, threshold),
        _fit_predict_model("Random Forest", random_forest, X_train, y_train, X_test, y_test, threshold),
    ]

    return {
        "target_col": target_col,
        "threshold": threshold,
        "train": train,
        "test": test,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "results": results,
    }


def get_random_forest_feature_importance(random_forest_result):
    model = random_forest_result["model"]
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return pd.Series(dtype=float)
    return pd.Series(importances, index=FEATURE_COLUMNS).sort_values(ascending=False)
