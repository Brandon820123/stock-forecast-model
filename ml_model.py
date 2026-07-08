"""Machine learning models for next trading day direction classification."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from baseline import train_test_split_time_series
from config import RANDOM_STATE, TEST_SIZE
from evaluate import evaluate_classification
from features import FEATURE_COLUMNS


def _fit_predict_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = evaluate_classification(y_test, predictions)
    return {
        "name": name,
        "model": model,
        "metrics": metrics,
        "y_true": y_test,
        "y_pred": predictions,
    }


def run_classification_models(feature_df, test_size=TEST_SIZE):
    train, test = train_test_split_time_series(feature_df, test_size=test_size)

    X_train = train[FEATURE_COLUMNS]
    y_train = train["label"].astype(int)
    X_test = test[FEATURE_COLUMNS]
    y_test = test["label"].astype(int)

    logistic = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    random_forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    results = [
        _fit_predict_model("Logistic Regression", logistic, X_train, y_train, X_test, y_test),
        _fit_predict_model("Random Forest", random_forest, X_train, y_train, X_test, y_test),
    ]

    return {
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

