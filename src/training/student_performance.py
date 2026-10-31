"""Train, tune and save the student performance model.

Run from the project root:
    python -m src.training.student_performance
"""
import json
import logging

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.common import settings
from src.common.preprocessing_util import FEATURES, build_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def evaluate(model, X_test, y_test) -> dict:
    pred = np.clip(model.predict(X_test), 0, settings.MAX_SCORE)
    return {
        "mae": round(float(mean_absolute_error(y_test, pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 3),
        "r2": round(float(r2_score(y_test, pred)), 3),
    }


def main():
    df = pd.read_csv(settings.DATASET_PATH)
    X, y = df[FEATURES], df[settings.TARGET]        # explicit list: placement_status can never leak in
    log.info("Loaded %s rows, features: %s", len(df), FEATURES)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.TEST_SIZE, random_state=settings.RANDOM_STATE
    )

    baseline = Pipeline([("preprocess", build_preprocessor(scale=True)), ("model", LinearRegression())])
    baseline.fit(X_train, y_train)
    baseline_scores = evaluate(baseline, X_test, y_test)
    log.info("Baseline Linear Regression: %s", baseline_scores)

    with open(settings.HYPERPARAMS_PATH) as f:
        cfg = yaml.safe_load(f)["gradient_boosting"]

    pipe = Pipeline([
        ("preprocess", build_preprocessor(scale=False)),
        ("model", GradientBoostingRegressor(random_state=settings.RANDOM_STATE)),
    ])
    grid = GridSearchCV(pipe, cfg["param_grid"], cv=cfg["cv"],
                        scoring="neg_mean_absolute_error", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    log.info("Best params: %s | CV MAE: %.3f", grid.best_params_, -grid.best_score_)

    assert list(best.feature_names_in_) == FEATURES, "Model has unexpected input columns"
    tuned_scores = evaluate(best, X_test, y_test)
    log.info("Tuned Gradient Boosting: %s", tuned_scores)

    settings.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    settings.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, settings.MODEL_PATH)
    with open(settings.METRICS_PATH, "w") as f:
        json.dump({"baseline_linear_regression": baseline_scores,
                   "gradient_boosting_tuned": tuned_scores,
                   "best_params": grid.best_params_,
                   "cv_mae": round(float(-grid.best_score_), 3)}, f, indent=2)
    log.info("Saved model to %s", settings.MODEL_PATH)


if __name__ == "__main__":
    main()