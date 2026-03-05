import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from .preprocess import build_preprocessor
from .models import make_model


def build_pipeline(
    X_train: pd.DataFrame,
    *,
    model_name: str,
    random_state: int,
    n_estimators: int = 400,
    max_depth=None,
    # HGB defaults (ignored by other models)
    hgb_learning_rate: float = 0.05,
    hgb_max_iter: int = 500,
    hgb_max_depth: int = 6,
):
    preprocessor = build_preprocessor(X_train)

    model = make_model(
        model_name,
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        hgb_learning_rate=hgb_learning_rate,
        hgb_max_iter=hgb_max_iter,
        hgb_max_depth=hgb_max_depth,
    )

    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])


def evaluate_on_test(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipe.predict(X_test)

    metrics = {
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, digits=4, output_dict=True
        ),
    }

    if hasattr(pipe, "predict_proba"):
        y_proba = pipe.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    else:
        metrics["roc_auc"] = None

    return metrics


def cv_auc(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, *, random_state: int, n_splits: int = 5) -> dict:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipe, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)
    return {
        "cv_folds": n_splits,
        "cv_auc_mean": float(np.mean(scores)),
        "cv_auc_std": float(np.std(scores)),
        "cv_auc_all": [float(s) for s in scores],
    }


def tune_hgb_random_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int,
    n_iter: int = 25,
    cv_folds: int = 3,
) -> tuple[Pipeline, dict]:
    """
    RandomizedSearchCV for HistGradientBoosting inside the pipeline.
    Only use when model_name == 'hist_gradient_boost'.
    """
    base_pipe = build_pipeline(
        X_train,
        model_name="hist_gradient_boost",
        random_state=random_state,
    )

    param_dist = {
        "model__learning_rate": np.linspace(0.01, 0.1, 10),
        "model__max_depth": [3, 4, 6, 8, None],
        "model__max_iter": [300, 500, 800, 1000],
        "model__l2_regularization": [0.0, 0.1, 1.0, 5.0, 10.0],
        # Optional: control leaf size / regularization-ish behavior
        "model__min_samples_leaf": [20, 50, 100, 200],
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=base_pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
        verbose=0,
    )

    search.fit(X_train, y_train)

    best_pipe = search.best_estimator_
    tuning = {
        "search_type": "RandomizedSearchCV",
        "n_iter": n_iter,
        "cv_folds": cv_folds,
        "best_score_cv_auc": float(search.best_score_),
        "best_params": {k: (v if np.isscalar(v) else str(v)) for k, v in search.best_params_.items()},
    }
    return best_pipe, tuning


def train_eval_with_cv_and_tuning(
    df: pd.DataFrame,
    *,
    model_name: str,
    test_size: float,
    random_state: int,
    do_cv: bool = True,
    cv_folds: int = 5,
    do_tuning: bool = False,
    tuning_iter: int = 25,
    tuning_cv_folds: int = 3,
    n_estimators: int = 400,
    max_depth=None,
) -> tuple[Pipeline, dict]:

    y = df["no_show"].astype(int)
    X = df.drop(columns=["no_show"])

    # Holdout split for final report (kept constant)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() == 2 else None,
    )

    out = {
        "model": model_name,
        "test_size": test_size,
        "random_state": random_state,
        "cv": None,
        "tuning": None,
        "test": None,
    }

    # Build baseline pipeline
    pipe = build_pipeline(
        X_train,
        model_name=model_name,
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )

    # CV on full data (pipeline is refit each fold internally)
    if do_cv:
        out["cv"] = cv_auc(pipe, X, y, random_state=random_state, n_splits=cv_folds)

    # Optional tuning (only implemented for HGB for now)
    if do_tuning and model_name == "hist_gradient_boost":
        tuned_pipe, tuning_info = tune_hgb_random_search(
            X_train, y_train,
            random_state=random_state,
            n_iter=tuning_iter,
            cv_folds=tuning_cv_folds,
        )
        out["tuning"] = tuning_info
        pipe = tuned_pipe  # replace with tuned

    else:
        # Fit baseline on train
        pipe.fit(X_train, y_train)

    # Final evaluation on holdout
    out["test"] = evaluate_on_test(pipe, X_test, y_test)

    return pipe, out


def save_model(pipe: Pipeline, path: str) -> None:
    joblib.dump(pipe, path)