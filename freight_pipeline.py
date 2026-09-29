"""
Consolidated freight rate ML pipeline — training, inference, and
template-filling, all in one script behind a single argparse CLI with
three subcommands:

    python freight_pipeline.py train --data train-test.csv --target posted_rate --output model.joblib
    python freight_pipeline.py predict --model model.joblib --data validation.csv --output validation_predictions.csv
    python freight_pipeline.py fill-template --model model.joblib --template december_chart_inputs.csv --output december_chart_inputs.csv

"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("freight_pipeline")

try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


ID_COLUMNS = ["load_id"]


# =============================================================================
# SECTION A — SHARED FEATURE ENGINEERING (used by both training & inference)
# =============================================================================

def load_data(path: str, target: str, column_map: Optional[dict] = None) -> pd.DataFrame:
    """Load a CSV, optionally rename columns, and drop rows missing the target."""
    df = pd.read_csv(path)
    if column_map:
        df = df.rename(columns=column_map)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Columns: {list(df.columns)}")
    before = len(df)
    df = df.dropna(subset=[target]).reset_index(drop=True)
    log.info("load_data: %d rows loaded, %d dropped for missing target", len(df), before - len(df))
    return df


def drop_id_columns(df: pd.DataFrame, id_columns: list = None) -> pd.DataFrame:
    """Remove identifier columns (e.g. load_id) that are never predictive."""
    id_columns = id_columns if id_columns is not None else ID_COLUMNS
    present = [c for c in id_columns if c in df.columns]
    if present:
        log.info("drop_id_columns: dropping %s", present)
    return df.drop(columns=present, errors="ignore")


def engineer_temporal_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Derive date-related features in two complementary forms:
    categorical (day_of_week, month) and numeric/cyclical (sin/cos
    encodings, is_weekend, is_month_start/end, is_peak_season)."""
    df = df.copy()
    if date_col in df.columns:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        dow = parsed.dt.dayofweek
        month_num = parsed.dt.month
        day_num = parsed.dt.day

        df["day_of_week"] = parsed.dt.day_name()
        df["month"] = month_num.astype("Int64").astype(str)
        df["is_weekend"] = dow.isin([5, 6]).astype(int)

        df["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7)
        df["month_sin"] = np.sin(2 * np.pi * month_num / 12)
        df["month_cos"] = np.cos(2 * np.pi * month_num / 12)

        df["is_month_start"] = parsed.dt.is_month_start.fillna(False).astype(int)
        df["is_month_end"] = parsed.dt.is_month_end.fillna(False).astype(int)

        is_peak = ((month_num == 11) & (day_num >= 15)) | (month_num == 12) | ((month_num == 1) & (day_num <= 5))
        df["is_peak_season"] = is_peak.fillna(False).astype(int)
    else:
        df["day_of_week"] = "unknown"
        df["month"] = "unknown"
        for col in ["is_weekend", "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos",
                    "is_month_start", "is_month_end", "is_peak_season"]:
            df[col] = 0
    return df


def haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized haversine distance in miles between two lat/lon arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 3958.8 * np.arcsin(np.sqrt(a))


def engineer_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive origin/destination region, haversine_distance, and
    route_circuity (route distance / straight-line distance)."""
    df = df.copy()
    for col, new_col in (("pickup", "origin_region"), ("delivery", "destination_region")):
        if col in df.columns:
            df[new_col] = df[col].astype(str).str.split(",").str[-1].str.strip()
        elif new_col not in df.columns:
            df[new_col] = "unknown"

    geo_cols = {"pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"}
    if geo_cols.issubset(df.columns):
        df["haversine_distance"] = haversine_miles(
            df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"]
        )
        if "distance" in df.columns:
            df["route_circuity"] = df["distance"] / df["haversine_distance"].replace(0, np.nan)
        else:
            df["route_circuity"] = np.nan
    else:
        df["haversine_distance"] = np.nan
        df["route_circuity"] = np.nan
    return df


def engineer_distance_buckets(df: pd.DataFrame, distance_col: str = "distance") -> pd.DataFrame:
    """Bucket raw distance into short/medium/long/very_long/cross_country."""
    df = df.copy()
    if distance_col in df.columns:
        df["distance_bucket"] = pd.cut(
            df[distance_col],
            bins=[-0.01, 250, 500, 1000, 2000, np.inf],
            labels=["short", "medium", "long", "very_long", "cross_country"],
        ).astype(str)
    else:
        df["distance_bucket"] = "unknown"
    return df


def engineer_features(df: pd.DataFrame, date_col: str = "date", distance_col: str = "distance") -> pd.DataFrame:
    """Run all feature engineering steps in sequence. Used identically
    at both training and inference time — this is what keeps the two
    from silently drifting apart."""
    df = engineer_temporal_features(df, date_col=date_col)
    df = engineer_geo_features(df)
    df = engineer_distance_buckets(df, distance_col=distance_col)
    return df


def get_feature_lists() -> tuple:
    """Return (numeric_features, categorical_features) column name lists."""
    numeric_features = [
        "distance", "weight", "market_index", "quote_signal",
        "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
        "haversine_distance", "route_circuity",
        "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos",
        "is_weekend", "is_month_start", "is_month_end", "is_peak_season",
    ]
    categorical_features = [
        "origin_region", "destination_region", "equipment",
        "day_of_week", "month", "distance_bucket",
    ]
    return numeric_features, categorical_features


def get_date_feature_names() -> tuple:
    """Return (date_numeric, date_categorical) — the date-derived subset
    of get_feature_lists(), used for importance summaries and ablation."""
    date_numeric = [
        "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos",
        "is_weekend", "is_month_start", "is_month_end", "is_peak_season",
    ]
    date_categorical = ["day_of_week", "month"]
    return date_numeric, date_categorical


def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """Impute+scale numerics, impute+one-hot categoricals."""
    numeric_pipe = Pipeline(steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[("num", numeric_pipe, numeric_features), ("cat", categorical_pipe, categorical_features)],
        remainder="drop",
    )


# =============================================================================
# SECTION B — TRAINING
# =============================================================================

def get_candidate_models(random_state: int = 42) -> dict:
    """Candidate regressors spanning several algorithm families —
    linear (Ridge, ElasticNet), bagged trees (RandomForest), and two
    different boosting implementations (GradientBoosting, which exposes
    feature_importances_, and HistGradientBoosting, which doesn't and
    needs permutation importance instead) — plus XGBoost/LightGBM if
    installed."""
    models = {
        "ridge_baseline": Ridge(alpha=1.0, random_state=random_state),
        "elastic_net": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state, max_iter=5000),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=random_state
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=random_state
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_depth=8, l2_regularization=0.1, random_state=random_state
        ),
    }
    if HAS_XGB:
        models["xgboost"] = XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, random_state=random_state, tree_method="hist",
        )
    if HAS_LGBM:
        models["lightgbm"] = LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=-1, num_leaves=63, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, random_state=random_state, verbosity=-1,
        )
    return models


def get_search_space(model_name: str) -> dict:
    """RandomizedSearchCV param grid for a given model name; unknown
    names get an empty dict (plain fit, no tuning)."""
    spaces = {
        "ridge_baseline": {"model__alpha": [0.1, 1.0, 5.0, 10.0, 50.0]},
        "elastic_net": {"model__alpha": [0.01, 0.1, 1.0, 5.0], "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
        "random_forest": {
            "model__n_estimators": [200, 300, 500, 800], "model__max_depth": [None, 8, 12, 20],
            "model__min_samples_leaf": [1, 2, 4], "model__max_features": ["sqrt", "log2", 0.6, 1.0],
        },
        "gradient_boosting": {
            "model__n_estimators": [200, 300, 500], "model__learning_rate": [0.02, 0.05, 0.1],
            "model__max_depth": [3, 4, 6], "model__subsample": [0.6, 0.8, 1.0],
        },
        "hist_gradient_boosting": {
            "model__max_iter": [200, 400, 600], "model__learning_rate": [0.02, 0.05, 0.1],
            "model__max_depth": [4, 6, 8, None], "model__l2_regularization": [0.0, 0.1, 0.5, 1.0],
        },
        "xgboost": {
            "model__n_estimators": [300, 500, 800], "model__learning_rate": [0.02, 0.05, 0.1],
            "model__max_depth": [4, 6, 8], "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0], "model__reg_lambda": [0.1, 1.0, 5.0],
        },
        "lightgbm": {
            "model__n_estimators": [300, 500, 800], "model__learning_rate": [0.02, 0.05, 0.1],
            "model__num_leaves": [31, 63, 127], "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0], "model__reg_lambda": [0.1, 1.0, 5.0],
        },
    }
    return spaces.get(model_name, {})


def compute_iqr_bounds(y: pd.Series, k: float = 1.5) -> tuple:
    """Return (lower, upper) Tukey IQR bounds for a target series."""
    q1, q3 = y.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def treat_target_outliers(X_train: pd.DataFrame, y_train: pd.Series, method: str = "none", k: float = 1.5) -> tuple:
    """Apply outlier treatment to the TARGET, fit only on the training
    split. method: "none" (default) | "clip" (winsorize) | "remove"."""
    if method == "none":
        return X_train, y_train
    if method not in {"clip", "remove"}:
        raise ValueError(f"Unknown outlier method: {method!r} (expected 'none', 'clip', or 'remove')")
    lower, upper = compute_iqr_bounds(y_train, k)
    n_flagged = ((y_train < lower) | (y_train > upper)).sum()
    log.info("treat_target_outliers: method=%s, IQR bounds=[%.4f, %.4f], %d/%d rows flagged",
              method, lower, upper, n_flagged, len(y_train))
    if method == "clip":
        return X_train, y_train.clip(lower, upper)
    mask = (y_train >= lower) & (y_train <= upper)
    return X_train[mask], y_train[mask]


def split_data(df: pd.DataFrame, feature_columns: list, target: str, test_size: float = 0.2, random_state: int = 42):
    """Split df into X_train, X_test, y_train, y_test."""
    X = df[feature_columns]
    y = df[target].astype(float)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def compare_models_cv(models: dict, preprocessor: ColumnTransformer, X_train: pd.DataFrame,
                       y_train: pd.Series, cv_folds: int = 5, random_state: int = 42) -> dict:
    """Cross-validate each candidate model, return {name: mean_MAE}."""
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results = {}
    for name, model in models.items():
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=kfold, scoring="neg_mean_absolute_error", n_jobs=-1)
        results[name] = -scores.mean()
        log.info("compare_models_cv: %-24s CV MAE = %.4f (+/- %.4f)", name, -scores.mean(), scores.std())
    return results


def select_best_model(cv_results: dict) -> str:
    """Return the model name with the lowest CV MAE."""
    return min(cv_results, key=cv_results.get)


def tune_model(model, preprocessor: ColumnTransformer, param_space: dict, X_train: pd.DataFrame,
               y_train: pd.Series, cv_folds: int = 5, n_iter: int = 25, random_state: int = 42):
    """Fit a RandomizedSearchCV (if param_space given) or a plain fit, return the fitted pipeline."""
    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    if not param_space:
        pipe.fit(X_train, y_train)
        return pipe
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        pipe, param_distributions=param_space, n_iter=n_iter, cv=kfold,
        scoring="neg_mean_absolute_error", random_state=random_state, n_jobs=-1, verbose=0,
    )
    search.fit(X_train, y_train)
    log.info("tune_model: best params = %s", json.dumps(search.best_params_, default=str))
    return search.best_estimator_


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, MAPE_%, R2."""
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "MAPE_%": round(mean_absolute_percentage_error(y_true, y_pred) * 100, 3),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


def get_feature_importance(fitted_pipeline: Pipeline, top_n: int = 15, X_val: pd.DataFrame = None,
                            y_val: pd.Series = None, random_state: int = 42) -> list:
    """Return [(feature_name, importance), ...] sorted descending.
    Falls back to permutation importance (needs X_val/y_val) for models
    like HistGradientBoostingRegressor that expose neither
    feature_importances_ nor coef_."""
    try:
        feature_names = fitted_pipeline.named_steps["prep"].get_feature_names_out()
        model_step = fitted_pipeline.named_steps["model"]
        if hasattr(model_step, "feature_importances_"):
            importances = model_step.feature_importances_
        elif hasattr(model_step, "coef_"):
            importances = np.abs(model_step.coef_)
        elif X_val is not None and y_val is not None:
            from sklearn.inspection import permutation_importance
            log.info("get_feature_importance: model has no feature_importances_/coef_ — using permutation importance")
            result = permutation_importance(fitted_pipeline, X_val, y_val, n_repeats=10, random_state=random_state, n_jobs=-1)
            importances = result.importances_mean
        else:
            log.warning("get_feature_importance: no feature_importances_/coef_ and no X_val/y_val passed — returning []")
            return []
        return sorted(zip(feature_names, importances), key=lambda t: -t[1])[:top_n]
    except Exception as e:  # noqa: BLE001
        log.warning("get_feature_importance: could not extract importances: %s", e)
        return []


def is_date_feature(encoded_name: str, date_numeric: list, date_categorical: list) -> bool:
    """Check whether an encoded (post-ColumnTransformer) feature name is date-derived."""
    if encoded_name.startswith("num__"):
        return encoded_name[len("num__"):] in date_numeric
    if encoded_name.startswith("cat__"):
        base = encoded_name[len("cat__"):]
        return any(base == dc or base.startswith(dc + "_") for dc in date_categorical)
    return False


def summarize_date_feature_importance(importance_list: list) -> dict:
    """Filter a full importance list down to just date-derived features and total them."""
    date_numeric, date_categorical = get_date_feature_names()
    date_items = [(name, val) for name, val in importance_list if is_date_feature(name, date_numeric, date_categorical)]
    date_items.sort(key=lambda t: -t[1])
    return {
        "date_features": date_items,
        "total_importance": sum(v for _, v in date_items),
        "top_date_feature": date_items[0] if date_items else None,
    }


def evaluate_date_feature_contribution(model, numeric_features: list, categorical_features: list,
                                        X_train: pd.DataFrame, y_train: pd.Series,
                                        X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Fit the SAME model twice — with vs. without date features — and
    compare test metrics directly. Positive mae_improvement_from_date
    means date features are measurably helping."""
    date_numeric, date_categorical = get_date_feature_names()
    y_test_arr = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)

    prep_with = build_preprocessor(numeric_features, categorical_features)
    pipe_with = Pipeline(steps=[("prep", prep_with), ("model", clone(model))])
    pipe_with.fit(X_train, y_train)
    metrics_with = evaluate_model(y_test_arr, pipe_with.predict(X_test))

    reduced_numeric = [c for c in numeric_features if c not in date_numeric]
    reduced_categorical = [c for c in categorical_features if c not in date_categorical]
    prep_without = build_preprocessor(reduced_numeric, reduced_categorical)
    pipe_without = Pipeline(steps=[("prep", prep_without), ("model", clone(model))])
    pipe_without.fit(X_train, y_train)
    metrics_without = evaluate_model(y_test_arr, pipe_without.predict(X_test))

    result = {
        "with_date_features": metrics_with,
        "without_date_features": metrics_without,
        "mae_improvement_from_date": round(metrics_without["MAE"] - metrics_with["MAE"], 4),
        "r2_improvement_from_date": round(metrics_with["R2"] - metrics_without["R2"], 4),
    }
    log.info(
        "evaluate_date_feature_contribution: with-date MAE=%.4f  without-date MAE=%.4f  improvement=%.4f (%s)",
        metrics_with["MAE"], metrics_without["MAE"], result["mae_improvement_from_date"],
        "date HELPS" if result["mae_improvement_from_date"] > 0.01 else "date has little/no effect",
    )
    return result


def save_model(fitted_pipeline: Pipeline, output_path: str) -> None:
    """Persist a fitted pipeline to disk via joblib."""
    import joblib
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_pipeline, output_path)
    log.info("save_model: saved to %s", output_path)


def save_metrics(metrics: dict, model_name: str, output_path: str, extra: dict = None) -> str:
    """Write metrics JSON next to the model file. Returns the metrics file path."""
    payload = {"model": model_name, "test_metrics": metrics}
    if extra:
        payload.update(extra)
    metrics_path = str(Path(output_path).with_suffix(".metrics.json"))
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    log.info("save_metrics: saved to %s", metrics_path)
    return metrics_path


def run_pipeline(data_path: str, target: str = "posted_rate", output_path: str = "freight_rate_model.joblib",
                  column_map: Optional[dict] = None, test_size: float = 0.2, cv_folds: int = 5,
                  random_state: int = 42, outlier_method: str = "none", outlier_k: float = 1.5,
                  run_date_ablation: bool = True) -> dict:
    """Train end to end: load -> engineer features -> compare models ->
    tune the winner -> evaluate -> (optionally) analyze date-feature
    contribution -> save. Returns the test-set metrics dict."""
    df = load_data(data_path, target, column_map)
    df = drop_id_columns(df)
    df = engineer_features(df)

    numeric_features, categorical_features = get_feature_lists()
    feature_columns = numeric_features + categorical_features
    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan

    X_train, X_test, y_train, y_test = split_data(df, feature_columns, target, test_size=test_size, random_state=random_state)
    log.info("run_pipeline: train/test split = %d / %d rows", len(X_train), len(X_test))

    X_train, y_train = treat_target_outliers(X_train, y_train, method=outlier_method, k=outlier_k)
    if outlier_method != "none":
        log.info("run_pipeline: %d rows in training set after outlier treatment", len(X_train))

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models = get_candidate_models(random_state=random_state)
    log.info("run_pipeline: comparing %d candidate models: %s", len(models), list(models.keys()))

    cv_results = compare_models_cv(models, preprocessor, X_train, y_train, cv_folds=cv_folds, random_state=random_state)
    best_name = select_best_model(cv_results)
    log.info("run_pipeline: best candidate = %s", best_name)

    param_space = get_search_space(best_name)
    final_model = tune_model(models[best_name], preprocessor, param_space, X_train, y_train, cv_folds=cv_folds, random_state=random_state)

    y_pred = final_model.predict(X_test)
    metrics = evaluate_model(y_test.values, y_pred)
    log.info("run_pipeline: test metrics = %s", metrics)

    top_features = get_feature_importance(final_model, top_n=15, X_val=X_test, y_val=y_test, random_state=random_state)
    if top_features:
        log.info("run_pipeline: top features:\n%s", "\n".join(f"  {n:35s} {v:.4f}" for n, v in top_features))

    date_summary = None
    date_ablation = None
    if run_date_ablation:
        full_importance = get_feature_importance(final_model, top_n=10_000, X_val=X_test, y_val=y_test, random_state=random_state)
        date_summary = summarize_date_feature_importance(full_importance)
        log.info("run_pipeline: date-feature importance total=%.4f, top date feature=%s",
                  date_summary["total_importance"], date_summary["top_date_feature"])
        date_ablation = evaluate_date_feature_contribution(
            models[best_name], numeric_features, categorical_features, X_train, y_train, X_test, y_test
        )

    save_model(final_model, output_path)
    save_metrics(metrics, best_name, output_path,
                 extra={"date_feature_importance": date_summary, "date_ablation": date_ablation} if run_date_ablation else None)
    return metrics


# =============================================================================
# SECTION C — INFERENCE (predict on new data / fill a fixed-format template)
# =============================================================================

def load_model(model_path: str):
    """Load a fitted sklearn Pipeline saved via save_model/joblib."""
    import joblib
    model = joblib.load(model_path)
    log.info("load_model: loaded pipeline from %s", model_path)
    return model


def load_new_data(data_path: str, column_map: Optional[dict] = None) -> pd.DataFrame:
    """Load a CSV of new loads to score. No target-column requirement —
    new data has no posted_rate yet, that's what we're predicting."""
    df = pd.read_csv(data_path)
    if column_map:
        df = df.rename(columns=column_map)
    log.info("load_new_data: loaded %d rows to score", len(df))
    return df


def prepare_inference_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact same drop_id_columns + engineer_features steps
    used at training time, then ensure every expected feature column
    exists (NaN-filled if genuinely absent)."""
    df = drop_id_columns(df)
    df = engineer_features(df)
    numeric_features, categorical_features = get_feature_lists()
    feature_columns = numeric_features + categorical_features
    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan
    return df[feature_columns]


def predict_rates(model, X: pd.DataFrame) -> np.ndarray:
    """Run the fitted pipeline's .predict on prepared feature columns."""
    preds = model.predict(X)
    log.info("predict_rates: generated %d predictions", len(preds))
    return preds


def attach_predictions(original_df: pd.DataFrame, predictions: np.ndarray, id_column: str = "load_id") -> pd.DataFrame:
    """Build an output DataFrame of [id_column, predicted_rate],
    falling back to a row index if id_column isn't present."""
    if id_column in original_df.columns:
        ids = original_df[id_column].reset_index(drop=True)
    else:
        ids = pd.Series(range(len(original_df)), name="row_index")
    return pd.DataFrame({ids.name: ids, "predicted_rate": predictions})


def save_predictions(predictions_df: pd.DataFrame, output_path: str) -> None:
    """Write predictions to CSV."""
    predictions_df.to_csv(output_path, index=False)
    log.info("save_predictions: wrote %d rows to %s", len(predictions_df), output_path)


def run_prediction(model_path: str, data_path: str, output_path: str = "predictions.csv",
                    column_map: Optional[dict] = None, id_column: str = "load_id") -> pd.DataFrame:
    """Full inference flow, output REDUCED to [id, predicted_rate].
    Use this for a standard scored-output file (e.g. validation_predictions.csv)."""
    model = load_model(model_path)
    raw_df = load_new_data(data_path, column_map)
    X = prepare_inference_features(raw_df)
    preds = predict_rates(model, X)
    result = attach_predictions(raw_df, preds, id_column=id_column)
    save_predictions(result, output_path)
    return result


def fill_template_predictions(model_path: str, template_path: str, output_path: str,
                               target_column: str = "predicted_rate") -> pd.DataFrame:
    """Fill an existing template CSV's target column with predictions,
    IN PLACE — every other original column and the original column
    order are preserved exactly. Use this (instead of run_prediction)
    whenever a downstream consumer — a scorer, a chart, a fixed
    submission format — requires the original columns to survive
    untouched (e.g. december_chart_inputs.csv)."""
    model = load_model(model_path)
    template_df = load_new_data(template_path)
    if target_column not in template_df.columns:
        raise ValueError(f"Template is missing its own '{target_column}' column to fill. Columns present: {list(template_df.columns)}")
    X = prepare_inference_features(template_df)
    preds = predict_rates(model, X)
    result = template_df.copy()
    result[target_column] = preds  # overwrite in place — column position preserved
    save_predictions(result, output_path)
    log.info("fill_template_predictions: filled '%s' on %d rows, columns unchanged: %s",
              target_column, len(result), list(result.columns))
    return result


# =============================================================================
# SECTION D — CLI (the only place argparse / sys.argv is touched)
# =============================================================================

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freight rate ML pipeline — train, predict, or fill a fixed-format template."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train", help="Train a model and save it")
    p_train.add_argument("--data", required=True, help="Path to training CSV (must include the target column)")
    p_train.add_argument("--target", default="posted_rate", help="Target column name")
    p_train.add_argument("--output", default="freight_rate_model.joblib", help="Path to save the trained model")
    p_train.add_argument("--column-map", default=None, help='JSON string mapping raw column names, e.g. \'{"src":"origin"}\'')
    p_train.add_argument("--outlier-method", default="none", choices=["none", "clip", "remove"],
                          help="Target outlier treatment on the training split only (default: none)")
    p_train.add_argument("--skip-date-ablation", action="store_true",
                          help="Skip the date-feature importance summary and with/without-date ablation (faster run)")

    p_predict = subparsers.add_parser("predict", help="Score new data, output reduced to [id, predicted_rate]")
    p_predict.add_argument("--model", required=True, help="Path to the saved .joblib pipeline")
    p_predict.add_argument("--data", required=True, help="Path to new-data CSV to score")
    p_predict.add_argument("--output", default="predictions.csv", help="Path to write predictions CSV")
    p_predict.add_argument("--id-column", default="load_id", help="Identifier column to carry through to the output")
    p_predict.add_argument("--column-map", default=None, help='JSON string mapping raw column names')

    p_fill = subparsers.add_parser("fill-template", help="Fill a fixed-format template's prediction column in place")
    p_fill.add_argument("--model", required=True, help="Path to the saved .joblib pipeline")
    p_fill.add_argument("--template", required=True, help="Path to the template CSV to fill")
    p_fill.add_argument("--output", required=True, help="Path to write the filled template CSV")
    p_fill.add_argument("--target-column", default="predicted_rate", help="Column name to fill")

    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        if args.command == "train":
            col_map = json.loads(args.column_map) if args.column_map else None
            run_pipeline(
                args.data, args.target, args.output, col_map,
                outlier_method=args.outlier_method,
                run_date_ablation=not args.skip_date_ablation,
            )
        elif args.command == "predict":
            col_map = json.loads(args.column_map) if args.column_map else None
            run_prediction(args.model, args.data, args.output, col_map, args.id_column)
        elif args.command == "fill-template":
            fill_template_predictions(args.model, args.template, args.output, target_column=args.target_column)
    except Exception as exc:  # noqa: BLE001
        log.error("%s failed: %s", args.command, exc)
        sys.exit(1)


if __name__ == "__main__":
    main()