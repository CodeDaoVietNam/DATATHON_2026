"""
Experimental COGS/Revenue ratio models for notebook 04b.

These helpers are intentionally separated from the main forecasting pipeline.
They forecast the margin ratio directly, then convert final Revenue forecasts
to COGS. The target metric remains original-scale COGS MAE.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from feature_engineering import add_time_features, add_vn_calendar
from models import CATBOOST_PARAMS, CatBoostRegressor


RATIO_CLIP_DEFAULT = (0.70, 0.995)
RATIO_MIN_HISTORY = 90


def _prepare_ratio_history(
    sales: pd.DataFrame,
    ratio_clip: Tuple[float, float],
) -> pd.DataFrame:
    hist = sales[["Date", "Revenue", "COGS"]].copy()
    hist["Date"] = pd.to_datetime(hist["Date"])
    hist = hist.sort_values("Date").reset_index(drop=True)
    hist["cogs_ratio"] = hist["COGS"] / hist["Revenue"].replace(0, np.nan)
    hist = hist.replace([np.inf, -np.inf], np.nan).dropna(subset=["cogs_ratio"])
    hist["cogs_ratio"] = hist["cogs_ratio"].clip(*ratio_clip)
    return hist.reset_index(drop=True)


def _calendar_row(date: pd.Timestamp) -> pd.Series:
    row = pd.DataFrame({"Date": [pd.Timestamp(date)]})
    row = add_time_features(row, "Date")
    row = add_vn_calendar(row, "Date")
    return row.drop(columns=["Date"]).iloc[0]


def _ratio_feature_row(
    date: pd.Timestamp,
    prior: pd.DataFrame,
    revenue_value: float,
    monthly_revenue_median: pd.Series,
    monthly_ratio_median: pd.Series,
    global_ratio: float,
) -> pd.Series:
    date = pd.Timestamp(date)
    row = _calendar_row(date).astype(float)
    ratio = prior["cogs_ratio"].astype(float)

    for lag in (1, 7, 30):
        row[f"ratio_lag_{lag}"] = float(ratio.iloc[-lag]) if len(ratio) >= lag else global_ratio

    for window in (7, 30, 90):
        tail = ratio.tail(window)
        row[f"ratio_roll_mean_{window}"] = float(tail.mean()) if len(tail) else global_ratio
        row[f"ratio_roll_std_{window}"] = float(tail.std(ddof=0)) if len(tail) > 1 else 0.0
        row[f"ratio_roll_median_{window}"] = float(tail.median()) if len(tail) else global_ratio

    month = int(date.month)
    month_rev = float(monthly_revenue_median.get(month, np.nan))
    if not np.isfinite(month_rev) or month_rev <= 0:
        month_rev = float(prior["Revenue"].median())

    month_ratio = float(monthly_ratio_median.get(month, global_ratio))
    revenue_value = float(max(revenue_value, 0.0))

    row["ratio_month_anchor"] = month_ratio
    row["revenue_level_log"] = float(np.log1p(revenue_value))
    row["revenue_vs_month_median"] = revenue_value / (month_rev + 1e-8)
    row["revenue_vs_recent_30"] = revenue_value / (float(prior["Revenue"].tail(30).median()) + 1e-8)
    row["ratio_lag_1_vs_month"] = row["ratio_lag_1"] / (month_ratio + 1e-8)
    return row


def _build_ratio_training_frame(
    train: pd.DataFrame,
    ratio_clip: Tuple[float, float],
    min_history: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series, float]:
    hist = _prepare_ratio_history(train, ratio_clip)
    monthly_revenue_median = hist.assign(month=hist["Date"].dt.month).groupby("month")["Revenue"].median()
    monthly_ratio_median = hist.assign(month=hist["Date"].dt.month).groupby("month")["cogs_ratio"].median()
    global_ratio = float(hist["cogs_ratio"].median())

    rows = []
    y = []
    start = min(max(min_history, 30), max(len(hist) - 1, 1))
    for idx in range(start, len(hist)):
        prior = hist.iloc[:idx]
        current = hist.iloc[idx]
        rows.append(
            _ratio_feature_row(
                current["Date"],
                prior,
                current["Revenue"],
                monthly_revenue_median,
                monthly_ratio_median,
                global_ratio,
            )
        )
        y.append(float(current["cogs_ratio"]))

    X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    y = pd.Series(y, name="cogs_ratio")
    return X, y, hist, monthly_revenue_median, monthly_ratio_median, global_ratio


def _fit_ratio_models(X: pd.DataFrame, y: pd.Series, random_seed: int) -> Dict[str, object]:
    if CatBoostRegressor is None:
        models: Dict[str, object] = {}
    else:
        base_params = CATBOOST_PARAMS.copy()
        base_params.update(
            {
                "iterations": 900,
                "depth": 4,
                "learning_rate": 0.03,
                "l2_leaf_reg": 8.0,
                "random_seed": random_seed,
                "verbose": False,
            }
        )

        mae_params = base_params.copy()
        mae_params["loss_function"] = "MAE"

        quantile_params = base_params.copy()
        quantile_params["loss_function"] = "Quantile:alpha=0.60"

        models = {
            "ratio_cat_mae": CatBoostRegressor(**mae_params),
            "ratio_cat_quantile_060": CatBoostRegressor(**quantile_params),
        }

    models["ratio_elasticnet"] = make_pipeline(
        StandardScaler(),
        ElasticNet(alpha=0.0005, l1_ratio=0.15, max_iter=20000, random_state=random_seed),
    )

    X_np = X.to_numpy(dtype=np.float32)
    y_np = y.to_numpy(dtype=np.float32)
    fitted = {}
    for name, model in models.items():
        model.fit(X_np, y_np)
        fitted[name] = model
    return fitted


def _recursive_predict_ratio(
    model: object,
    feature_cols: Iterable[str],
    fill_values: pd.Series,
    hist: pd.DataFrame,
    revenue_pred: np.ndarray,
    future_dates: Iterable[pd.Timestamp],
    monthly_revenue_median: pd.Series,
    monthly_ratio_median: pd.Series,
    global_ratio: float,
    ratio_clip: Tuple[float, float],
) -> np.ndarray:
    feature_cols = list(feature_cols)
    state = hist[["Date", "Revenue", "cogs_ratio"]].copy()
    preds = []

    for date, revenue_value in zip(pd.to_datetime(future_dates), revenue_pred):
        row = _ratio_feature_row(
            date,
            state,
            float(revenue_value),
            monthly_revenue_median,
            monthly_ratio_median,
            global_ratio,
        )
        X_row = pd.DataFrame([row]).reindex(columns=feature_cols)
        X_row = X_row.replace([np.inf, -np.inf], np.nan).fillna(fill_values)
        pred = float(model.predict(X_row.to_numpy(dtype=np.float32))[0])
        pred = float(np.clip(pred, ratio_clip[0], ratio_clip[1]))
        preds.append(pred)
        state.loc[len(state)] = {
            "Date": pd.Timestamp(date),
            "Revenue": float(revenue_value),
            "cogs_ratio": pred,
        }

    return np.asarray(preds, dtype=float)


def forecast_cogs_ratio_specialized(
    train: pd.DataFrame,
    revenue_pred: np.ndarray,
    future_dates: Iterable[pd.Timestamp],
    ratio_clip: Tuple[float, float] = RATIO_CLIP_DEFAULT,
    min_history: int = RATIO_MIN_HISTORY,
    random_seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Train compact COGS/Revenue-ratio models and forecast future COGS.

    Returns a mapping:
        model_name -> {"cogs": cogs_pred, "ratio": ratio_pred}
    """
    revenue_pred = np.asarray(revenue_pred, dtype=float)
    X, y, hist, monthly_rev, monthly_ratio, global_ratio = _build_ratio_training_frame(
        train,
        ratio_clip=ratio_clip,
        min_history=min_history,
    )
    fill_values = X.median(numeric_only=True).fillna(0.0)
    X = X.fillna(fill_values)
    models = _fit_ratio_models(X, y, random_seed=random_seed)

    output: Dict[str, Dict[str, np.ndarray]] = {}
    for name, model in models.items():
        ratio_pred = _recursive_predict_ratio(
            model,
            X.columns,
            fill_values,
            hist,
            revenue_pred,
            future_dates,
            monthly_rev,
            monthly_ratio,
            global_ratio,
            ratio_clip,
        )
        cogs_pred = np.minimum(
            np.maximum(revenue_pred * ratio_pred, 0.0),
            np.maximum(revenue_pred, 0.0) * ratio_clip[1],
        )
        output[name] = {"cogs": cogs_pred, "ratio": ratio_pred}
    return output
