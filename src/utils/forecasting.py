"""
forecasting_v2.py
=================
Leakage-aware forecasting utilities for the Datathon revenue task.

The key difference from the notebook-first pipeline is inference:
tree models are rolled forward one day at a time, so lag and rolling
features for future dates are based on prior predictions instead of
global bfill/ffill.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
try:
    import lightgbm as lgb
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    lgb = None

try:
    import xgboost as xgb
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    xgb = None

try:
    from catboost import CatBoostRegressor
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    CatBoostRegressor = None

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import re

from feature_engineering import SALE_EVENT_DAYS, TET_DATES, add_time_features, add_vn_calendar


RANDOM_SEED = 42
RECURSIVE_BASELINE_BLEND = 0.10
RECURSIVE_EWM_ALPHA = 0.85

LAGS = [1, 2, 3, 7, 14, 21, 28, 30, 60, 90, 180, 358, 364, 365]
ROLL_WINDOWS = [7, 14, 30, 60, 90]
BUSINESS_LAGS = [1, 7, 14, 30]
BUSINESS_ROLL_WINDOWS = [7, 14, 30]

BUSINESS_CONFIGS = {
    "V0_current": [],
    "V1_orders_items": ["orders_items"],
    "V2_payments_mix": ["orders_items", "payments"],
    "V3_returns_reviews": ["orders_items", "payments", "returns_reviews"],
    "V4_customers_shipments": ["orders_items", "payments", "customers_shipments"],
    "V5_all_business": ["orders_items", "payments", "returns_reviews", "customers_shipments"],
}

LGBM_V2_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "n_estimators": 5000,
    "learning_rate": 0.02,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 15,
    "feature_fraction": 0.80,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

XGB_V2_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 4000,
    "learning_rate": 0.02,
    "max_depth": 7,
    "subsample": 0.85,
    "colsample_bytree": 0.80,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "tree_method": "hist",
}

CATBOOST_V2_PARAMS = {
    "loss_function": "MAE",
    "iterations": 2000,
    "learning_rate": 0.02,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "subsample": 0.85,
    "random_seed": RANDOM_SEED,
    "verbose": False,
}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_pred = np.maximum(y_pred, 0)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _geometric_growth(series: pd.Series) -> float:
    """Estimate a stable multiplicative growth factor from annual history."""
    clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    clean = clean[clean > 0]
    if len(clean) < 2:
        return 1.0

    growth = clean.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if growth.empty:
        return 1.0

    factor = float((1.0 + growth).prod() ** (1.0 / len(growth)))
    return factor if np.isfinite(factor) and factor > 0 else 1.0


def _linear_trend(values: pd.Series, x: pd.Series) -> float:
    """Estimate a simple per-period additive trend for future-safe profiles."""
    data = pd.DataFrame({"x": x, "y": values}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 2:
        return 0.0

    slope = np.polyfit(data["x"].to_numpy(dtype=float), data["y"].to_numpy(dtype=float), 1)[0]
    return float(slope) if np.isfinite(slope) else 0.0


def _safe_feature_token(value: object) -> str:
    """Convert category values to stable ASCII-ish feature name tokens."""
    token = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower()).strip("_")
    return token or "unknown"


def predict_cogs(
    revenue_pred: np.ndarray,
    dates: pd.Series,
    historical_train: pd.DataFrame,
) -> np.ndarray:
    hist = historical_train.copy()
    hist["month"] = hist["Date"].dt.month
    hist["cogs_ratio"] = hist["COGS"] / hist["Revenue"].replace(0, np.nan)
    monthly_ratio = hist.groupby("month")["cogs_ratio"].median().to_dict()
    global_ratio = hist["cogs_ratio"].median()
    test_months = pd.to_datetime(dates).dt.month.values
    ratios = np.array([monthly_ratio.get(m, global_ratio) for m in test_months])
    return np.maximum(revenue_pred * ratios, 0)


@dataclass
class ForecastArtifacts:
    feature_cols: List[str]
    holdout_scores: pd.DataFrame
    primary_model: object
    secondary_model: object
    ensemble_weights: Dict[str, float]


def fix_revenue_cogs_swaps(sales: pd.DataFrame) -> pd.DataFrame:
    """Swap Revenue/COGS on rows that violate COGS <= Revenue."""
    fixed = sales.copy()
    mask = fixed["COGS"] > fixed["Revenue"]
    fixed.loc[mask, ["Revenue", "COGS"]] = fixed.loc[mask, ["COGS", "Revenue"]].to_numpy()
    return fixed


DEFAULT_EVENT_CALIBRATION_STRENGTHS = {
    "tet": 0.30,
    "pre_tet": 0.10,
    "post_tet": 0.00,
    "pre_sale_event": 0.25,
    "sale_event_day": 0.05,
    "post_sale_event": 0.00,
    "q1_end_spike": 0.20,
    "aug_end_spike": 0.20,
    "back2school_extended": 0.10,
}


def _event_bucket_masks(dates: Sequence[pd.Timestamp | str]) -> Dict[str, pd.Series]:
    """Return mutually interpretable known-future spike buckets."""
    d = pd.Series(pd.to_datetime(list(dates)))
    if d.empty:
        keys = [
            "tet",
            "pre_tet",
            "post_tet",
            "pre_sale_event",
            "sale_event_day",
            "post_sale_event",
            "q1_end_spike",
            "aug_end_spike",
            "back2school_extended",
        ]
        return {key: pd.Series(False, index=d.index) for key in keys}

    masks: Dict[str, pd.Series] = {
        "tet": pd.Series(False, index=d.index),
        "pre_tet": pd.Series(False, index=d.index),
        "post_tet": pd.Series(False, index=d.index),
        "pre_sale_event": pd.Series(False, index=d.index),
        "sale_event_day": pd.Series(False, index=d.index),
        "post_sale_event": pd.Series(False, index=d.index),
        "q1_end_spike": ((d.dt.month == 3) & (d.dt.day >= 25)),
        "aug_end_spike": ((d.dt.month == 8) & (d.dt.day >= 25)),
        "back2school_extended": (
            ((d.dt.month == 8) & (d.dt.day >= 25))
            | ((d.dt.month == 9) & (d.dt.day <= 10))
        ),
    }

    for year in range(int(d.dt.year.min()) - 1, int(d.dt.year.max()) + 2):
        for month, day in SALE_EVENT_DAYS:
            event_date = pd.Timestamp(year=year, month=month, day=day)
            masks["pre_sale_event"] |= (d >= event_date - pd.Timedelta(days=3)) & (d < event_date)
            masks["sale_event_day"] |= d.eq(event_date)
            masks["post_sale_event"] |= (d > event_date) & (d <= event_date + pd.Timedelta(days=2))

    for tet_date in TET_DATES:
        masks["pre_tet"] |= (d >= tet_date - pd.Timedelta(days=14)) & (d < tet_date)
        masks["tet"] |= (d >= tet_date) & (d <= tet_date + pd.Timedelta(days=3))
        masks["post_tet"] |= (d > tet_date + pd.Timedelta(days=3)) & (d <= tet_date + pd.Timedelta(days=7))

    return masks


def _event_window_flags(dates: Sequence[pd.Timestamp | str], include_tet: bool = True) -> pd.Series:
    """Return known-future event windows where revenue spikes are expected."""
    masks = _event_bucket_masks(dates)
    include = ["pre_sale_event", "sale_event_day", "post_sale_event", "q1_end_spike", "aug_end_spike", "back2school_extended"]
    if include_tet:
        include += ["pre_tet", "tet", "post_tet"]
    flags = pd.Series(False, index=pd.RangeIndex(len(pd.Series(pd.to_datetime(list(dates))))))
    for key in include:
        flags |= masks[key].reset_index(drop=True)
    return flags


def compute_event_sample_weights(
    dates: Sequence[pd.Timestamp | str],
    base_weight: float = 1.0,
    sale_weight: float = 2.0,
    sale_day_weight: float = 1.4,
    tet_weight: float = 2.5,
    q1_end_weight: float = 2.2,
    aug_end_weight: float = 2.0,
    back2school_weight: float = 1.5,
) -> np.ndarray:
    """Weight known peak windows higher for tree training."""
    masks = _event_bucket_masks(dates)
    weights = np.full(len(pd.Series(pd.to_datetime(list(dates)))), float(base_weight), dtype=float)

    weight_map = {
        "pre_sale_event": sale_weight,
        "sale_event_day": sale_day_weight,
        "tet": tet_weight,
        "pre_tet": tet_weight,
        "q1_end_spike": q1_end_weight,
        "aug_end_spike": aug_end_weight,
        "back2school_extended": back2school_weight,
    }
    for key, weight in weight_map.items():
        flags = masks[key].to_numpy(dtype=bool)
        weights[flags] = np.maximum(weights[flags], float(weight))

    return weights


def compute_recency_sample_weights(
    dates: Sequence[pd.Timestamp | str],
    early_weight: float = 0.65,
    transition_weight: float = 0.90,
    recent_weight: float = 1.25,
) -> np.ndarray:
    """Weight recent years higher without discarding older seasonality."""
    d = pd.Series(pd.to_datetime(list(dates)))
    weights = np.full(len(d), float(early_weight), dtype=float)
    if d.empty:
        return weights

    years = d.dt.year.to_numpy()
    weights[(years >= 2019) & (years <= 2020)] = float(transition_weight)
    weights[years >= 2021] = float(recent_weight)
    return weights


def combine_sample_weights(*weights: Sequence[float]) -> np.ndarray:
    """Multiply sample-weight vectors and normalize their mean to 1.0."""
    clean = [np.asarray(w, dtype=float) for w in weights if w is not None]
    if not clean:
        return np.array([], dtype=float)

    n = len(clean[0])
    combined = np.ones(n, dtype=float)
    for w in clean:
        if len(w) != n:
            raise ValueError("All sample-weight vectors must have the same length")
        combined *= np.where(np.isfinite(w), w, 1.0)

    mean_weight = float(np.nanmean(combined))
    if np.isfinite(mean_weight) and mean_weight > 0:
        combined = combined / mean_weight
    return combined


def _normalize_event_strengths(strength: float | Dict[str, float]) -> Dict[str, float]:
    if isinstance(strength, dict):
        merged = DEFAULT_EVENT_CALIBRATION_STRENGTHS.copy()
        merged.update({k: float(v) for k, v in strength.items()})
        return merged
    scalar = float(strength)
    return {k: scalar for k in DEFAULT_EVENT_CALIBRATION_STRENGTHS}


def calibrate_event_spikes(
    preds: np.ndarray,
    dates: Sequence[pd.Timestamp | str],
    historical_train: pd.DataFrame,
    target_col: str = "Revenue",
    strength: float | Dict[str, float] = DEFAULT_EVENT_CALIBRATION_STRENGTHS,
    min_factor: float = 0.90,
    max_factor: float = 1.45,
) -> np.ndarray:
    """
    Apply a mild, future-safe event uplift learned from historical event windows.

    The multiplier is estimated from known sale/Tet windows relative to same-month
    non-event days, then blended into predictions only on known event windows.
    """
    strengths = _normalize_event_strengths(strength)
    if max(strengths.values(), default=0.0) <= 0:
        return np.asarray(preds, dtype=float)

    pred = np.asarray(preds, dtype=float).copy()
    out_dates = pd.Series(pd.to_datetime(list(dates)))
    out_masks = _event_bucket_masks(out_dates)
    if not any(mask.any() and strengths.get(bucket, 0.0) > 0 for bucket, mask in out_masks.items()):
        return np.maximum(pred, 0.0)

    hist = historical_train[["Date", target_col]].copy().reset_index(drop=True)
    hist["Date"] = pd.to_datetime(hist["Date"])
    hist["month"] = hist["Date"].dt.month
    hist_masks = _event_bucket_masks(hist["Date"])
    hist["is_any_event_window"] = False
    for mask in hist_masks.values():
        hist["is_any_event_window"] |= mask.to_numpy(dtype=bool)

    global_non_event = hist.loc[~hist["is_any_event_window"], target_col].median()
    if not np.isfinite(global_non_event) or global_non_event <= 0:
        global_non_event = hist[target_col].median()

    # Use one calibration bucket per day. This prevents overlapping windows
    # such as Aug-end and back-to-school from multiplying the same prediction.
    bucket_priority = [
        "tet",
        "pre_tet",
        "sale_event_day",
        "pre_sale_event",
        "post_sale_event",
        "post_tet",
        "q1_end_spike",
        "aug_end_spike",
        "back2school_extended",
    ]
    calibrated = np.zeros(len(pred), dtype=bool)

    for bucket in bucket_priority:
        out_mask = out_masks[bucket]
        bucket_strength = strengths.get(bucket, 0.0)
        if bucket_strength <= 0 or not out_mask.any():
            continue

        hist_mask = hist_masks[bucket].to_numpy(dtype=bool)
        hist_bucket = hist.loc[hist_mask].copy()
        if len(hist_bucket) < 2:
            continue

        month_factors: Dict[int, float] = {}
        for month, month_df in hist.groupby("month"):
            month_bucket = hist_bucket[hist_bucket["month"] == month]
            if len(month_bucket) < 1:
                continue
            non_bucket = month_df.loc[~pd.Series(hist_mask, index=hist.index).loc[month_df.index], target_col]
            baseline = non_bucket.median() if len(non_bucket) else global_non_event
            if not np.isfinite(baseline) or baseline <= 0:
                baseline = global_non_event
            factor = float(month_bucket[target_col].median() / max(baseline, 1e-8))
            month_factors[int(month)] = float(np.clip(factor, min_factor, max_factor))

        if not month_factors:
            continue
        months = out_dates.dt.month.to_numpy()
        factors = np.array([month_factors.get(int(m), 1.0) for m in months], dtype=float)
        blended_factors = 1.0 + float(bucket_strength) * (factors - 1.0)
        flags = out_mask.to_numpy(dtype=bool) & ~calibrated
        pred[flags] = pred[flags] * blended_factors[flags]
        calibrated[flags] = True

    return np.maximum(pred, 0.0)


def build_promo_profile(
    promotions: pd.DataFrame,
    train_cutoff: Optional[pd.Timestamp | str] = None,
) -> pd.DataFrame:
    """
    Build a reusable month/day promo profile from historical promotions.

    Future promotion schedules are not provided, so direct promo lookup would
    produce all-zero test features. A month/day profile keeps the historical
    recurring sale rhythm without leaking unknown future campaign IDs.
    """
    if promotions.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "day",
                "promo_profile_count",
                "promo_profile_discount",
                "promo_profile_stackable",
                "promo_profile_pct_count",
                "promo_profile_fixed_count",
            ]
        )

    work = promotions.copy()
    work["start_date"] = pd.to_datetime(work["start_date"])
    work["end_date"] = pd.to_datetime(work["end_date"])
    if train_cutoff is not None:
        cutoff = pd.Timestamp(train_cutoff)
        work = work[work["start_date"] <= cutoff].copy()
        work["end_date"] = work["end_date"].clip(upper=cutoff)
    if work.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "day",
                "promo_profile_count",
                "promo_profile_discount",
                "promo_profile_stackable",
                "promo_profile_pct_count",
                "promo_profile_fixed_count",
            ]
        )

    start = work["start_date"].min()
    end = work["end_date"].max()
    calendar = pd.DataFrame({"Date": pd.date_range(start, end, freq="D")})

    rows = []
    for d in calendar["Date"]:
        active = work[
            (work["start_date"] <= d)
            & (work["end_date"] >= d)
        ]
        rows.append(
            {
                "month": d.month,
                "day": d.day,
                "promo_profile_count": float(len(active)),
                "promo_profile_discount": float(active["discount_value"].max()) if len(active) else 0.0,
                "promo_profile_stackable": float(active["stackable_flag"].max()) if len(active) else 0.0,
                "promo_profile_pct_count": float((active["promo_type"] == "percentage").sum()) if len(active) else 0.0,
                "promo_profile_fixed_count": float((active["promo_type"] == "fixed").sum()) if len(active) else 0.0,
            }
        )

    profile = (
        pd.DataFrame(rows)
        .groupby(["month", "day"], as_index=False)
        .agg(
            promo_profile_count=("promo_profile_count", "mean"),
            promo_profile_discount=("promo_profile_discount", "mean"),
            promo_profile_stackable=("promo_profile_stackable", "mean"),
            promo_profile_pct_count=("promo_profile_pct_count", "mean"),
            promo_profile_fixed_count=("promo_profile_fixed_count", "mean"),
        )
    )
    return profile


def build_traffic_profile(
    web_traffic: pd.DataFrame,
    use_growth_adjustment: bool = False,
    train_cutoff: Optional[pd.Timestamp | str] = None,
) -> pd.DataFrame:
    """Build future-safe traffic profiles from recurring date patterns plus growth."""
    if web_traffic.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "day",
                "day_of_week",
                "traffic_profile_sessions_mdow",
                "traffic_profile_visitors_mdow",
                "traffic_profile_page_views_mdow",
                "traffic_profile_bounce_mdow",
                "traffic_profile_session_dur_mdow",
                "traffic_profile_last_year",
                "traffic_growth_sessions",
                "traffic_growth_visitors",
                "traffic_growth_page_views",
                "traffic_growth_bounce",
                "traffic_growth_session_dur",
            ]
        )

    work = web_traffic.copy()
    work["date"] = pd.to_datetime(work["date"])
    if train_cutoff is not None:
        work = work[work["date"] <= pd.Timestamp(train_cutoff)].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "day",
                "day_of_week",
                "traffic_profile_sessions_mdow",
                "traffic_profile_visitors_mdow",
                "traffic_profile_page_views_mdow",
                "traffic_profile_bounce_mdow",
                "traffic_profile_session_dur_mdow",
                "traffic_profile_last_year",
                "traffic_growth_sessions",
                "traffic_growth_visitors",
                "traffic_growth_page_views",
                "traffic_growth_bounce",
                "traffic_growth_session_dur",
            ]
        )

    daily = (
        work.groupby("date", as_index=False)
        .agg(
            sessions=("sessions", "sum"),
            unique_visitors=("unique_visitors", "sum"),
            page_views=("page_views", "sum"),
            bounce_rate=("bounce_rate", "mean"),
            avg_session_duration_sec=("avg_session_duration_sec", "mean"),
        )
        .rename(columns={"date": "Date"})
    )
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily["month"] = daily["Date"].dt.month
    daily["day"] = daily["Date"].dt.day
    daily["day_of_week"] = daily["Date"].dt.dayofweek
    daily["year"] = daily["Date"].dt.year

    annual = (
        daily.groupby("year", as_index=False)
        .agg(
            sessions=("sessions", "sum"),
            unique_visitors=("unique_visitors", "sum"),
            page_views=("page_views", "sum"),
            bounce_rate=("bounce_rate", "mean"),
            avg_session_duration_sec=("avg_session_duration_sec", "mean"),
        )
    )
    last_year = int(annual["year"].max())
    growth_sessions = _geometric_growth(annual["sessions"])
    growth_visitors = _geometric_growth(annual["unique_visitors"])
    growth_page_views = _geometric_growth(annual["page_views"])
    growth_bounce = _geometric_growth(annual["bounce_rate"].clip(lower=1e-6))
    growth_session_dur = _geometric_growth(annual["avg_session_duration_sec"].clip(lower=1e-6))

    profile = (
        daily.groupby(["month", "day", "day_of_week"], as_index=False)
        .agg(
            traffic_profile_sessions_mdow=("sessions", "median"),
            traffic_profile_visitors_mdow=("unique_visitors", "median"),
            traffic_profile_page_views_mdow=("page_views", "median"),
            traffic_profile_bounce_mdow=("bounce_rate", "median"),
            traffic_profile_session_dur_mdow=("avg_session_duration_sec", "median"),
        )
    )
    if "traffic_source" in work.columns:
        source_daily = (
            work.assign(traffic_source=work["traffic_source"].fillna("unknown"))
            .groupby(["date", "traffic_source"], as_index=False)["sessions"]
            .sum()
        )
        top_sources = source_daily.groupby("traffic_source")["sessions"].sum().nlargest(5).index
        source_daily["traffic_source"] = np.where(
            source_daily["traffic_source"].isin(top_sources),
            source_daily["traffic_source"],
            "other",
        )
        source_pivot = source_daily.pivot_table(
            index="date",
            columns="traffic_source",
            values="sessions",
            aggfunc="sum",
            fill_value=0.0,
        )
        source_pivot = source_pivot.div(source_pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        source_pivot.columns = [
            f"traffic_source_{_safe_feature_token(col)}_share" for col in source_pivot.columns
        ]
        source_profile = source_pivot.reset_index().rename(columns={"date": "Date"})
        source_profile["Date"] = pd.to_datetime(source_profile["Date"])
        source_profile["month"] = source_profile["Date"].dt.month
        source_profile["day"] = source_profile["Date"].dt.day
        source_profile["day_of_week"] = source_profile["Date"].dt.dayofweek
        source_profile = (
            source_profile.drop(columns=["Date"])
            .groupby(["month", "day", "day_of_week"], as_index=False)
            .median(numeric_only=True)
        )
        profile = profile.merge(source_profile, on=["month", "day", "day_of_week"], how="left")

    if use_growth_adjustment:
        profile["traffic_profile_last_year"] = last_year
        profile["traffic_growth_sessions"] = growth_sessions
        profile["traffic_growth_visitors"] = growth_visitors
        profile["traffic_growth_page_views"] = growth_page_views
        profile["traffic_growth_bounce"] = growth_bounce
        profile["traffic_growth_session_dur"] = growth_session_dur
    return profile


def build_inventory_profile(
    inventory: pd.DataFrame,
    use_future_trend: bool = False,
    train_cutoff: Optional[pd.Timestamp | str] = None,
) -> pd.DataFrame:
    """Build future-safe monthly inventory profiles plus a simple monthly trend."""
    if inventory.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "inv_profile_fill_rate",
                "inv_profile_stockout",
                "inv_profile_stock_on_hand",
                "inv_profile_sell_through",
                "inv_profile_last_period_idx",
                "inv_trend_fill_rate",
                "inv_trend_stockout",
                "inv_trend_stock_on_hand",
                "inv_trend_sell_through",
            ]
        )

    work = inventory.copy()
    work["snapshot_date"] = pd.to_datetime(work["snapshot_date"])
    if train_cutoff is not None:
        work = work[work["snapshot_date"] <= pd.Timestamp(train_cutoff)].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "inv_profile_fill_rate",
                "inv_profile_stockout",
                "inv_profile_stock_on_hand",
                "inv_profile_sell_through",
                "inv_profile_last_period_idx",
                "inv_trend_fill_rate",
                "inv_trend_stockout",
                "inv_trend_stock_on_hand",
                "inv_trend_sell_through",
            ]
        )

    work["month"] = work["snapshot_date"].dt.month
    monthly = (
        work.groupby("snapshot_date", as_index=False)
        .agg(
            avg_fill_rate=("fill_rate", "mean"),
            total_stockout=("stockout_flag", "sum"),
            avg_stock_on_hand=("stock_on_hand", "mean"),
            avg_sell_through=("sell_through_rate", "mean"),
        )
    )
    monthly["snapshot_date"] = pd.to_datetime(monthly["snapshot_date"])
    monthly["month"] = monthly["snapshot_date"].dt.month
    monthly["period_idx"] = monthly["snapshot_date"].dt.year * 12 + monthly["snapshot_date"].dt.month
    profile = (
        monthly.groupby("month", as_index=False)
        .agg(
            inv_profile_fill_rate=("avg_fill_rate", "median"),
            inv_profile_stockout=("total_stockout", "median"),
            inv_profile_stock_on_hand=("avg_stock_on_hand", "median"),
            inv_profile_sell_through=("avg_sell_through", "median"),
        )
    )
    if use_future_trend:
        last_period_idx = int(monthly["period_idx"].max())
        profile["inv_profile_last_period_idx"] = last_period_idx
        profile["inv_trend_fill_rate"] = _linear_trend(monthly["avg_fill_rate"], monthly["period_idx"])
        profile["inv_trend_stockout"] = _linear_trend(monthly["total_stockout"], monthly["period_idx"])
        profile["inv_trend_stock_on_hand"] = _linear_trend(monthly["avg_stock_on_hand"], monthly["period_idx"])
        profile["inv_trend_sell_through"] = _linear_trend(monthly["avg_sell_through"], monthly["period_idx"])

    for dim in ["category", "segment"]:
        if dim not in work.columns:
            continue
        top_values = work.groupby(dim)["units_sold"].sum().nlargest(5).index
        dim_work = work[work[dim].isin(top_values)].copy()
        if dim_work.empty:
            continue
        dim_work["_token"] = dim_work[dim].map(_safe_feature_token)
        dim_daily = (
            dim_work.groupby(["snapshot_date", "month", "_token"], as_index=False)
            .agg(
                fill_rate=("fill_rate", "mean"),
                stockout=("stockout_flag", "mean"),
                sell_through=("sell_through_rate", "mean"),
            )
        )
        dim_month = (
            dim_daily.groupby(["month", "_token"], as_index=False)
            .agg(
                fill_rate=("fill_rate", "median"),
                stockout=("stockout", "median"),
                sell_through=("sell_through", "median"),
            )
        )
        value_parts = []
        for metric in ["fill_rate", "stockout", "sell_through"]:
            pivot = dim_month.pivot(index="month", columns="_token", values=metric).reset_index()
            pivot.columns = [
                "month" if col == "month" else f"inv_{dim}_{col}_{metric}" for col in pivot.columns
            ]
            value_parts.append(pivot)
        dim_profile = value_parts[0]
        for part in value_parts[1:]:
            dim_profile = dim_profile.merge(part, on="month", how="outer")
        profile = profile.merge(dim_profile, on="month", how="left")
    return profile


def _empty_business_daily() -> pd.DataFrame:
    return pd.DataFrame(columns=["Date"])


def _normalize_business_daily(daily: pd.DataFrame, date_index: Optional[pd.Series] = None) -> pd.DataFrame:
    if daily.empty:
        return _empty_business_daily()

    out = daily.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.groupby("Date", as_index=False).sum(numeric_only=True)
    if date_index is not None:
        dates = pd.DataFrame({"Date": pd.to_datetime(date_index).sort_values().unique()})
        out = dates.merge(out, on="Date", how="left")
    out = out.sort_values("Date").reset_index(drop=True)
    numeric_cols = [c for c in out.columns if c != "Date"]
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _one_hot_daily_share(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    prefix: str,
    top_n: int = 4,
) -> pd.DataFrame:
    if df.empty or value_col not in df.columns:
        return _empty_business_daily()

    top_values = df[value_col].fillna("unknown").value_counts().head(top_n).index.tolist()
    work = df[[date_col, value_col]].copy()
    work[value_col] = np.where(work[value_col].isin(top_values), work[value_col], "other")
    counts = pd.crosstab(pd.to_datetime(work[date_col]), work[value_col])
    counts = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    counts.columns = [f"{prefix}_{str(c).lower()}_share" for c in counts.columns]
    return counts.reset_index().rename(columns={date_col: "Date", "index": "Date"})


def build_business_daily_table(
    orders: Optional[pd.DataFrame] = None,
    order_items: Optional[pd.DataFrame] = None,
    payments: Optional[pd.DataFrame] = None,
    returns: Optional[pd.DataFrame] = None,
    reviews: Optional[pd.DataFrame] = None,
    customers: Optional[pd.DataFrame] = None,
    shipments: Optional[pd.DataFrame] = None,
    products: Optional[pd.DataFrame] = None,
    date_index: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Aggregate auxiliary business tables to daily signals.

    Raw target-like value sums are intentionally excluded. Downstream code only
    exposes these signals through lags, rolling windows, or historical profiles.
    """
    if orders is None or orders.empty:
        return _empty_business_daily()

    orders_base = orders.copy()
    orders_base["order_date"] = pd.to_datetime(orders_base["order_date"])
    parts = []

    orders_daily = (
        orders_base.groupby("order_date", as_index=False)
        .agg(
            biz_orders_count=("order_id", "count"),
            biz_unique_customers=("customer_id", "nunique"),
        )
        .rename(columns={"order_date": "Date"})
    )
    orders_daily["biz_orders_per_customer"] = (
        orders_daily["biz_orders_count"] / orders_daily["biz_unique_customers"].replace(0, np.nan)
    )
    parts.append(orders_daily)

    for col, prefix in [
        ("payment_method", "biz_order_payment_method"),
        ("device_type", "biz_device"),
        ("order_source", "biz_source"),
        ("order_status", "biz_status"),
    ]:
        parts.append(_one_hot_daily_share(orders_base, "order_date", col, prefix))

    if order_items is not None and not order_items.empty:
        items = order_items.copy()
        items = items.merge(orders_base[["order_id", "order_date"]], on="order_id", how="left")
        items["gross_item_value"] = items["quantity"] * items["unit_price"]
        items["discount_amount"] = items["discount_amount"].fillna(0.0)
        if "is_promo_applied" in items.columns:
            items["has_promo"] = items["is_promo_applied"].fillna(0).astype(float)
        else:
            promo_cols = [c for c in ["promo_id", "promo_id_2"] if c in items.columns]
            items["has_promo"] = items[promo_cols].notna().any(axis=1).astype(float) if promo_cols else 0.0
        item_daily = (
            items.groupby("order_date", as_index=False)
            .agg(
                biz_items_sold=("quantity", "sum"),
                biz_unique_products_sold=("product_id", "nunique"),
                biz_avg_unit_price=("unit_price", "mean"),
                biz_discount_amount_mean=("discount_amount", "mean"),
                biz_discount_rate=("discount_amount", lambda s: float(s.sum())),
                biz_gross_value_for_rate=("gross_item_value", "sum"),
                biz_promo_item_share=("has_promo", "mean"),
            )
            .rename(columns={"order_date": "Date"})
        )
        item_daily["biz_discount_rate"] = (
            item_daily["biz_discount_rate"] / item_daily["biz_gross_value_for_rate"].replace(0, np.nan)
        )
        item_daily = item_daily.drop(columns=["biz_gross_value_for_rate"])
        item_daily = item_daily.merge(orders_daily[["Date", "biz_orders_count"]], on="Date", how="left")
        item_daily["biz_avg_items_per_order"] = (
            item_daily["biz_items_sold"] / item_daily["biz_orders_count"].replace(0, np.nan)
        )
        item_daily = item_daily.drop(columns=["biz_orders_count"])
        parts.append(item_daily)

        if products is not None and not products.empty:
            item_products = items.merge(products[["product_id", "category", "segment"]], on="product_id", how="left")
            parts.append(_one_hot_daily_share(item_products, "order_date", "category", "biz_category"))
            parts.append(_one_hot_daily_share(item_products, "order_date", "segment", "biz_segment"))

    if payments is not None and not payments.empty:
        pay = payments.copy().merge(orders_base[["order_id", "order_date"]], on="order_id", how="left")
        high_value_cutoff = pay["payment_value"].quantile(0.90)
        pay["is_high_value_payment"] = (pay["payment_value"] >= high_value_cutoff).astype(float)
        pay_daily = (
            pay.groupby("order_date", as_index=False)
            .agg(
                biz_payment_value_mean=("payment_value", "mean"),
                biz_payment_value_std=("payment_value", "std"),
                biz_installments_mean=("installments", "mean"),
                biz_high_value_payment_share=("is_high_value_payment", "mean"),
            )
            .rename(columns={"order_date": "Date"})
        )
        parts.append(pay_daily)
        parts.append(_one_hot_daily_share(pay, "order_date", "payment_method", "biz_payment"))

    if returns is not None and not returns.empty:
        ret = returns.copy()
        ret["return_date"] = pd.to_datetime(ret["return_date"])
        ret_daily = (
            ret.groupby("return_date", as_index=False)
            .agg(
                biz_return_count=("return_id", "count"),
                biz_return_quantity=("return_quantity", "sum"),
                biz_refund_amount_mean=("refund_amount", "mean"),
            )
            .rename(columns={"return_date": "Date"})
        )
        ret_daily = ret_daily.merge(orders_daily[["Date", "biz_orders_count"]], on="Date", how="left")
        ret_daily["biz_return_rate_proxy"] = (
            ret_daily["biz_return_count"] / ret_daily["biz_orders_count"].replace(0, np.nan)
        )
        ret_daily = ret_daily.drop(columns=["biz_orders_count"])
        parts.append(ret_daily)

    if reviews is not None and not reviews.empty:
        rev = reviews.copy()
        rev["review_date"] = pd.to_datetime(rev["review_date"])
        rev["bad_review"] = (rev["rating"] <= 2).astype(float)
        review_daily = (
            rev.groupby("review_date", as_index=False)
            .agg(
                biz_review_count=("review_id", "count"),
                biz_avg_rating=("rating", "mean"),
                biz_rating_std=("rating", "std"),
                biz_bad_review_ratio=("bad_review", "mean"),
            )
            .rename(columns={"review_date": "Date"})
        )
        parts.append(review_daily)

    if customers is not None and not customers.empty:
        cust = customers.copy()
        cust["signup_date"] = pd.to_datetime(cust["signup_date"])
        order_customer = orders_base.merge(
            cust[["customer_id", "signup_date", "gender", "age_group", "acquisition_channel"]],
            on="customer_id",
            how="left",
        )
        order_customer["is_new_customer"] = (
            order_customer["signup_date"].dt.normalize() == order_customer["order_date"].dt.normalize()
        ).astype(float)
        customer_daily = (
            order_customer.groupby("order_date", as_index=False)
            .agg(
                biz_new_customers=("is_new_customer", "sum"),
                biz_new_customer_share=("is_new_customer", "mean"),
            )
            .rename(columns={"order_date": "Date"})
        )
        customer_daily["biz_returning_customer_share"] = 1.0 - customer_daily["biz_new_customer_share"]
        parts.append(customer_daily)
        parts.append(_one_hot_daily_share(order_customer, "order_date", "acquisition_channel", "biz_acq"))
        parts.append(_one_hot_daily_share(order_customer, "order_date", "age_group", "biz_age"))
        parts.append(_one_hot_daily_share(order_customer, "order_date", "gender", "biz_gender"))

    if shipments is not None and not shipments.empty:
        ship = shipments.copy()
        ship["ship_date"] = pd.to_datetime(ship["ship_date"])
        ship["delivery_date"] = pd.to_datetime(ship["delivery_date"])
        ship["delivery_delay_days"] = (ship["delivery_date"] - ship["ship_date"]).dt.days
        ship["is_delayed_shipment"] = (ship["delivery_delay_days"] > 5).astype(float)
        ship_daily = (
            ship.groupby("ship_date", as_index=False)
            .agg(
                biz_shipping_volume=("order_id", "count"),
                biz_shipping_fee_mean=("shipping_fee", "mean"),
                biz_delivery_delay_mean=("delivery_delay_days", "mean"),
                biz_delayed_shipment_share=("is_delayed_shipment", "mean"),
            )
            .rename(columns={"ship_date": "Date"})
        )
        parts.append(ship_daily)

    if not parts:
        return _empty_business_daily()

    daily = parts[0]
    for part in parts[1:]:
        if part.empty:
            continue
        daily = daily.merge(part, on="Date", how="outer")
    return _normalize_business_daily(daily, date_index=date_index)


def build_business_profile(
    business_daily: Optional[pd.DataFrame],
    enabled_groups: Sequence[str],
) -> pd.DataFrame:
    """Build future-safe business profiles by recurring calendar pattern."""
    if business_daily is None or business_daily.empty or not enabled_groups:
        return pd.DataFrame(columns=["month", "day", "day_of_week"])

    selected = _select_business_columns(business_daily, enabled_groups)
    if not selected:
        return pd.DataFrame(columns=["month", "day", "day_of_week"])

    daily = business_daily[["Date"] + selected].copy()
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily["month"] = daily["Date"].dt.month
    daily["day"] = daily["Date"].dt.day
    daily["day_of_week"] = daily["Date"].dt.dayofweek
    profile = (
        daily.groupby(["month", "day", "day_of_week"], as_index=False)[selected]
        .median()
        .rename(columns={c: f"{c}_profile" for c in selected})
    )
    return profile


def _select_business_columns(business_daily: pd.DataFrame, enabled_groups: Sequence[str]) -> List[str]:
    group_prefixes = {
        "orders_items_core": [
            "biz_orders",
            "biz_unique_customers",
            "biz_items",
            "biz_unique_products",
            "biz_avg_items",
            "biz_avg_unit",
            "biz_discount",
            "biz_promo",
        ],
        "orders_items": [
            "biz_orders",
            "biz_unique_customers",
            "biz_items",
            "biz_unique_products",
            "biz_avg_items",
            "biz_avg_unit",
            "biz_discount",
            "biz_promo",
            "biz_category",
            "biz_segment",
            "biz_device",
            "biz_source",
            "biz_status",
        ],
        "payments": ["biz_payment", "biz_installments", "biz_high_value"],
        "returns_reviews": ["biz_return", "biz_refund", "biz_review", "biz_avg_rating", "biz_rating", "biz_bad"],
        "customers_shipments": ["biz_new", "biz_returning", "biz_acq", "biz_age", "biz_gender", "biz_shipping", "biz_delivery", "biz_delayed"],
    }
    prefixes = [p for group in enabled_groups for p in group_prefixes.get(group, [])]
    return [
        c
        for c in business_daily.columns
        if c != "Date" and any(c.startswith(prefix) for prefix in prefixes)
    ]


def add_known_future_features(
    df: pd.DataFrame,
    promo_profile: pd.DataFrame,
    traffic_profile: Optional[pd.DataFrame] = None,
    inventory_profile: Optional[pd.DataFrame] = None,
    business_profile: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Add only features that are knowable for future dates."""
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out = add_time_features(out, "Date")
    out = add_vn_calendar(out, "Date")
    out["month_dow"] = out["month"] * 10 + out["day_of_week"]

    out = out.merge(promo_profile, on=["month", "day"], how="left")
    promo_cols = [
        "promo_profile_count",
        "promo_profile_discount",
        "promo_profile_stackable",
        "promo_profile_pct_count",
        "promo_profile_fixed_count",
    ]
    for col in promo_cols:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = out[col].fillna(0.0)
    out["promo_profile_intensity"] = out["promo_profile_count"] * out["promo_profile_discount"]

    if traffic_profile is not None and not traffic_profile.empty:
        out = out.merge(traffic_profile, on=["month", "day", "day_of_week"], how="left")
    traffic_cols = [
        "traffic_profile_sessions_mdow",
        "traffic_profile_visitors_mdow",
        "traffic_profile_page_views_mdow",
        "traffic_profile_bounce_mdow",
        "traffic_profile_session_dur_mdow",
    ]
    for col in traffic_cols:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = out[col].fillna(0.0)
    if "traffic_profile_last_year" in out.columns:
        years_ahead = (out["year"] - out["traffic_profile_last_year"].fillna(out["year"])).clip(lower=0)
        traffic_scalers = {
            "traffic_profile_sessions_mdow": "traffic_growth_sessions",
            "traffic_profile_visitors_mdow": "traffic_growth_visitors",
            "traffic_profile_page_views_mdow": "traffic_growth_page_views",
            "traffic_profile_bounce_mdow": "traffic_growth_bounce",
            "traffic_profile_session_dur_mdow": "traffic_growth_session_dur",
        }
        for value_col, growth_col in traffic_scalers.items():
            if growth_col in out.columns:
                scale = np.power(out[growth_col].fillna(1.0).clip(lower=0.5, upper=1.5), years_ahead)
                out[value_col] = out[value_col] * scale
    out["traffic_profile_intensity"] = (
        out["traffic_profile_sessions_mdow"] * out["traffic_profile_page_views_mdow"]
    ) / (out["traffic_profile_visitors_mdow"] + 1.0)
    for col in [c for c in out.columns if c.startswith("traffic_source_") and c.endswith("_share")]:
        out[col] = out[col].fillna(0.0)

    if inventory_profile is not None and not inventory_profile.empty:
        out = out.merge(inventory_profile, on="month", how="left")
    inv_cols = [
        "inv_profile_fill_rate",
        "inv_profile_stockout",
        "inv_profile_stock_on_hand",
        "inv_profile_sell_through",
    ]
    for col in inv_cols:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = out[col].fillna(0.0)
    if "inv_profile_last_period_idx" in out.columns:
        period_idx = out["year"] * 12 + out["month"]
        months_ahead = (period_idx - out["inv_profile_last_period_idx"].fillna(period_idx)).clip(lower=0)
        trend_specs = {
            "inv_profile_fill_rate": ("inv_trend_fill_rate", 0.0, 1.0),
            "inv_profile_stockout": ("inv_trend_stockout", 0.0, None),
            "inv_profile_stock_on_hand": ("inv_trend_stock_on_hand", 0.0, None),
            "inv_profile_sell_through": ("inv_trend_sell_through", 0.0, 1.0),
        }
        for value_col, (trend_col, lower, upper) in trend_specs.items():
            if trend_col in out.columns:
                out[value_col] = out[value_col] + months_ahead * out[trend_col].fillna(0.0)
                if lower is not None:
                    out[value_col] = out[value_col].clip(lower=lower)
                if upper is not None:
                    out[value_col] = out[value_col].clip(upper=upper)
    for col in [c for c in out.columns if c.startswith("inv_category_") or c.startswith("inv_segment_")]:
        out[col] = out[col].fillna(0.0)

    if business_profile is not None and not business_profile.empty:
        out = out.merge(business_profile, on=["month", "day", "day_of_week"], how="left")
    business_profile_cols = [c for c in out.columns if c.startswith("biz_") and c.endswith("_profile")]
    for col in business_profile_cols:
        out[col] = out[col].fillna(0.0)

    return out


def add_business_lag_features(
    df: pd.DataFrame,
    business_daily: Optional[pd.DataFrame],
    enabled_groups: Sequence[str],
    lags: Sequence[int] = BUSINESS_LAGS,
    windows: Sequence[int] = BUSINESS_ROLL_WINDOWS,
) -> pd.DataFrame:
    """Add shifted business features only; raw same-day signals are dropped."""
    if business_daily is None or business_daily.empty or not enabled_groups:
        return df

    business_cols = _select_business_columns(business_daily, enabled_groups)
    if not business_cols:
        return df

    out = df.copy()
    daily = business_daily[["Date"] + business_cols].copy()
    daily["Date"] = pd.to_datetime(daily["Date"])
    out = out.merge(daily, on="Date", how="left")
    out = out.sort_values("Date").reset_index(drop=True)
    out[business_cols] = out[business_cols].fillna(0.0)

    derived = {}
    for col in business_cols:
        shifted = out[col].shift(1)
        for lag in lags:
            derived[f"{col}_lag_{lag}"] = out[col].shift(lag)
        for window in windows:
            base = shifted.rolling(window, min_periods=max(2, window // 2))
            derived[f"{col}_roll_mean_{window}"] = base.mean()
            derived[f"{col}_roll_std_{window}"] = base.std().fillna(0.0)

    out = pd.concat([out.drop(columns=business_cols), pd.DataFrame(derived, index=out.index)], axis=1)
    return out.copy()


def add_autoregressive_features(
    df: pd.DataFrame,
    target_col: str = "Revenue",
    lags: Sequence[int] = LAGS,
    windows: Sequence[int] = ROLL_WINDOWS,
) -> pd.DataFrame:
    """Add lag and rolling features using only previous target values."""
    out = df.sort_values("Date").reset_index(drop=True).copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[target_col].shift(lag)

    shifted = out[target_col].shift(1)
    for window in windows:
        base = shifted.rolling(window, min_periods=max(2, window // 2))
        out[f"roll_mean_{window}"] = base.mean()
        out[f"roll_std_{window}"] = base.std().fillna(0.0)
        out[f"roll_min_{window}"] = base.min()
        out[f"roll_max_{window}"] = base.max()
        out[f"roll_median_{window}"] = base.median()

    out["lag_7_over_30"] = out["lag_7"] / (out["lag_30"] + 1e-8)
    out["lag_ratio_7_30"] = (out["lag_7"] + 1.0) / (out["lag_30"] + 1.0)
    out["lag_364_over_365"] = (out.get("lag_364", np.nan) + 1.0) / (out.get("lag_365", np.nan) + 1.0)
    out["roll_7_over_30"] = out["roll_mean_7"] / (out["roll_mean_30"] + 1e-8)
    if "is_pre_tet" in out.columns:
        out["pre_tet_x_roll30"] = out["is_pre_tet"] * out["roll_mean_30"]
    out["rev_diff_7"] = out[target_col].shift(1) - out[target_col].shift(8)
    out["rev_diff_30"] = out[target_col].shift(1) - out[target_col].shift(31)
    out["rev_pct_7"] = (
        out[target_col].shift(1) / (out[target_col].shift(8) + 1e-8) - 1.0
    ).clip(-5, 5)
    out["rev_pct_30"] = (
        out[target_col].shift(1) / (out[target_col].shift(31) + 1e-8) - 1.0
    ).clip(-5, 5)
    out["rev_vs_ma30"] = out["lag_1"] / (out["roll_mean_30"] + 1e-8)
    out[["lag_7_over_30", "lag_ratio_7_30", "lag_364_over_365", "roll_7_over_30"]] = out[
        ["lag_7_over_30", "lag_ratio_7_30", "lag_364_over_365", "roll_7_over_30"]
    ].replace([np.inf, -np.inf], np.nan)
    out[["rev_pct_7", "rev_pct_30", "rev_vs_ma30"]] = out[
        ["rev_pct_7", "rev_pct_30", "rev_vs_ma30"]
    ].replace([np.inf, -np.inf], np.nan)
    return out


def make_supervised_frame(
    sales: pd.DataFrame,
    promo_profile: pd.DataFrame,
    traffic_profile: Optional[pd.DataFrame] = None,
    inventory_profile: Optional[pd.DataFrame] = None,
    business_daily: Optional[pd.DataFrame] = None,
    business_profile: Optional[pd.DataFrame] = None,
    business_groups: Optional[Sequence[str]] = None,
    target_col: str = "Revenue",
) -> Tuple[pd.DataFrame, List[str]]:
    """Create a supervised training frame for one-step-ahead tree models."""
    base = sales[["Date", target_col]].copy()
    feat = add_known_future_features(
        base[["Date"]],
        promo_profile,
        traffic_profile=traffic_profile,
        inventory_profile=inventory_profile,
        business_profile=business_profile,
    )
    feat = add_business_lag_features(
        feat,
        business_daily=business_daily,
        enabled_groups=business_groups or [],
    )
    feat[target_col] = base[target_col].values
    feat = add_autoregressive_features(feat, target_col=target_col)

    exclude = {"Date", target_col}
    feature_cols = [c for c in feat.columns if c not in exclude]
    feat = feat.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    return feat, feature_cols


def impute_feature_frame(
    X: pd.DataFrame,
    fill_values: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Impute feature matrix in a revenue-safe way.

    1. Forward-fill down each feature column to preserve temporal continuity.
    2. Fill remaining NaN with provided medians or medians learned from X.
    3. Use 0.0 only as a last-resort fallback for features that are entirely NaN.
    """
    out = X.copy()
    out = out.ffill()

    if fill_values is None:
        fill_values = out.median(numeric_only=True)
    else:
        fill_values = fill_values.reindex(out.columns)

    safe_fill = fill_values.fillna(0.0)
    out = out.fillna(safe_fill)
    out = out.fillna(0.0)
    return out, safe_fill


class _AutoregressiveState:
    """Lightweight incremental cache for recursive forecasting."""

    def __init__(self, history: pd.DataFrame, target_col: str = "Revenue"):
        hist = history.sort_values("Date").reset_index(drop=True)
        self.values = hist[target_col].astype(float).tolist()

    def append(self, value: float) -> None:
        self.values.append(float(value))

    def lag(self, lag: int) -> float:
        return self.values[-lag] if len(self.values) >= lag else np.nan

    def window_values(self, window: int) -> Optional[np.ndarray]:
        min_periods = max(2, window // 2)
        if len(self.values) < min_periods:
            return None
        tail = self.values[-window:]
        return np.asarray(tail, dtype=float)


class _BusinessFeatureState:
    """Incremental cache for lagged business profile estimates."""

    def __init__(
        self,
        history_dates: pd.Series,
        business_daily: Optional[pd.DataFrame],
        business_profile: Optional[pd.DataFrame],
        business_groups: Sequence[str],
    ):
        self.business_profile = business_profile
        self.business_groups = list(business_groups)
        self.cols = _select_business_columns(business_daily, self.business_groups) if business_daily is not None else []
        self.values = {col: [] for col in self.cols}
        if not self.cols:
            return

        hist = pd.DataFrame({"Date": pd.to_datetime(history_dates)})
        hist = hist.merge(business_daily[["Date"] + self.cols], on="Date", how="left")
        hist[self.cols] = hist[self.cols].fillna(0.0)
        for col in self.cols:
            self.values[col] = hist[col].astype(float).tolist()

    def profile_row(self, date: pd.Timestamp) -> Dict[str, float]:
        if not self.cols or self.business_profile is None or self.business_profile.empty:
            return {col: 0.0 for col in self.cols}
        mask = (
            (self.business_profile["month"] == date.month)
            & (self.business_profile["day"] == date.day)
            & (self.business_profile["day_of_week"] == date.dayofweek)
        )
        if not mask.any():
            return {col: 0.0 for col in self.cols}
        row = self.business_profile.loc[mask].iloc[0]
        return {col: float(row.get(f"{col}_profile", 0.0)) for col in self.cols}

    def append_profile(self, date: pd.Timestamp) -> None:
        profile_values = self.profile_row(date)
        for col in self.cols:
            self.values[col].append(float(profile_values.get(col, 0.0)))

    def add_lag_features(self, row: pd.DataFrame) -> pd.DataFrame:
        features = {}
        for col in self.cols:
            vals = self.values[col]
            for lag in BUSINESS_LAGS:
                features[f"{col}_lag_{lag}"] = vals[-lag] if len(vals) >= lag else np.nan
            for window in BUSINESS_ROLL_WINDOWS:
                min_periods = max(2, window // 2)
                if len(vals) >= min_periods:
                    tail = np.asarray(vals[-window:], dtype=float)
                    features[f"{col}_roll_mean_{window}"] = float(np.mean(tail))
                    features[f"{col}_roll_std_{window}"] = float(np.std(tail, ddof=1)) if len(tail) > 1 else 0.0
                else:
                    features[f"{col}_roll_mean_{window}"] = np.nan
                    features[f"{col}_roll_std_{window}"] = 0.0
        if not features:
            return row
        return pd.concat([row, pd.DataFrame([features], index=row.index)], axis=1)


def _feature_row_from_history(
    date: pd.Timestamp,
    state: _AutoregressiveState,
    promo_profile: pd.DataFrame,
    feature_cols: Sequence[str],
    traffic_profile: Optional[pd.DataFrame] = None,
    inventory_profile: Optional[pd.DataFrame] = None,
    business_profile: Optional[pd.DataFrame] = None,
    business_state: Optional[_BusinessFeatureState] = None,
    feature_fill_values: Optional[pd.Series] = None,
) -> pd.DataFrame:
    row = add_known_future_features(
        pd.DataFrame({"Date": [date]}),
        promo_profile,
        traffic_profile=traffic_profile,
        inventory_profile=inventory_profile,
        business_profile=business_profile,
    )
    if business_state is not None:
        row = business_state.add_lag_features(row)
    ar_features = {}
    for lag in LAGS:
        ar_features[f"lag_{lag}"] = state.lag(lag)

    for window in ROLL_WINDOWS:
        window_values = state.window_values(window)
        if window_values is not None:
            ar_features[f"roll_mean_{window}"] = float(np.mean(window_values))
            ar_features[f"roll_std_{window}"] = float(np.std(window_values, ddof=1)) if len(window_values) > 1 else 0.0
            ar_features[f"roll_min_{window}"] = float(np.min(window_values))
            ar_features[f"roll_max_{window}"] = float(np.max(window_values))
            ar_features[f"roll_median_{window}"] = float(np.median(window_values))
        else:
            ar_features[f"roll_mean_{window}"] = np.nan
            ar_features[f"roll_std_{window}"] = 0.0
            ar_features[f"roll_min_{window}"] = np.nan
            ar_features[f"roll_max_{window}"] = np.nan
            ar_features[f"roll_median_{window}"] = np.nan

    ar_features["lag_7_over_30"] = ar_features["lag_7"] / (ar_features["lag_30"] + 1e-8)
    ar_features["lag_ratio_7_30"] = (ar_features["lag_7"] + 1.0) / (ar_features["lag_30"] + 1.0)
    ar_features["lag_364_over_365"] = (ar_features.get("lag_364", np.nan) + 1.0) / (
        ar_features.get("lag_365", np.nan) + 1.0
    )
    ar_features["roll_7_over_30"] = ar_features["roll_mean_7"] / (ar_features["roll_mean_30"] + 1e-8)
    row_flags = row.iloc[0] if len(row) else {}
    ar_features["pre_tet_x_roll30"] = float(row_flags.get("is_pre_tet", 0.0)) * ar_features["roll_mean_30"]
    if len(state.values) >= 8:
        ar_features["rev_diff_7"] = state.values[-1] - state.values[-8]
        ar_features["rev_pct_7"] = np.clip(state.values[-1] / (state.values[-8] + 1e-8) - 1.0, -5, 5)
    else:
        ar_features["rev_diff_7"] = np.nan
        ar_features["rev_pct_7"] = np.nan
    if len(state.values) >= 31:
        ar_features["rev_diff_30"] = state.values[-1] - state.values[-31]
        ar_features["rev_pct_30"] = np.clip(state.values[-1] / (state.values[-31] + 1e-8) - 1.0, -5, 5)
    else:
        ar_features["rev_diff_30"] = np.nan
        ar_features["rev_pct_30"] = np.nan
    ar_features["rev_vs_ma30"] = ar_features["lag_1"] / (ar_features["roll_mean_30"] + 1e-8)
    row = pd.concat([row, pd.DataFrame([ar_features], index=row.index)], axis=1)

    # Reindex instead of strict .loc so optional profile groups can be disabled
    # without crashing recursive inference. Missing columns are imputed below.
    result = row.reindex(columns=feature_cols).copy()
    result, _ = impute_feature_frame(result, fill_values=feature_fill_values)
    return result


def recursive_forecast(
    model,
    history: pd.DataFrame,
    future_dates: Iterable[pd.Timestamp],
    feature_cols: Sequence[str],
    promo_profile: pd.DataFrame,
    traffic_profile: Optional[pd.DataFrame] = None,
    inventory_profile: Optional[pd.DataFrame] = None,
    business_daily: Optional[pd.DataFrame] = None,
    business_profile: Optional[pd.DataFrame] = None,
    business_groups: Optional[Sequence[str]] = None,
    feature_fill_values: Optional[pd.Series] = None,
    anchor_pred: Optional[Sequence[float]] = None,
    blend_weight: float = 0.0,
    ewm_alpha: float = 1.0,
    predict_log_target: bool = False,
    target_col: str = "Revenue",
) -> np.ndarray:
    """
    Forecast each future day and feed predictions back into lag features.

    Uses an incremental autoregressive state instead of rebuilding / appending
    a full history DataFrame every step.
    """
    state = _AutoregressiveState(history, target_col=target_col)
    business_state = _BusinessFeatureState(
        history_dates=history["Date"],
        business_daily=business_daily,
        business_profile=business_profile,
        business_groups=business_groups or [],
    )
    preds = []
    dates = pd.to_datetime(list(future_dates))
    anchor_values = None if anchor_pred is None else np.asarray(anchor_pred, dtype=float)
    prev_pred = None

    for idx, date in enumerate(dates):
        x_row = _feature_row_from_history(
            date,
            state,
            promo_profile,
            feature_cols,
            traffic_profile=traffic_profile,
            inventory_profile=inventory_profile,
            business_profile=business_profile,
            business_state=business_state,
            feature_fill_values=feature_fill_values,
        )
        raw_pred = float(model.predict(x_row.to_numpy(dtype=np.float32))[0])
        pred = float(np.expm1(raw_pred)) if predict_log_target else raw_pred
        pred = float(np.maximum(pred, 0.0))
        if anchor_values is not None and idx < len(anchor_values):
            anchor = float(np.maximum(anchor_values[idx], 0.0))
            pred = (1.0 - blend_weight) * pred + blend_weight * anchor
        if prev_pred is not None:
            pred = ewm_alpha * pred + (1.0 - ewm_alpha) * prev_pred
        pred = float(np.maximum(pred, 0.0))
        preds.append(pred)
        state.append(pred)
        business_state.append_profile(date)
        prev_pred = pred

    return np.array(preds)


def calibrate_monthly_predictions(
    preds: Sequence[float],
    dates: Sequence[pd.Timestamp],
    train: pd.DataFrame,
    target_col: str = "Revenue",
    growth_clip: Tuple[float, float] = (0.90, 1.12),
    scale_clip: Tuple[float, float] = (0.75, 1.30),
    strength: float = 1.0,
) -> np.ndarray:
    """Scale forecast months toward historical monthly seasonal expectation."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return np.maximum(np.asarray(preds, dtype=float), 0.0)

    pred_df = pd.DataFrame({"Date": pd.to_datetime(list(dates)), target_col: np.asarray(preds, dtype=float)})
    hist = train[["Date", target_col]].copy()
    hist["year"] = hist["Date"].dt.year
    hist["month"] = hist["Date"].dt.month

    yearly = hist.groupby("year")[target_col].sum().sort_index()
    if len(yearly) >= 2:
        growth = float(yearly.iloc[-1] / max(yearly.iloc[-2], 1.0))
        growth = float(np.clip(growth, *growth_clip))
    else:
        growth = 1.0

    monthly_expectation = hist.groupby("month")[target_col].mean()
    pred_df["year"] = pred_df["Date"].dt.year
    pred_df["month"] = pred_df["Date"].dt.month
    base_year = int(hist["year"].max())

    for (year, month), grp in pred_df.groupby(["year", "month"]):
        expected = float(monthly_expectation.get(month, grp[target_col].mean()))
        years_ahead = max(0, int(year) - base_year)
        expected *= growth ** years_ahead
        current = float(grp[target_col].mean())
        if current <= 0 or not np.isfinite(current):
            continue
        raw_scale = float(np.clip(expected / current, *scale_clip))
        scale = 1.0 + strength * (raw_scale - 1.0)
        pred_df.loc[grp.index, target_col] *= scale

    return np.maximum(pred_df[target_col].to_numpy(dtype=float), 0.0)


def apply_flat_scale(preds: Sequence[float], scale: float) -> np.ndarray:
    """Apply one multiplicative level scale across the whole horizon."""
    return np.maximum(np.asarray(preds, dtype=float) * float(scale), 0.0)


def apply_linear_tilt(
    preds: Sequence[float],
    start_scale: float,
    end_scale: float,
) -> np.ndarray:
    """Apply a linear horizon uplift from start_scale to end_scale."""
    values = np.asarray(preds, dtype=float)
    if len(values) == 0:
        return values
    curve = np.linspace(float(start_scale), float(end_scale), len(values))
    return np.maximum(values * curve, 0.0)


def apply_piecewise_year_tilt(
    preds: Sequence[float],
    dates: Sequence[pd.Timestamp],
    scale_2023_start: float,
    scale_2023_end: float,
    scale_2024_start: float,
    scale_2024_end: float,
) -> np.ndarray:
    """Apply separate linear horizon uplift curves for 2023 and 2024+."""
    values = np.asarray(preds, dtype=float)
    date_index = pd.to_datetime(list(dates))
    if len(values) != len(date_index):
        raise ValueError("preds and dates must have the same length")
    if len(values) == 0:
        return values

    curve = np.ones(len(values), dtype=float)
    mask_2023 = date_index < pd.Timestamp("2024-01-01")
    mask_2024 = ~mask_2023
    if mask_2023.any():
        curve[mask_2023] = np.linspace(float(scale_2023_start), float(scale_2023_end), int(mask_2023.sum()))
    if mask_2024.any():
        curve[mask_2024] = np.linspace(float(scale_2024_start), float(scale_2024_end), int(mask_2024.sum()))
    return np.maximum(values * curve, 0.0)


def apply_compound_tilt(
    preds: Sequence[float],
    base_start_scale: float,
    base_end_scale: float,
    extra_start_scale: float,
    extra_end_scale: float,
    global_scale: float = 1.0,
) -> np.ndarray:
    """Apply the product of two linear uplift curves plus an optional level scale."""
    values = np.asarray(preds, dtype=float)
    if len(values) == 0:
        return values
    base_curve = np.linspace(float(base_start_scale), float(base_end_scale), len(values))
    extra_curve = np.linspace(float(extra_start_scale), float(extra_end_scale), len(values))
    curve = float(global_scale) * base_curve * extra_curve
    return np.maximum(values * curve, 0.0)


def summarize_uplift_curve(
    n_periods: int,
    mode: str,
    dates: Optional[Sequence[pd.Timestamp]] = None,
    **params,
) -> Dict[str, float]:
    """Return first/last/mean scale for a configured uplift curve."""
    if n_periods <= 0:
        return {"scale_first": np.nan, "scale_last": np.nan, "scale_mean": np.nan}

    base = np.ones(int(n_periods), dtype=float)
    if mode == "flat":
        scaled = apply_flat_scale(base, params.get("scale", 1.0))
    elif mode == "linear":
        scaled = apply_linear_tilt(base, params.get("start_scale", 1.0), params.get("end_scale", 1.0))
    elif mode == "piecewise_year":
        if dates is None:
            raise ValueError("dates are required for piecewise_year uplift")
        scaled = apply_piecewise_year_tilt(
            base,
            dates,
            params.get("scale_2023_start", 1.0),
            params.get("scale_2023_end", 1.0),
            params.get("scale_2024_start", 1.0),
            params.get("scale_2024_end", 1.0),
        )
    elif mode == "compound":
        scaled = apply_compound_tilt(
            base,
            params.get("base_start_scale", 1.0),
            params.get("base_end_scale", 1.0),
            params.get("extra_start_scale", 1.0),
            params.get("extra_end_scale", 1.0),
            params.get("global_scale", 1.0),
        )
    else:
        raise ValueError(f"Unknown uplift mode: {mode}")

    return {
        "scale_first": float(scaled[0]),
        "scale_last": float(scaled[-1]),
        "scale_mean": float(np.mean(scaled)),
    }


def seasonal_window_baseline(
    train: pd.DataFrame,
    future_dates: Sequence[pd.Timestamp],
    target_col: str = "Revenue",
    window: int = 7,
) -> np.ndarray:
    """Direct seasonal forecast from historical same-weekday day-of-year neighbors."""
    hist = train[["Date", target_col]].copy()
    hist["doy"] = hist["Date"].dt.dayofyear
    hist["dow"] = hist["Date"].dt.dayofweek
    doys = hist["doy"].to_numpy(dtype=int)
    dows = hist["dow"].to_numpy(dtype=int)
    values = hist[target_col].to_numpy(dtype=float)
    fallback = float(np.nanmedian(values))

    preds = []
    for date in pd.to_datetime(list(future_dates)):
        doy = int(date.dayofyear)
        dow = int(date.dayofweek)
        diff = np.abs(doys - doy)
        dist = np.minimum(diff, 366 - diff)
        mask = (dows == dow) & (dist <= window)
        if mask.sum() < 3:
            mask = (dows == dow) & (dist <= window + 7)
        if mask.sum() < 3:
            mask = dist <= window
        preds.append(float(np.nanmedian(values[mask])) if mask.sum() else fallback)

    return np.maximum(np.asarray(preds, dtype=float), 0.0)


def build_transaction_component_table(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    products: pd.DataFrame,
    sales: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build daily business components that exactly reconstruct Revenue/COGS."""
    required_orders = {"order_id", "order_date"}
    required_items = {"order_id", "product_id", "quantity", "unit_price"}
    required_products = {"product_id", "cogs"}
    missing = (
        (required_orders - set(orders.columns))
        | (required_items - set(order_items.columns))
        | (required_products - set(products.columns))
    )
    if missing:
        raise ValueError(f"Missing required transaction columns: {sorted(missing)}")

    order_base = orders[["order_id", "order_date"]].copy()
    order_base["Date"] = pd.to_datetime(order_base["order_date"])
    items = order_items[["order_id", "product_id", "quantity", "unit_price"]].copy()
    items["quantity"] = pd.to_numeric(items["quantity"], errors="coerce").fillna(0.0)
    items["unit_price"] = pd.to_numeric(items["unit_price"], errors="coerce").fillna(0.0)
    prod = products[["product_id", "cogs"]].copy()
    prod["cogs"] = pd.to_numeric(prod["cogs"], errors="coerce").fillna(0.0)

    detail = order_base.merge(items, on="order_id", how="left").merge(prod, on="product_id", how="left")
    detail["quantity"] = detail["quantity"].fillna(0.0)
    detail["unit_price"] = detail["unit_price"].fillna(0.0)
    detail["cogs"] = detail["cogs"].fillna(0.0)
    detail["gross_value"] = detail["quantity"] * detail["unit_price"]
    detail["cogs_value"] = detail["quantity"] * detail["cogs"]

    daily_items = detail.groupby("Date").agg(
        items_quantity=("quantity", "sum"),
        order_lines=("order_id", "count"),
        Revenue_reconstructed=("gross_value", "sum"),
        COGS_reconstructed=("cogs_value", "sum"),
    )
    daily_orders = order_base.groupby("Date").agg(orders_count=("order_id", "nunique"))
    daily = daily_orders.join(daily_items, how="outer").fillna(0.0).reset_index()

    if sales is not None and not sales.empty:
        all_dates = pd.DataFrame({"Date": pd.to_datetime(sales["Date"])})
        daily = all_dates.merge(daily, on="Date", how="left").fillna(0.0)

    daily = daily.sort_values("Date").reset_index(drop=True)
    denom_orders = daily["orders_count"].replace(0, np.nan)
    denom_qty = daily["items_quantity"].replace(0, np.nan)
    daily["qty_per_order"] = (daily["items_quantity"] / denom_orders).fillna(0.0)
    daily["gross_per_qty"] = (daily["Revenue_reconstructed"] / denom_qty).fillna(0.0)
    daily["cogs_per_qty"] = (daily["COGS_reconstructed"] / denom_qty).fillna(0.0)
    daily["gross_per_order"] = (daily["Revenue_reconstructed"] / denom_orders).fillna(0.0)
    daily["cogs_per_order"] = (daily["COGS_reconstructed"] / denom_orders).fillna(0.0)
    return daily


def component_reconstruction_diagnostics(
    component_daily: pd.DataFrame,
    sales: pd.DataFrame,
) -> Dict[str, float]:
    """Measure how closely transaction components reconstruct sales targets."""
    merged = sales[["Date", "Revenue", "COGS"]].copy()
    merged["Date"] = pd.to_datetime(merged["Date"])
    comp = component_daily[["Date", "Revenue_reconstructed", "COGS_reconstructed"]].copy()
    comp["Date"] = pd.to_datetime(comp["Date"])
    merged = merged.merge(comp, on="Date", how="left").fillna(0.0)
    return {
        "Revenue_reconstruction_MAE": float(
            mean_absolute_error(merged["Revenue"], merged["Revenue_reconstructed"])
        ),
        "COGS_reconstruction_MAE": float(
            mean_absolute_error(merged["COGS"], merged["COGS_reconstructed"])
        ),
        "Revenue_reconstruction_max_abs": float(
            np.max(np.abs(merged["Revenue"] - merged["Revenue_reconstructed"]))
        ),
        "COGS_reconstruction_max_abs": float(
            np.max(np.abs(merged["COGS"] - merged["COGS_reconstructed"]))
        ),
    }


def build_component_shape_profile(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    products: pd.DataFrame,
    sales: Optional[pd.DataFrame] = None,
    train_cutoff: Optional[pd.Timestamp | str] = None,
) -> pd.DataFrame:
    """Build future-safe transaction component profiles for shape adjustment."""
    component_daily = build_transaction_component_table(orders, order_items, products, sales=sales)
    if train_cutoff is not None:
        component_daily = component_daily[component_daily["Date"] <= pd.Timestamp(train_cutoff)].copy()
    if component_daily.empty:
        return pd.DataFrame(columns=["month", "day", "day_of_week"])

    daily = component_daily.copy()
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily["month"] = daily["Date"].dt.month
    daily["day"] = daily["Date"].dt.day
    daily["day_of_week"] = daily["Date"].dt.dayofweek
    daily["component_cogs_ratio"] = (
        daily["COGS_reconstructed"] / daily["Revenue_reconstructed"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    cols = ["orders_count", "qty_per_order", "gross_per_qty", "component_cogs_ratio"]
    for col in cols:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")

    profile = (
        daily.groupby(["month", "day", "day_of_week"], as_index=False)[cols]
        .median()
        .rename(columns={c: f"component_profile_{c}" for c in cols})
    )
    fallback = {
        f"component_profile_{c}": float(daily[c].replace([np.inf, -np.inf], np.nan).median())
        for c in cols
    }
    for col, value in fallback.items():
        profile[col] = profile[col].fillna(value)
    return profile


def forecast_component_shape(
    component_profile: pd.DataFrame,
    future_dates: Sequence[pd.Timestamp],
) -> pd.DataFrame:
    """Forecast Revenue/COGS shape from historical transaction component profiles."""
    dates = pd.to_datetime(list(future_dates))
    future = pd.DataFrame({"Date": dates})
    future["month"] = future["Date"].dt.month
    future["day"] = future["Date"].dt.day
    future["day_of_week"] = future["Date"].dt.dayofweek
    if component_profile is None or component_profile.empty:
        future["Revenue"] = 0.0
        future["COGS"] = 0.0
        return future[["Date", "Revenue", "COGS"]]

    prof = component_profile.copy()
    value_cols = [c for c in prof.columns if c.startswith("component_profile_")]
    fallbacks = {c: float(pd.to_numeric(prof[c], errors="coerce").median()) for c in value_cols}
    future = future.merge(prof, on=["month", "day", "day_of_week"], how="left")
    for col, value in fallbacks.items():
        future[col] = pd.to_numeric(future[col], errors="coerce").fillna(value)

    orders = future["component_profile_orders_count"].clip(lower=0.0)
    qty_per_order = future["component_profile_qty_per_order"].clip(lower=0.0)
    gross_per_qty = future["component_profile_gross_per_qty"].clip(lower=0.0)
    cogs_ratio = future["component_profile_component_cogs_ratio"].clip(lower=0.0, upper=0.995)
    revenue = orders * qty_per_order * gross_per_qty
    cogs = revenue * cogs_ratio
    future["Revenue"] = np.maximum(revenue, 0.0)
    future["COGS"] = np.minimum(np.maximum(cogs, 0.0), future["Revenue"] * 0.995)
    return future[["Date", "Revenue", "COGS"]]


def blend_with_normalized_shape(
    base_pred: Sequence[float],
    shape_pred: Sequence[float],
    weight: float,
) -> np.ndarray:
    """Blend base forecast with a shape forecast normalized to the base mean level."""
    base = np.asarray(base_pred, dtype=float)
    shape = np.asarray(shape_pred, dtype=float)
    if len(base) != len(shape):
        raise ValueError("base_pred and shape_pred must have the same length")
    weight = float(np.clip(weight, 0.0, 1.0))
    if len(base) == 0 or weight <= 0:
        return np.maximum(base, 0.0)
    shape_mean = float(np.nanmean(shape))
    base_mean = float(np.nanmean(base))
    if not np.isfinite(shape_mean) or shape_mean <= 0 or not np.isfinite(base_mean):
        return np.maximum(base, 0.0)
    normalized_shape = shape * (base_mean / shape_mean)
    blended = (1.0 - weight) * base + weight * normalized_shape
    return np.maximum(blended, 0.0)


def apply_recovery_trend_calibration(
    preds: Sequence[float],
    dates: Sequence[pd.Timestamp | str],
    historical_sales: pd.DataFrame,
    strength: float,
    cap_year1: float = 1.12,
    cap_year2: float = 1.18,
) -> np.ndarray:
    """
    Apply a mild annual recovery uplift learned from the latest yearly trend.

    This is intentionally conservative: it only uses historical annual totals up
    to the training cutoff and clips the first/second forecast-year uplifts.
    """
    pred = np.asarray(preds, dtype=float).copy()
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0 or len(pred) == 0:
        return np.maximum(pred, 0.0)

    d = pd.Series(pd.to_datetime(list(dates)))
    if d.empty or historical_sales.empty:
        return np.maximum(pred, 0.0)

    hist = historical_sales[["Date", "Revenue"]].copy()
    hist["Date"] = pd.to_datetime(hist["Date"])
    hist["year"] = hist["Date"].dt.year
    yearly = hist.groupby("year")["Revenue"].sum().sort_index()
    max_hist_year = int(hist["year"].max())

    growth_candidates = []
    if 2020 in yearly.index and 2021 in yearly.index and yearly.loc[2020] > 0:
        growth_candidates.append((0.25, float(yearly.loc[2021] / yearly.loc[2020])))
    if 2021 in yearly.index and 2022 in yearly.index and yearly.loc[2021] > 0:
        growth_candidates.append((0.75, float(yearly.loc[2022] / yearly.loc[2021])))
    if not growth_candidates:
        recent = yearly.tail(3)
        yoy = recent.pct_change().dropna()
        if yoy.empty:
            return np.maximum(pred, 0.0)
        recovery_growth = float(1.0 + yoy.mean())
    else:
        total_weight = sum(w for w, _ in growth_candidates)
        recovery_growth = sum(w * g for w, g in growth_candidates) / max(total_weight, 1e-8)

    recovery_growth = max(1.0, recovery_growth)
    years = d.dt.year.to_numpy()
    factors = np.ones(len(pred), dtype=float)
    for idx, year in enumerate(years):
        horizon_year = int(year) - max_hist_year
        if horizon_year <= 0:
            continue
        raw_factor = 1.0 + strength * ((recovery_growth ** horizon_year) - 1.0)
        cap = cap_year1 if horizon_year == 1 else cap_year2
        factors[idx] = float(np.clip(raw_factor, 1.0, cap))

    return np.maximum(pred * factors, 0.0)


def forecast_cogs_ratio_model(
    sales: pd.DataFrame,
    revenue_pred: Sequence[float],
    future_dates: Sequence[pd.Timestamp],
    promo_profile: pd.DataFrame,
    traffic_profile: Optional[pd.DataFrame] = None,
    inventory_profile: Optional[pd.DataFrame] = None,
    business_daily: Optional[pd.DataFrame] = None,
    business_profile: Optional[pd.DataFrame] = None,
    business_groups: Optional[Sequence[str]] = None,
    blend_weight: float = 0.15,
    ewm_alpha: float = 0.90,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forecast COGS/Revenue ratio and convert Revenue predictions to COGS."""
    hist = sales[["Date", "Revenue", "COGS"]].copy()
    hist["Date"] = pd.to_datetime(hist["Date"])
    hist["cogs_ratio"] = (
        hist["COGS"] / hist["Revenue"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    hist = hist.dropna(subset=["cogs_ratio"]).copy()
    if hist.empty:
        revenue = np.asarray(revenue_pred, dtype=float)
        return np.maximum(revenue * 0.85, 0.0), np.full(len(revenue), 0.85)

    ratio_target = hist[["Date", "cogs_ratio"]].copy()
    frame, feature_cols = make_supervised_frame(
        ratio_target,
        promo_profile,
        traffic_profile=traffic_profile,
        inventory_profile=inventory_profile,
        business_daily=business_daily,
        business_profile=business_profile,
        business_groups=business_groups or [],
        target_col="cogs_ratio",
    )
    X, fills = impute_feature_frame(frame[feature_cols])
    y = frame["cogs_ratio"].astype(float).clip(lower=0.0, upper=1.5)
    if lgb is not None:
        ratio_params = {
            **LGBM_V2_PARAMS,
            "n_estimators": 1200,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 20,
        }
        model = lgb.LGBMRegressor(**ratio_params)
        model.fit(X.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32))
    else:
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.04,
            max_iter=500,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=RANDOM_SEED,
        )
        model.fit(X.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32))

    future_dates = pd.to_datetime(list(future_dates))
    monthly_ratio = ratio_target.assign(month=ratio_target["Date"].dt.month).groupby("month")["cogs_ratio"].median()
    global_ratio = float(ratio_target["cogs_ratio"].median())
    anchor = np.asarray([monthly_ratio.get(d.month, global_ratio) for d in future_dates], dtype=float)
    ratio_pred = recursive_forecast(
        model,
        ratio_target,
        future_dates,
        feature_cols,
        promo_profile,
        traffic_profile=traffic_profile,
        inventory_profile=inventory_profile,
        business_daily=business_daily,
        business_profile=business_profile,
        business_groups=business_groups or [],
        feature_fill_values=fills,
        anchor_pred=anchor,
        blend_weight=blend_weight,
        ewm_alpha=ewm_alpha,
        predict_log_target=False,
        target_col="cogs_ratio",
    )
    lower = float(ratio_target["cogs_ratio"].quantile(0.02))
    upper = min(float(ratio_target["cogs_ratio"].quantile(0.98)), 0.995)
    ratio_pred = np.clip(ratio_pred, lower, upper)
    revenue = np.asarray(revenue_pred, dtype=float)
    cogs = np.minimum(np.maximum(revenue * ratio_pred, 0.0), np.maximum(revenue, 0.0) * 0.995)
    return cogs, ratio_pred


def forecast_decomposition_components(
    component_daily: pd.DataFrame,
    future_dates: Sequence[pd.Timestamp],
    promo_profile: pd.DataFrame,
    traffic_profile: Optional[pd.DataFrame] = None,
    inventory_profile: Optional[pd.DataFrame] = None,
    component_targets: Sequence[str] = ("orders_count", "qty_per_order", "gross_per_qty", "cogs_per_qty"),
    blend_weight: float = 0.15,
    ewm_alpha: float = 0.80,
) -> pd.DataFrame:
    """Forecast transaction components and reconstruct Revenue/COGS."""
    history = component_daily.sort_values("Date").reset_index(drop=True).copy()
    future_dates = pd.to_datetime(list(future_dates))
    preds: Dict[str, np.ndarray] = {}

    for target in component_targets:
        target_history = history[["Date", target]].copy()
        target_history[target] = pd.to_numeric(target_history[target], errors="coerce").fillna(0.0).clip(lower=0.0)
        frame, feature_cols = make_supervised_frame(
            target_history,
            promo_profile,
            traffic_profile=traffic_profile,
            inventory_profile=inventory_profile,
            target_col=target,
        )
        X, fill_values = impute_feature_frame(frame[feature_cols])
        y = np.log1p(frame[target].clip(lower=0.0))
        _, model = train_primary_model(X, y)
        anchor = seasonal_growth_baseline(target_history, future_dates, target)
        pred = recursive_forecast(
            model,
            target_history,
            future_dates,
            feature_cols,
            promo_profile,
            traffic_profile=traffic_profile,
            inventory_profile=inventory_profile,
            feature_fill_values=fill_values,
            anchor_pred=anchor,
            blend_weight=blend_weight,
            ewm_alpha=ewm_alpha,
            predict_log_target=True,
            target_col=target,
        )
        if target in {"gross_per_qty", "cogs_per_qty", "qty_per_order"}:
            nonzero = target_history[target_history[target] > 0][target]
            if len(nonzero) >= 20:
                lower = float(nonzero.quantile(0.02))
                upper = float(nonzero.quantile(0.98))
                pred = np.clip(pred, lower, upper)
        preds[target] = np.maximum(pred, 0.0)

    orders_count = preds["orders_count"]
    qty_per_order = preds["qty_per_order"]
    gross_per_qty = preds["gross_per_qty"]
    cogs_per_qty = preds["cogs_per_qty"]
    items_quantity = np.maximum(orders_count * qty_per_order, 0.0)
    revenue = np.maximum(items_quantity * gross_per_qty, 0.0)
    cogs = np.maximum(items_quantity * cogs_per_qty, 0.0)
    cogs = np.minimum(cogs, revenue * 0.995)

    result = pd.DataFrame(
        {
            "Date": future_dates,
            "orders_count": orders_count,
            "qty_per_order": qty_per_order,
            "gross_per_qty": gross_per_qty,
            "cogs_per_qty": cogs_per_qty,
            "items_quantity": items_quantity,
            "Revenue": revenue,
            "COGS": cogs,
        }
    )
    return result


def seasonal_growth_baseline(
    train: pd.DataFrame,
    future_dates: pd.Series,
    target_col: str = "Revenue",
) -> np.ndarray:
    """BTC-style seasonal profile baseline using only the provided train slice."""
    hist = train[["Date", target_col]].copy()
    hist["year"] = hist["Date"].dt.year
    hist["month"] = hist["Date"].dt.month
    hist["day"] = hist["Date"].dt.day

    annual = hist.groupby("year")[target_col].sum()
    full_years = annual[(annual.index > hist["year"].min()) & (annual.index < hist["year"].max() + 1)]
    yoy = full_years.pct_change().dropna()
    growth = (1 + yoy).prod() ** (1 / len(yoy)) if len(yoy) else 1.0

    last_year = int(annual.index.max())
    base = annual.loc[last_year] / (366 if pd.Timestamp(last_year, 12, 31).is_leap_year else 365)

    annual_mean = hist.groupby("year")[target_col].transform("mean")
    hist["norm"] = hist[target_col] / annual_mean.replace(0, np.nan)
    profile = hist.groupby(["month", "day"])["norm"].mean().reset_index()

    future = pd.DataFrame({"Date": pd.to_datetime(future_dates)})
    future["month"] = future["Date"].dt.month
    future["day"] = future["Date"].dt.day
    future["years_ahead"] = future["Date"].dt.year - last_year
    future = future.merge(profile, on=["month", "day"], how="left")
    future["norm"] = future["norm"].fillna(1.0)
    return np.maximum(base * (growth ** future["years_ahead"]) * future["norm"], 0.0).to_numpy()


def train_primary_model(X: pd.DataFrame, y: pd.Series):
    if lgb is not None:
        model = lgb.LGBMRegressor(**LGBM_V2_PARAMS)
        model.fit(X.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32))
        return "lgbm_recursive", model

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.04,
        max_iter=500,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=RANDOM_SEED,
    )
    model.fit(X.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32))
    return "hgb_recursive", model


def train_secondary_model(X: pd.DataFrame, y: pd.Series):
    if xgb is not None:
        model = xgb.XGBRegressor(**XGB_V2_PARAMS)
        model.fit(X.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32))
        return "xgb_recursive", model

    if CatBoostRegressor is not None:
        model = CatBoostRegressor(**CATBOOST_V2_PARAMS)
        model.fit(X.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32))
        return "catboost_recursive", model

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32))
    return "rf_recursive", model


def fit_and_forecast_holdout(
    sales: pd.DataFrame,
    promotions: pd.DataFrame,
    web_traffic: Optional[pd.DataFrame] = None,
    inventory: Optional[pd.DataFrame] = None,
    business_daily: Optional[pd.DataFrame] = None,
    business_configs: Optional[Sequence[str]] = None,
    holdout_start: str = "2022-01-01",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float], List[str]]:
    """Train on history before holdout_start and recursively forecast holdout."""
    train = sales[sales["Date"] < holdout_start].copy()
    holdout = sales[sales["Date"] >= holdout_start].copy()
    train_cutoff = train["Date"].max()
    promo_profile = build_promo_profile(promotions, train_cutoff=train_cutoff)
    traffic_profile = (
        build_traffic_profile(web_traffic, train_cutoff=train_cutoff)
        if web_traffic is not None
        else None
    )
    inventory_profile = (
        build_inventory_profile(inventory, train_cutoff=train_cutoff)
        if inventory is not None
        else None
    )
    if business_configs is None:
        business_configs = ["V0_current"] if business_daily is None or business_daily.empty else list(BUSINESS_CONFIGS)

    all_rows = []
    best_score = np.inf
    best_weights: Dict[str, float] = {}
    best_feature_cols: List[str] = []

    train_business_daily = None
    if business_daily is not None and not business_daily.empty:
        train_business_daily = business_daily[pd.to_datetime(business_daily["Date"]) < pd.Timestamp(holdout_start)].copy()

    for config_name in business_configs:
        if verbose:
            print(f"Evaluating feature config: {config_name}")
        business_groups = BUSINESS_CONFIGS.get(config_name, [])
        business_profile = build_business_profile(train_business_daily, business_groups)

        frame, feature_cols = make_supervised_frame(
            train,
            promo_profile,
            traffic_profile=traffic_profile,
            inventory_profile=inventory_profile,
            business_daily=train_business_daily,
            business_profile=business_profile,
            business_groups=business_groups,
        )
        X, feature_fill_values = impute_feature_frame(frame[feature_cols])
        y = np.log1p(frame["Revenue"].clip(lower=0.0))

        primary_name, primary_model = train_primary_model(X, y)
        secondary_name, secondary_model = train_secondary_model(X, y)

        primary_pred = recursive_forecast(
            primary_model,
            train,
            holdout["Date"],
            feature_cols,
            promo_profile,
            traffic_profile=traffic_profile,
            inventory_profile=inventory_profile,
            business_daily=train_business_daily,
            business_profile=business_profile,
            business_groups=business_groups,
            feature_fill_values=feature_fill_values,
            predict_log_target=True,
        )
        secondary_pred = recursive_forecast(
            secondary_model,
            train,
            holdout["Date"],
            feature_cols,
            promo_profile,
            traffic_profile=traffic_profile,
            inventory_profile=inventory_profile,
            business_daily=train_business_daily,
            business_profile=business_profile,
            business_groups=business_groups,
            feature_fill_values=feature_fill_values,
            predict_log_target=True,
        )
        base_pred = seasonal_growth_baseline(train, holdout["Date"], "Revenue")

        candidates = {
            "seasonal_growth": base_pred,
            primary_name: primary_pred,
            secondary_name: secondary_pred,
        }
        rows = []
        for name, pred in candidates.items():
            rows.append({"config": config_name, "model": name, **compute_metrics(holdout["Revenue"].values, pred)})

        best_two = (
            pd.DataFrame(rows)
            .sort_values("MAE")
            .head(2)["model"]
            .tolist()
        )
        inv_mae = {r["model"]: 1 / max(r["MAE"], 1.0) for r in rows if r["model"] in best_two}
        total = sum(inv_mae.values())
        weights = {name: inv_mae.get(name, 0.0) / total for name in candidates}
        ens_pred = sum(weights[name] * candidates[name] for name in candidates)
        ens_metrics = compute_metrics(holdout["Revenue"].values, ens_pred)
        rows.append({"config": config_name, "model": "weighted_top2", **ens_metrics})
        all_rows.extend(rows)

        if ens_metrics["MAE"] < best_score:
            best_score = ens_metrics["MAE"]
            best_weights = weights
            best_feature_cols = feature_cols

    scores = pd.DataFrame(all_rows).sort_values(["MAE", "config", "model"]).reset_index(drop=True)
    return scores, best_weights, best_feature_cols


def train_final_and_predict(
    sales: pd.DataFrame,
    sample_sub: pd.DataFrame,
    promotions: pd.DataFrame,
    weights: Dict[str, float],
    web_traffic: Optional[pd.DataFrame] = None,
    inventory: Optional[pd.DataFrame] = None,
    business_daily: Optional[pd.DataFrame] = None,
    business_config: str = "V0_current",
    output_path: str = "outputs/submission.csv",
) -> Tuple[pd.DataFrame, ForecastArtifacts]:
    """Train final models on all history and produce a submission DataFrame."""
    train_cutoff = sales["Date"].max()
    promo_profile = build_promo_profile(promotions, train_cutoff=train_cutoff)
    traffic_profile = (
        build_traffic_profile(web_traffic, train_cutoff=train_cutoff)
        if web_traffic is not None
        else None
    )
    inventory_profile = (
        build_inventory_profile(inventory, train_cutoff=train_cutoff)
        if inventory is not None
        else None
    )
    business_groups = BUSINESS_CONFIGS.get(business_config, [])
    business_profile = build_business_profile(business_daily, business_groups)
    frame, feature_cols = make_supervised_frame(
        sales,
        promo_profile,
        traffic_profile=traffic_profile,
        inventory_profile=inventory_profile,
        business_daily=business_daily,
        business_profile=business_profile,
        business_groups=business_groups,
    )
    X, feature_fill_values = impute_feature_frame(frame[feature_cols])
    y = np.log1p(frame["Revenue"].clip(lower=0.0))

    primary_name, primary_model = train_primary_model(X, y)
    secondary_name, secondary_model = train_secondary_model(X, y)

    future_dates = pd.to_datetime(sample_sub["Date"])
    preds = {
        "seasonal_growth": seasonal_growth_baseline(sales, future_dates, "Revenue"),
        primary_name: recursive_forecast(
            primary_model,
            sales,
            future_dates,
            feature_cols,
            promo_profile,
            traffic_profile=traffic_profile,
            inventory_profile=inventory_profile,
            business_daily=business_daily,
            business_profile=business_profile,
            business_groups=business_groups,
            feature_fill_values=feature_fill_values,
            predict_log_target=True,
        ),
        secondary_name: recursive_forecast(
            secondary_model,
            sales,
            future_dates,
            feature_cols,
            promo_profile,
            traffic_profile=traffic_profile,
            inventory_profile=inventory_profile,
            business_daily=business_daily,
            business_profile=business_profile,
            business_groups=business_groups,
            feature_fill_values=feature_fill_values,
            predict_log_target=True,
        ),
    }
    revenue_pred = sum(weights.get(name, 0.0) * pred for name, pred in preds.items())
    if not np.any(revenue_pred):
        revenue_pred = preds[primary_name]
    revenue_pred = calibrate_monthly_predictions(revenue_pred, future_dates, sales, "Revenue")

    cogs_pred = predict_cogs(revenue_pred, sample_sub["Date"], sales)
    submission = sample_sub[["Date"]].copy()
    submission["Revenue"] = np.maximum(revenue_pred, 0.0).round(2)
    submission["COGS"] = np.maximum(cogs_pred, 0.0).round(2)
    submission["Date"] = pd.to_datetime(submission["Date"]).dt.strftime("%Y-%m-%d")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    artifacts = ForecastArtifacts(
        feature_cols=feature_cols,
        holdout_scores=pd.DataFrame(),
        primary_model=primary_model,
        secondary_model=secondary_model,
        ensemble_weights=weights,
    )
    return submission, artifacts
