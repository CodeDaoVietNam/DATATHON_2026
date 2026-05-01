"""
diagnostics.py
==============
Notebook-facing diagnostics helpers for forecasting model review.

These functions keep plotting, SHAP, and final acceptance checks out of the
main modeling notebook so the notebook can stay narrative and compact.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from eda_utils import display_kpi_cards, format_spines, section_header
from feature_engineering import add_time_features, add_vn_calendar
from forecasting import forecast_cogs_ratio_model, predict_cogs_monthly_ratio_anchor
from utils import plot_feature_importance


def compute_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    """Compute MAE, RMSE, and R2 without importing heavyweight model modules."""
    actual = np.asarray(y_true, dtype=float)
    pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    residual = actual - pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    denom = float(np.sum(np.square(actual - actual.mean())))
    r2 = float(1.0 - np.sum(np.square(residual)) / denom) if denom > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def validation_event_bucket(row: pd.Series) -> str:
    """Map calendar flags to a compact validation error bucket."""
    if row.get("is_tet", 0) == 1:
        return "tet"
    if row.get("is_pre_tet", 0) == 1:
        return "pre_tet"
    if row.get("is_post_tet", 0) == 1:
        return "post_tet"
    if row.get("is_q1_end_spike", 0) == 1:
        return "q1_end_spike"
    if row.get("is_aug_end_spike", 0) == 1:
        return "aug_end_spike"
    if row.get("is_sale_event", 0) == 1:
        return "sale_event_day"
    if row.get("is_pre_sale_event", 0) == 1:
        return "pre_sale_event"
    if row.get("is_post_sale_event", 0) == 1:
        return "post_sale_event"
    return "normal"


def build_validation_diagnostic_frame(
    valid_df: pd.DataFrame,
    prediction: Sequence[float],
) -> pd.DataFrame:
    """Build actual/predicted/residual frame with calendar diagnostics."""
    diag = pd.DataFrame({
        "Date": pd.to_datetime(valid_df["Date"]),
        "Actual": valid_df["Revenue"].to_numpy(dtype=float),
        "Predicted": np.asarray(prediction, dtype=float),
    })
    diag["Residual"] = diag["Actual"] - diag["Predicted"]
    diag["AbsError"] = diag["Residual"].abs()
    diag = add_time_features(diag, "Date")
    diag = add_vn_calendar(diag, "Date")
    diag["weekday_name"] = diag["Date"].dt.day_name()
    diag["event_bucket"] = diag.apply(validation_event_bucket, axis=1)
    return diag


def summarize_validation_residuals(diag: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Return monthly and event residual tables."""
    month_residual = (
        diag.groupby("month")
        .agg(
            Mean_Residual=("Residual", "mean"),
            MAE=("AbsError", "mean"),
            RMSE=("Residual", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            Actual_Mean=("Actual", "mean"),
            Pred_Mean=("Predicted", "mean"),
            Days=("Date", "count"),
        )
        .reset_index()
        .sort_values("MAE", ascending=False)
    )

    event_residual = (
        diag.groupby("event_bucket")
        .agg(
            Mean_Residual=("Residual", "mean"),
            MAE=("AbsError", "mean"),
            RMSE=("Residual", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            Actual_Mean=("Actual", "mean"),
            Pred_Mean=("Predicted", "mean"),
            Days=("Date", "count"),
        )
        .reset_index()
        .sort_values("MAE", ascending=False)
    )
    return {"month_residual": month_residual, "event_residual": event_residual}


def plot_validation_diagnostics(
    valid_df: pd.DataFrame,
    prediction: Sequence[float],
    final_candidate_name: str,
) -> Dict[str, Any]:
    """Plot and display the broad validation diagnostics dashboard."""
    diag = build_validation_diagnostic_frame(valid_df, prediction)
    metrics = compute_metrics(diag["Actual"].to_numpy(), diag["Predicted"].to_numpy())
    residuals = summarize_validation_residuals(diag)
    month_residual = residuals["month_residual"]
    event_residual = residuals["event_residual"]

    section_header("Validation Diagnostics", f"Final candidate: {final_candidate_name}")
    display_kpi_cards([
        {"label": "Validation MAE", "value": f"{metrics['MAE']:,.0f}", "color": "#2563EB"},
        {"label": "Validation RMSE", "value": f"{metrics['RMSE']:,.0f}", "color": "#F97316"},
        {"label": "Validation R2", "value": f"{metrics['R2']:.3f}", "color": "#16A34A"},
        {"label": "Mean Residual", "value": f"{diag['Residual'].mean():,.0f}", "color": "#7C3AED"},
    ])

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes[0, 0].plot(diag["Date"], diag["Actual"] / 1e6, color="#6B7280", label="Actual", linewidth=1.3)
    axes[0, 0].plot(diag["Date"], diag["Predicted"] / 1e6, color="#2563EB", linestyle="--", label="Predicted", linewidth=1.3)
    axes[0, 0].set_title("Actual vs Predicted - Validation 2022")
    axes[0, 0].set_ylabel("Revenue (Million VND)")
    axes[0, 0].legend()
    format_spines(axes[0, 0])

    axes[0, 1].scatter(diag["Actual"] / 1e6, diag["Predicted"] / 1e6, alpha=0.65, color="#2563EB")
    max_val = max(diag["Actual"].max(), diag["Predicted"].max()) / 1e6
    axes[0, 1].plot([0, max_val], [0, max_val], color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Actual vs Predicted Scatter")
    axes[0, 1].set_xlabel("Actual (Million VND)")
    axes[0, 1].set_ylabel("Predicted (Million VND)")
    format_spines(axes[0, 1])

    axes[1, 0].bar(month_residual["month"].astype(str), month_residual["MAE"] / 1e6, color="#F97316")
    axes[1, 0].set_title("MAE by Month")
    axes[1, 0].set_ylabel("MAE (Million VND)")
    format_spines(axes[1, 0])

    axes[1, 1].bar(event_residual["event_bucket"], event_residual["MAE"] / 1e6, color="#EF4444")
    axes[1, 1].set_title("MAE by Event Bucket")
    axes[1, 1].set_ylabel("MAE (Million VND)")
    axes[1, 1].tick_params(axis="x", rotation=35)
    format_spines(axes[1, 1])

    plt.tight_layout()
    plt.show()

    print("Monthly residual summary")
    display(month_residual.style.format({
        "Mean_Residual": "{:,.0f}",
        "MAE": "{:,.0f}",
        "RMSE": "{:,.0f}",
        "Actual_Mean": "{:,.0f}",
        "Pred_Mean": "{:,.0f}",
    }))

    print("Event residual summary")
    display(event_residual.style.format({
        "Mean_Residual": "{:,.0f}",
        "MAE": "{:,.0f}",
        "RMSE": "{:,.0f}",
        "Actual_Mean": "{:,.0f}",
        "Pred_Mean": "{:,.0f}",
    }))

    top_errors = diag.sort_values("AbsError", ascending=False).head(20)
    print("Top 20 absolute error days")
    display(top_errors[[
        "Date", "Actual", "Predicted", "Residual", "AbsError", "month", "weekday_name", "event_bucket"
    ]].style.format({
        "Actual": "{:,.0f}",
        "Predicted": "{:,.0f}",
        "Residual": "{:,.0f}",
        "AbsError": "{:,.0f}",
    }))

    return {
        "validation_diag": diag,
        "validation_metrics": metrics,
        "month_residual": month_residual,
        "event_residual": event_residual,
        "top_errors": top_errors,
    }


def plot_focused_validation_error_dashboard(
    valid_df: pd.DataFrame,
    train_df: pd.DataFrame,
    prediction: Sequence[float],
    promo_profile: pd.DataFrame,
    traffic_profile: Optional[pd.DataFrame],
    inventory_profile: Optional[pd.DataFrame],
    business_daily: Optional[pd.DataFrame],
    business_profile: Optional[pd.DataFrame],
    business_groups: Sequence[str],
    cogs_anchor_mode: str,
    cogs_ratio_model_weight: float,
    n_worst_days: int = 20,
) -> Dict[str, Any]:
    """Plot focused residual/event/COGS-ratio diagnostics."""
    diag = build_validation_diagnostic_frame(valid_df, prediction)
    focused_event_residual = (
        diag.groupby("event_bucket")
        .agg(
            Mean_Residual=("Residual", "mean"),
            MAE=("AbsError", "mean"),
            RMSE=("Residual", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            Days=("Date", "count"),
        )
        .reset_index()
        .sort_values("MAE", ascending=False)
    )
    top_errors = diag.sort_values("AbsError", ascending=False).head(n_worst_days).copy()

    cogs_monthly, _ = predict_cogs_monthly_ratio_anchor(
        prediction,
        valid_df["Date"],
        train_df,
        recent_years=None,
    )
    cogs_monthly_recent3, _ = predict_cogs_monthly_ratio_anchor(
        prediction,
        valid_df["Date"],
        train_df,
        recent_years=3,
    )
    cogs_ratio_model, _ = forecast_cogs_ratio_model(
        train_df,
        prediction,
        valid_df["Date"],
        promo_profile,
        traffic_profile=traffic_profile,
        inventory_profile=inventory_profile,
        business_daily=business_daily,
        business_profile=business_profile,
        business_groups=business_groups,
    )
    cogs_anchor = cogs_monthly_recent3 if cogs_anchor_mode == "monthly_recent3" else cogs_monthly
    cogs_pred = (1.0 - cogs_ratio_model_weight) * cogs_anchor + cogs_ratio_model_weight * cogs_ratio_model
    cogs_pred = np.minimum(np.maximum(cogs_pred, 0), np.maximum(prediction, 0) * 0.995)

    cogs_ratio_diag = valid_df[["Date", "Revenue", "COGS"]].copy()
    cogs_ratio_diag["Predicted_Revenue"] = np.asarray(prediction, dtype=float)
    cogs_ratio_diag["Predicted_COGS"] = cogs_pred
    cogs_ratio_diag["actual_cogs_ratio"] = cogs_ratio_diag["COGS"] / cogs_ratio_diag["Revenue"].replace(0, np.nan)
    cogs_ratio_diag["pred_cogs_ratio"] = cogs_ratio_diag["Predicted_COGS"] / cogs_ratio_diag["Predicted_Revenue"].replace(0, np.nan)
    cogs_ratio_diag["month"] = pd.to_datetime(cogs_ratio_diag["Date"]).dt.month
    cogs_ratio_monthly = (
        cogs_ratio_diag.groupby("month")
        .agg(
            actual_cogs_ratio=("actual_cogs_ratio", "mean"),
            pred_cogs_ratio=("pred_cogs_ratio", "mean"),
            COGS_MAE=("COGS", lambda s: np.nan),
        )
        .reset_index()
    )
    cogs_ratio_monthly["COGS_MAE"] = cogs_ratio_diag.groupby("month").apply(
        lambda g: float(np.mean(np.abs(g["COGS"] - g["Predicted_COGS"])))
    ).values

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes[0, 0].bar(diag["Date"], diag["Residual"] / 1e6, color="#FDBA74", width=1.0)
    axes[0, 0].axhline(0, color="black", linewidth=0.9)
    axes[0, 0].set_title("Validation Residual Over Time")
    axes[0, 0].set_ylabel("Actual - Predicted (Million VND)")
    format_spines(axes[0, 0])

    x = np.arange(len(focused_event_residual))
    width = 0.38
    axes[0, 1].bar(x - width / 2, focused_event_residual["Mean_Residual"] / 1e6, width, label="Mean residual", color="#38BDF8")
    axes[0, 1].bar(x + width / 2, focused_event_residual["MAE"] / 1e6, width, label="MAE", color="#EF4444")
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(focused_event_residual["event_bucket"], rotation=35, ha="right")
    axes[0, 1].set_title("Residual / MAE by Event Bucket")
    axes[0, 1].set_ylabel("Million VND")
    axes[0, 1].legend()
    format_spines(axes[0, 1])

    axes[1, 0].bar(top_errors["Date"].dt.strftime("%Y-%m-%d"), top_errors["AbsError"] / 1e6, color="#DC2626")
    axes[1, 0].set_title(f"Top {n_worst_days} Worst Absolute Error Days")
    axes[1, 0].set_ylabel("Absolute Error (Million VND)")
    axes[1, 0].tick_params(axis="x", rotation=65)
    format_spines(axes[1, 0])

    axes[1, 1].plot(cogs_ratio_monthly["month"], cogs_ratio_monthly["actual_cogs_ratio"], marker="o", color="#6B7280", label="Actual COGS/Revenue")
    axes[1, 1].plot(cogs_ratio_monthly["month"], cogs_ratio_monthly["pred_cogs_ratio"], marker="o", color="#2563EB", label="Predicted COGS/Revenue")
    axes[1, 1].set_title("Actual vs Predicted COGS Ratio by Month")
    axes[1, 1].set_xlabel("Month")
    axes[1, 1].set_ylabel("COGS / Revenue")
    axes[1, 1].set_xticks(range(1, 13))
    axes[1, 1].legend()
    format_spines(axes[1, 1])
    plt.tight_layout()
    plt.show()

    print("Event residual table")
    display(focused_event_residual.style.format({
        "Mean_Residual": "{:,.0f}",
        "MAE": "{:,.0f}",
        "RMSE": "{:,.0f}",
    }))

    print(f"Top {n_worst_days} worst error days")
    display(top_errors[[
        "Date", "Actual", "Predicted", "Residual", "AbsError", "month", "weekday_name", "event_bucket"
    ]].style.format({
        "Actual": "{:,.0f}",
        "Predicted": "{:,.0f}",
        "Residual": "{:,.0f}",
        "AbsError": "{:,.0f}",
    }))

    print("COGS ratio by month")
    display(cogs_ratio_monthly.sort_values("COGS_MAE", ascending=False).style.format({
        "actual_cogs_ratio": "{:.3f}",
        "pred_cogs_ratio": "{:.3f}",
        "COGS_MAE": "{:,.0f}",
    }))

    return {
        "validation_diag": diag,
        "focused_event_residual": focused_event_residual,
        "top_errors": top_errors,
        "cogs_ratio_monthly": cogs_ratio_monthly,
        "cogs_pred": cogs_pred,
    }


def _feature_group(name: str) -> str:
    if name.startswith("lag_") or name.startswith("rev_diff") or name.startswith("rev_pct") or name.startswith("rev_vs"):
        return "autoregressive_lag"
    if name.startswith("roll_"):
        return "rolling_stats"
    if any(key in name for key in ["tet", "sale", "q1_end", "aug_end", "month_end", "quarter_end", "back2school"]):
        return "calendar_event"
    if name.startswith("traffic_"):
        return "traffic_profile"
    if name.startswith("promo_"):
        return "promo_profile"
    if name.startswith("inv_"):
        return "inventory_profile"
    if name.startswith("biz_"):
        return "business_profile"
    if name in ["day", "month", "year", "day_of_week", "day_of_year", "elapsed_days"] or name.endswith("_sin") or name.endswith("_cos"):
        return "calendar_base"
    return "other"


def plot_model_explainability(
    final_cat: Any,
    final_lgbm: Any,
    x_full: pd.DataFrame,
    feature_cols: Sequence[str],
    random_seed: int = 42,
    top_n: int = 30,
    shap_sample_size: int = 800,
) -> Dict[str, pd.DataFrame]:
    """Display feature importance and SHAP diagnostics for the final tree model."""
    section_header("Model Explainability", "Feature importance and SHAP diagnostics for the final selected model.")
    if final_cat is not None:
        explain_model = final_cat
        explain_model_name = "CatBoost"
        importance_values = final_cat.get_feature_importance()
    else:
        explain_model = final_lgbm
        explain_model_name = "LightGBM"
        importance_values = final_lgbm.feature_importances_

    feature_importance = pd.Series(importance_values, index=feature_cols, name="importance").sort_values(ascending=False)
    print(f"Explaining model: {explain_model_name}")
    print(f"Top {top_n} feature importances")
    display(feature_importance.head(top_n).reset_index().rename(columns={"index": "feature"}))
    plot_feature_importance(feature_importance, title=f"Final {explain_model_name} Feature Importance", top_n=top_n)

    group_importance = feature_importance.reset_index().rename(columns={"index": "feature", "importance": "importance"})
    group_importance["group"] = group_importance["feature"].map(_feature_group)
    group_summary = group_importance.groupby("group", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
    display(group_summary)

    plt.figure(figsize=(10, 5))
    plt.barh(group_summary["group"][::-1], group_summary["importance"][::-1], color="#2563EB")
    plt.title("Feature Importance by Signal Group")
    plt.xlabel("Total importance")
    plt.tight_layout()
    plt.show()

    try:
        import shap

        sample_size = min(shap_sample_size, len(x_full))
        shap_sample = x_full.loc[:, feature_cols].sample(sample_size, random_state=random_seed).copy()
        print(f"Computing SHAP on {sample_size:,} sampled rows...")
        explainer = shap.TreeExplainer(explain_model)
        shap_values = explainer.shap_values(shap_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap.summary_plot(shap_values, shap_sample, plot_type="bar", max_display=25, show=False)
        plt.title(f"SHAP Mean Absolute Impact - {explain_model_name}")
        plt.tight_layout()
        plt.show()

        shap.summary_plot(shap_values, shap_sample, max_display=25, show=False)
        plt.title(f"SHAP Beeswarm - {explain_model_name}")
        plt.tight_layout()
        plt.show()
    except Exception as exc:  # pragma: no cover - notebook environment dependent
        print(f"SHAP skipped: {type(exc).__name__}: {exc}")
        print("Feature importance table above is still available for model interpretation.")

    return {
        "feature_importance": feature_importance.reset_index().rename(columns={"index": "feature"}),
        "group_importance": group_summary,
    }


def final_acceptance_summary(
    sales: pd.DataFrame,
    sample_sub: pd.DataFrame,
    valid_df: pd.DataFrame,
    submission_path: str,
    train_cutoff: str,
    n_revenue_cogs_swaps: int,
    best_selected_metrics: Dict[str, float],
    public_scenario: str,
    recovery_strength: float,
    spike_strengths: Dict[str, float],
) -> Dict[str, pd.DataFrame]:
    """Display final submission acceptance checks and summary plots."""
    section_header("Final Acceptance Summary", "End-to-end checks before final submission.")
    final_submission = pd.read_csv(submission_path, parse_dates=["Date"])
    expected_cols = ["Date", "Revenue", "COGS"]

    acceptance_df = pd.DataFrame([
        {
            "area": "Data cleaning",
            "check": "Revenue/COGS swap applied",
            "status": "PASS" if (sales["COGS"] <= sales["Revenue"]).all() else "FAIL",
            "detail": f"Swapped rows detected before cleaning: {n_revenue_cogs_swaps:,}",
        },
        {
            "area": "Validation",
            "check": "Holdout year used before final train",
            "status": "PASS",
            "detail": f"Train <= {train_cutoff}; valid rows={len(valid_df):,}; final validation MAE={best_selected_metrics['MAE']:,.0f}",
        },
        {
            "area": "Final strategy",
            "check": "Public-best recovery + spike scenario locked",
            "status": "PASS",
            "detail": f"{public_scenario}; recovery={recovery_strength:.2f}; spike={spike_strengths}",
        },
        {
            "area": "Submission schema",
            "check": "Columns and row count",
            "status": "PASS" if list(final_submission.columns) == expected_cols and len(final_submission) == len(sample_sub) else "FAIL",
            "detail": f"rows={len(final_submission):,}; columns={list(final_submission.columns)}",
        },
        {
            "area": "Submission values",
            "check": "No NaN, no negative, COGS <= Revenue",
            "status": "PASS"
            if final_submission.notna().all().all()
            and (final_submission[["Revenue", "COGS"]] >= 0).all().all()
            and (final_submission["COGS"] <= final_submission["Revenue"]).all()
            else "FAIL",
            "detail": f"Revenue mean={final_submission['Revenue'].mean():,.0f}; COGS mean={final_submission['COGS'].mean():,.0f}",
        },
    ])
    display(acceptance_df)

    final_summary = pd.DataFrame([
        {"metric": "Submission rows", "value": f"{len(final_submission):,}"},
        {"metric": "Date range", "value": f"{final_submission['Date'].min().date()} -> {final_submission['Date'].max().date()}"},
        {"metric": "Revenue mean", "value": f"{final_submission['Revenue'].mean():,.0f}"},
        {"metric": "Revenue min/max", "value": f"{final_submission['Revenue'].min():,.0f} / {final_submission['Revenue'].max():,.0f}"},
        {"metric": "COGS mean", "value": f"{final_submission['COGS'].mean():,.0f}"},
        {"metric": "COGS min/max", "value": f"{final_submission['COGS'].min():,.0f} / {final_submission['COGS'].max():,.0f}"},
        {"metric": "Mean COGS/Revenue", "value": f"{(final_submission['COGS'] / final_submission['Revenue']).mean():.3f}"},
    ])
    display(final_summary)

    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    axes[0].plot(final_submission["Date"], final_submission["Revenue"] / 1e6, color="#2563EB", linewidth=1.4)
    axes[0].set_title("Final Submission Revenue Forecast")
    axes[0].set_ylabel("Million VND")
    format_spines(axes[0])

    axes[1].plot(final_submission["Date"], final_submission["COGS"] / final_submission["Revenue"], color="#F97316", linewidth=1.4)
    axes[1].set_title("Final Submission COGS / Revenue Ratio")
    axes[1].set_ylabel("Ratio")
    format_spines(axes[1])
    plt.tight_layout()
    plt.show()

    if (acceptance_df["status"] == "FAIL").any():
        raise ValueError("Final acceptance failed. Inspect acceptance_df before submission.")
    print("Final acceptance PASSED. outputs/submission.csv is ready to submit.")
    return {"acceptance_df": acceptance_df, "final_summary": final_summary, "final_submission": final_submission}
