"""
src/utils.py
============
Helper functions: visualization, submission builder, sanity checks.

Author : Team Gridbreakers
Seed   : 42
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import List, Dict, Optional


# ── Theme ─────────────────────────────────────────────────────────────────────
COLORS = {
    "primary":   "#1b6ca8",
    "secondary": "#52b788",
    "accent":    "#f4a261",
    "danger":    "#e76f51",
    "success":   "#2d6a4f",
    "neutral":   "#6c757d",
    "purple":    "#9b5de5",
}

def setup_plot_style():
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "figure.dpi":        120,
    })


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_cv_results(cv_df: pd.DataFrame, title: str = "", save_path: Optional[str] = None):
    """Visualize cross-validation results."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Cross-Validation Results — {title}", fontsize=13, fontweight="bold")

    metrics = ["MAE", "RMSE", "R2"]
    labels  = ["MAE (↓ better)", "RMSE (↓ better)", "R² (↑ better)"]
    palette = [COLORS["danger"], COLORS["accent"], COLORS["success"]]

    for ax, m, lbl, col in zip(axes, metrics, labels, palette):
        vals = cv_df[m].values
        folds = [f"Fold {i+1}" for i in range(len(vals))]
        bars = ax.bar(folds, vals, color=col, alpha=0.85)
        ax.axhline(vals.mean(), color="black", ls="--", lw=1.5,
                   label=f"Mean: {vals.mean():.3f}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() * 1.01,
                    f"{v:,.0f}" if m != "R2" else f"{v:.3f}",
                    ha="center", fontsize=9, fontweight="bold")
        ax.set_title(lbl)
        ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.show()


def plot_actual_vs_predicted(
    dates, y_true, y_pred,
    title: str = "Actual vs Predicted",
    save_path: Optional[str] = None,
):
    """Line chart actual vs predicted với residuals."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(title, fontsize=13, fontweight="bold")

    axes[0].plot(dates, y_true / 1e6, color=COLORS["neutral"],
                 lw=1.2, alpha=0.8, label="Actual")
    axes[0].plot(dates, y_pred / 1e6, color=COLORS["primary"],
                 lw=1.5, ls="--", label="Predicted", alpha=0.9)
    axes[0].set_ylabel("Revenue (Triệu VNĐ)")
    axes[0].legend(fontsize=10)

    residuals = y_true - y_pred
    axes[1].bar(dates, residuals / 1e6, color=COLORS["accent"], alpha=0.6, width=1.0)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("Residual (Triệu VNĐ)")
    axes[1].set_xlabel("Date")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.show()


def plot_ensemble_forecast(
    train: pd.DataFrame,
    test_dates, lgbm_pred, xgb_pred,
    prophet_pred=None,
    ensemble_pred=None,
    w_lgbm: float = 0.0,
    w_xgb: float = 0.0,
    w_prophet: float = 0.0,
    save_path: Optional[str] = None,
):
    """4-panel ensemble forecast dashboard."""
    if ensemble_pred is None:
        raise ValueError("ensemble_pred is required")

    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle("Ensemble Forecast - Revenue 2023-2024", fontsize=15, fontweight="bold")

    # (A) Historical + Forecast
    ax_a = fig.add_subplot(gs[0, :])
    monthly_hist = train.set_index("Date")["Revenue"].resample("M").sum() / 1e6
    ax_a.fill_between(monthly_hist.index, monthly_hist.values, alpha=0.15, color=COLORS["neutral"])
    ax_a.plot(monthly_hist.index, monthly_hist.values,
              color=COLORS["neutral"], lw=1.5, alpha=0.7, label="Historical (monthly)")

    test_df_m = (
        pd.DataFrame({"Date": test_dates, "Revenue": ensemble_pred})
        .set_index(pd.to_datetime(test_dates))["Revenue"]
        .resample("M").sum() / 1e6
    )
    ax_a.fill_between(test_df_m.index, test_df_m.values, alpha=0.4, color=COLORS["primary"])
    ax_a.plot(test_df_m.index, test_df_m.values,
              color=COLORS["primary"], lw=2.5, marker="o", ms=4,
              label=f"Ensemble Forecast 2023-2024")
    ax_a.set_title("(A) Historical Revenue + 18-Month Ensemble Forecast")
    ax_a.set_ylabel("Revenue (Triệu VNĐ)")
    ax_a.legend(fontsize=10)

    # (B) Model comparison daily
    ax_b = fig.add_subplot(gs[1, 0])
    n = len(ensemble_pred)
    ax_b.plot(test_dates[:n], lgbm_pred[:n] / 1e6, color=COLORS["accent"],
              lw=0.8, alpha=0.75, label=f"LightGBM (w={w_lgbm:.0%})")
    ax_b.plot(test_dates[:n], xgb_pred[:n] / 1e6, color=COLORS["secondary"],
              lw=0.8, alpha=0.75, label=f"XGBoost (w={w_xgb:.0%})")
    if prophet_pred is not None and w_prophet > 0:
        ax_b.plot(test_dates[:n], prophet_pred[:n] / 1e6, color=COLORS["purple"],
                  lw=0.8, alpha=0.75, label=f"Prophet (w={w_prophet:.0%})")
    ax_b.plot(test_dates[:n], ensemble_pred / 1e6, color=COLORS["primary"],
              lw=2.2, label="Ensemble", zorder=5)
    ax_b.set_title("(B) Model Comparison — Daily Forecast")
    ax_b.set_ylabel("Revenue (Triệu VNĐ)")
    ax_b.legend(fontsize=8)

    # (C) Monthly forecast bar
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.bar(test_df_m.index, test_df_m.values,
             color=COLORS["primary"], alpha=0.8, width=20)
    for x, y in zip(test_df_m.index, test_df_m.values):
        ax_c.text(x, y + y * 0.01, f"{y:.0f}M", ha="center", fontsize=8, fontweight="bold")
    ax_c.set_title("(C) Monthly Forecast Breakdown")
    ax_c.set_ylabel("Revenue (Triệu VNĐ)")
    ax_c.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.show()


def plot_feature_importance(
    feat_importances: pd.Series,
    title: str = "Feature Importance",
    top_n: int = 25,
    save_path: Optional[str] = None,
):
    """Horizontal bar chart of feature importances."""
    top = feat_importances.nlargest(top_n).sort_values()
    colors = [
        COLORS["danger"] if v == top.max() else
        COLORS["accent"] if v >= top.quantile(0.75) else
        COLORS["neutral"]
        for v in top.values
    ]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    ax.barh(top.index, top.values, color=colors, alpha=0.85)
    ax.set_title(f"{title} — Top {top_n} Features", fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.show()


# ── Sanity Checks ─────────────────────────────────────────────────────────────

def run_submission_checks(
    submission: pd.DataFrame,
    sample_sub: pd.DataFrame,
) -> bool:
    """
    Chạy 7 sanity checks trước khi submit.
    Trả về True nếu tất cả pass.
    """
    errors   = []
    warnings_ = []

    print("=" * 55)
    print("  SUBMISSION SANITY CHECKS")
    print("=" * 55)

    # 1. Columns
    required = {"Date", "Revenue", "COGS"}
    missing_cols = required - set(submission.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    else:
        print("Columns: Date, Revenue, COGS present")

    # 2. Row count
    if len(submission) != len(sample_sub):
        errors.append(f"Row count {len(submission)} ≠ expected {len(sample_sub)}")
    else:
        print(f"Row count: {len(submission)} (matches sample_submission)")

    # 3. Date order
    sub_dates    = pd.to_datetime(submission["Date"]).values
    sample_dates = pd.to_datetime(sample_sub["Date"]).values
    n_check = min(len(sub_dates), len(sample_dates))
    if not (sub_dates[:n_check] == sample_dates[:n_check]).all():
        errors.append("Date order does NOT match sample_submission")
    else:
        print("Date order matches sample_submission exactly")

    # 4. Negative values
    neg_rev  = (submission["Revenue"] < 0).sum()
    neg_cogs = (submission["COGS"] < 0).sum()
    if neg_rev > 0:
        errors.append(f"{neg_rev} negative Revenue values")
    else:
        print("No negative Revenue values")
    if neg_cogs > 0:
        errors.append(f"{neg_cogs} negative COGS values")
    else:
        print("No negative COGS values")

    # 5. NaN check
    nan_total = submission[["Revenue", "COGS"]].isna().sum().sum()
    if nan_total > 0:
        errors.append(f"{nan_total} NaN values found")
    else:
        print("No NaN values")

    # 6. COGS < Revenue
    cogs_gt_rev = (submission["COGS"] > submission["Revenue"]).sum()
    if cogs_gt_rev > 0:
        warnings_.append(f"{cogs_gt_rev} days COGS > Revenue (unusual)")
    else:
        print("COGS <= Revenue for all days")

    # 7. Value range reasonability
    mean_rev = submission["Revenue"].mean()
    if mean_rev < 100 or mean_rev > 1e10:
        warnings_.append(f"Revenue mean={mean_rev:,.0f} seems unusual")
    else:
        print(f"Revenue range reasonable (mean={mean_rev:,.0f})")

    print()
    if warnings_:
        for w in warnings_:
            print(f"Warning: {w}")
    if errors:
        for e in errors:
            print(f"Error: {e}")
        print("\nCHECKS FAILED - Fix errors before submitting!")
        return False
    else:
        print("All checks PASSED - Ready to submit!")
        return True


# ── Submission Builder ────────────────────────────────────────────────────────

def build_submission(
    sample_sub: pd.DataFrame,
    revenue_pred: np.ndarray,
    cogs_pred: np.ndarray,
    output_path: str = "dataset/submission.csv",
) -> pd.DataFrame:
    """Build và save submission.csv theo đúng format."""
    n = min(len(sample_sub), len(revenue_pred), len(cogs_pred))
    sub = sample_sub[["Date"]].head(n).copy()
    sub["Revenue"] = revenue_pred[:n].round(2)
    sub["COGS"]    = cogs_pred[:n].round(2)
    sub["Date"]    = pd.to_datetime(sub["Date"]).dt.strftime("%Y-%m-%d")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)
    print(f"Saved {len(sub)} rows to {output_path}")
    return sub


# ── Model Summary Card ────────────────────────────────────────────────────────

def print_model_card(
    ensemble_summary: str,
    n_features: int,
    cv_mae: float,
    cv_rmse: float,
    cv_r2: float,
    n_rows: int,
):
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║           GRIDBREAKERS — MODEL SUMMARY CARD             ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Strategy    : {ensemble_summary:<43}║")
    print(f"║  Features    : {n_features:<2} engineered features                  ║")
    print(f"║  CV Strategy : 5-Fold TimeSeriesSplit (no random split)  ║")
    print(f"║  CV MAE      : {cv_mae:>12,.0f} VNĐ                       ║")
    print(f"║  CV RMSE     : {cv_rmse:>12,.0f} VNĐ                       ║")
    print(f"║  CV R²       : {cv_r2:>12.3f}                           ║")
    print(f"║  Random seed : 42 (all models)                           ║")
    print(f"║  Submission  : {n_rows} rows (01/01/2023 → 01/07/2024)     ║")
    print("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    print("utils.py loaded OK")
    setup_plot_style()
