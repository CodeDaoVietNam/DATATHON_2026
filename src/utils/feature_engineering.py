"""
src/feature_engineering.py
==========================
Toàn bộ feature engineering pipeline cho Sales Forecasting.
Tất cả features được chứng minh từ EDA (ACF, STL, cross-correlation).
Không sử dụng dữ liệu ngoài.

Author : Team Gridbreakers
Seed   : 42
"""

import pandas as pd
import numpy as np
from typing import List, Optional


# ── VN Calendar Constants ─────────────────────────────────────────────────────
TET_DATES = pd.to_datetime([
    "2013-02-10", "2014-01-31", "2015-02-19", "2016-02-08",
    "2017-01-28", "2018-02-16", "2019-02-05", "2020-01-25",
    "2021-02-12", "2022-02-01", "2023-01-22", "2024-02-10",
])

MIN_TRAIN_DATE = pd.Timestamp("2012-07-04")


# ── Core Functions ────────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Thêm các đặc trưng thời gian cơ bản + cyclical encoding."""
    df = df.copy()
    dt = df[date_col].dt

    df["year"]          = dt.year
    df["month"]         = dt.month
    df["day"]           = dt.day
    df["day_of_week"]   = dt.dayofweek          # 0=Mon
    df["day_of_year"]   = dt.dayofyear
    df["week_of_year"]  = dt.isocalendar().week.astype(int)
    df["quarter"]       = dt.quarter
    df["is_weekend"]    = (dt.dayofweek >= 5).astype(int)
    df["is_month_end"]  = dt.is_month_end.astype(int)
    df["is_month_start"]= dt.is_month_start.astype(int)
    df["is_quarter_end"]= dt.is_quarter_end.astype(int)

    # Elapsed days — long-term trend proxy (confirmed from STL analysis)
    df["elapsed_days"]  = (df[date_col] - MIN_TRAIN_DATE).dt.days

    # Cyclical encoding — avoids ordinal assumption for periodic features
    df["month_sin"]     = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]     = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]       = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]       = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["doy_sin"]       = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]       = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["woy_sin"]       = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["woy_cos"]       = np.cos(2 * np.pi * df["week_of_year"] / 52)

    return df


def add_vn_calendar(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Thêm VN-specific calendar features.
    Tất cả được xác nhận từ STL seasonal component trong EDA.
    """
    df = df.copy()
    dates = df[date_col]

    # ── Tết windows ──────────────────────────────────────────────────────────
    pre_tet   = pd.Series(False, index=df.index)
    tet_day   = pd.Series(False, index=df.index)
    post_tet  = pd.Series(False, index=df.index)

    for t in TET_DATES:
        pre_tet  |= (dates >= t - pd.Timedelta(days=14)) & (dates < t)
        tet_day  |= (dates >= t) & (dates <= t + pd.Timedelta(days=3))
        post_tet |= (dates > t + pd.Timedelta(days=3)) & (dates <= t + pd.Timedelta(days=10))

    # Days-to-Tet (negative after Tết) — continuous signal richer than binary
    def days_to_nearest_tet(d):
        diffs = [(d - t).days for t in TET_DATES]
        return min(diffs, key=abs)

    df["is_pre_tet"]      = pre_tet.astype(int)
    df["is_tet"]          = tet_day.astype(int)
    df["is_post_tet"]     = post_tet.astype(int)
    df["days_to_tet"]     = dates.apply(days_to_nearest_tet).clip(-30, 30)

    # ── Super Sale events (from EDA: 3-5x spike) ─────────────────────────────
    df["is_1111"]         = ((dates.dt.month == 11) & (dates.dt.day == 11)).astype(int)
    df["is_1212"]         = ((dates.dt.month == 12) & (dates.dt.day == 12)).astype(int)
    df["is_pre_1111"]     = ((dates.dt.month == 11) & (dates.dt.day.between(8, 10))).astype(int)
    df["is_pre_1212"]     = ((dates.dt.month == 12) & (dates.dt.day.between(9, 11))).astype(int)

    # Black Friday: last Friday of November
    df["is_black_friday"] = (
        (dates.dt.month == 11) & (dates.dt.dayofweek == 4) & (dates.dt.day >= 23)
    ).astype(int)

    # ── Seasonal patterns ─────────────────────────────────────────────────────
    df["is_summer"]       = dates.dt.month.isin([5, 6, 7, 8]).astype(int)
    df["is_back2school"]  = ((dates.dt.month.isin([8, 9])) & (dates.dt.day < 15)).astype(int)
    df["is_womens_day"]   = ((dates.dt.month == 3) & (dates.dt.day == 8)).astype(int)
    df["is_valentines"]   = ((dates.dt.month == 2) & (dates.dt.day == 14)).astype(int)
    df["is_year_end"]     = ((dates.dt.month == 12) & (dates.dt.day >= 20)).astype(int)

    # ── Structural break ──────────────────────────────────────────────────────
    df["is_covid"]        = dates.dt.year.isin([2020, 2021]).astype(int)
    df["is_post_covid"]   = (dates.dt.year == 2022).astype(int)

    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "Revenue",
    lags: List[int] = [1, 2, 3, 7, 14, 21, 28, 30, 60, 90, 180, 365],
) -> pd.DataFrame:
    """
    Tạo lag features — confirmed significant from ACF analysis in EDA.
    QUAN TRỌNG: Luôn shift() để tránh data leakage.
    """
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "Revenue",
    windows: List[int] = [7, 14, 30, 60, 90],
) -> pd.DataFrame:
    """
    Tạo rolling statistics. shift(1) để tránh leakage (không dùng giá trị hiện tại).
    """
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    for w in windows:
        base = df[target_col].shift(1).rolling(w, min_periods=max(1, w // 2))
        df[f"roll_mean_{w}"]   = base.mean()
        df[f"roll_std_{w}"]    = base.std()
        df[f"roll_max_{w}"]    = base.max()
        df[f"roll_min_{w}"]    = base.min()
        df[f"roll_median_{w}"] = base.median()

    # Momentum features
    df["rev_diff_7"]  = df[target_col].diff(7)
    df["rev_diff_30"] = df[target_col].diff(30)
    df["rev_pct_7"]   = df[target_col].pct_change(7).replace([np.inf, -np.inf], 0).clip(-5, 5)
    df["rev_pct_30"]  = df[target_col].pct_change(30).replace([np.inf, -np.inf], 0).clip(-5, 5)

    # Ratio: current vs rolling average (speed vs trend)
    mean_30 = df[target_col].shift(1).rolling(30, min_periods=10).mean()
    df["rev_vs_ma30"] = (df[target_col].shift(1) / (mean_30 + 1e-8)).clip(0, 5)

    return df


def add_promo_features(
    df: pd.DataFrame,
    promotions: pd.DataFrame,
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Tạo daily promotion matrix từ promotions.csv.
    Không dùng conversion_rate (không tồn tại trong dataset).
    """
    df = df.copy()
    dates = df[date_col].values

    active_counts  = []
    max_discounts  = []
    has_stackable  = []
    pct_promo_cnt  = []
    fixed_promo_cnt= []

    for d in dates:
        d = pd.Timestamp(d)
        active = promotions[
            (promotions["start_date"] <= d) & (promotions["end_date"] >= d)
        ]
        n = len(active)
        active_counts.append(n)
        max_discounts.append(active["discount_value"].max() if n > 0 else 0)
        has_stackable.append(int(active["stackable_flag"].max() > 0) if n > 0 else 0)
        pct_promo_cnt.append((active["promo_type"] == "percentage").sum() if n > 0 else 0)
        fixed_promo_cnt.append((active["promo_type"] == "fixed").sum() if n > 0 else 0)

    df["active_promo_count"]  = active_counts
    df["max_discount"]        = max_discounts
    df["has_stackable"]       = has_stackable
    df["pct_promo_count"]     = pct_promo_cnt
    df["fixed_promo_count"]   = fixed_promo_cnt
    df["promo_intensity"]     = df["active_promo_count"] * df["max_discount"]

    return df


def add_web_traffic_features(
    df: pd.DataFrame,
    web_traffic: pd.DataFrame,
    date_col: str = "Date",
    best_lag: int = 2,  # confirmed from cross-correlation in EDA
) -> pd.DataFrame:
    """
    Thêm web traffic features.
    LƯU Ý: KHÔNG có conversion_rate trong web_traffic.csv.
    Lag = best_lag confirmed từ cross-correlation analysis trong EDA.
    """
    df = df.copy()

    traffic_daily = (
        web_traffic.groupby("date")
        .agg(
            sessions        = ("sessions",                  "sum"),
            unique_visitors = ("unique_visitors",           "sum"),
            page_views      = ("page_views",                "sum"),
            bounce_rate     = ("bounce_rate",               "mean"),
            avg_session_dur = ("avg_session_duration_sec",  "mean"),
        )
        .reset_index()
        .rename(columns={"date": date_col})
    )

    traffic_daily[date_col] = pd.to_datetime(traffic_daily[date_col])
    df = df.merge(traffic_daily, on=date_col, how="left")

    # Lag traffic (cross-correlation lag from EDA)
    for lag in [best_lag, best_lag + 1, 7]:
        df[f"sessions_lag_{lag}"]    = df["sessions"].shift(lag)
        df[f"bounce_rate_lag_{lag}"] = df["bounce_rate"].shift(lag)

    # Rolling traffic
    df["sessions_roll_7"] = df["sessions"].shift(1).rolling(7, min_periods=3).mean()
    df["traffic_momentum"] = df["sessions"].pct_change(7).clip(-5, 5)

    return df


def add_inventory_features(
    df: pd.DataFrame,
    inventory: pd.DataFrame,
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Aggregate inventory metrics to monthly level, then merge with daily sales.
    Supply constraint proxy cho revenue.
    """
    df = df.copy()

    inv_monthly = (
        inventory.groupby("snapshot_date")
        .agg(
            avg_fill_rate    = ("fill_rate",       "mean"),
            total_stockout   = ("stockout_flag",    "sum"),
            avg_stock        = ("stock_on_hand",    "mean"),
            avg_sell_through = ("sell_through_rate","mean"),
        )
        .reset_index()
    )
    inv_monthly["snapshot_date"] = pd.to_datetime(inv_monthly["snapshot_date"])
    inv_monthly["year_month"] = inv_monthly["snapshot_date"].dt.to_period("M")

    df["year_month"] = df[date_col].dt.to_period("M")
    df = df.merge(
        inv_monthly[["year_month", "avg_fill_rate", "total_stockout", "avg_sell_through"]],
        on="year_month", how="left"
    )
    df = df.drop(columns=["year_month"])

    return df


def build_full_feature_matrix(
    sales: pd.DataFrame,
    sample_sub: pd.DataFrame,
    promotions: Optional[pd.DataFrame] = None,
    web_traffic: Optional[pd.DataFrame] = None,
    inventory: Optional[pd.DataFrame] = None,
    target_col: str = "Revenue",
    lag_list: Optional[List[int]] = None,
    roll_windows: Optional[List[int]] = None,
    traffic_best_lag: int = 2,
    use_promo: bool = True,
    use_traffic: bool = True,
    use_inventory: bool = True,
) -> tuple:
    """
    Build complete feature matrix cho cả train & test.
    Trả về: (train_feat, test_feat, FEAT_COLS)

    Args:
        use_promo:     Bật/tắt promotion features (cho ablation study).
        use_traffic:   Bật/tắt web traffic features (cho ablation study).
        use_inventory: Bật/tắt inventory features (cho ablation study).

    Cách hoạt động:
    - Concatenate train + test dates vào FULL dataframe
    - Tính lag/rolling trên FULL (để test có valid lag values từ train tail)
    - Tách lại thành train và test
    """
    if lag_list is None:
        lag_list = [1, 2, 3, 7, 14, 21, 28, 30, 60, 90, 180, 365]
    if roll_windows is None:
        roll_windows = [7, 14, 30, 60, 90]

    # ── Merge train + test into FULL ─────────────────────────────────────────
    train_base = sales[["Date", target_col, "COGS"]].copy()
    test_base  = sample_sub[["Date"]].copy().assign(**{target_col: np.nan, "COGS": np.nan})

    FULL = pd.concat([train_base, test_base], ignore_index=True)
    FULL = FULL.sort_values("Date").reset_index(drop=True)

    # ── Add features on FULL ─────────────────────────────────────────────────
    FULL = add_time_features(FULL)
    FULL = add_vn_calendar(FULL)
    FULL = add_lag_features(FULL, target_col=target_col, lags=lag_list)
    FULL = add_rolling_features(FULL, target_col=target_col, windows=roll_windows)

    # ── Conditional external features (ablation flags) ───────────────────────
    if use_promo and promotions is not None and not promotions.empty:
        FULL = add_promo_features(FULL, promotions)

    if use_traffic and web_traffic is not None and not web_traffic.empty:
        FULL = add_web_traffic_features(FULL, web_traffic, best_lag=traffic_best_lag)

    if use_inventory and inventory is not None and not inventory.empty:
        FULL = add_inventory_features(FULL, inventory)

    # ── Split back ───────────────────────────────────────────────────────────
    train_feat = FULL[FULL[target_col].notna()].copy()
    test_feat  = FULL[FULL[target_col].isna()].copy()

    # ── Feature column list ──────────────────────────────────────────────────
    EXCLUDE = {
        "Date", target_col, "COGS", "gross_margin",
        "rev_diff_7", "rev_diff_30", "rev_pct_7", "rev_pct_30", "rev_vs_ma30",
    }
    FEAT_COLS = [c for c in train_feat.columns if c not in EXCLUDE]

    return train_feat, test_feat, FEAT_COLS


# ── Utility ───────────────────────────────────────────────────────────────────

def get_feature_groups(feat_cols: List[str]) -> dict:
    """Phân loại features theo nhóm để report."""
    groups = {
        "Time":       [c for c in feat_cols if c in [
            "year","month","day","day_of_week","day_of_year","week_of_year",
            "quarter","is_weekend","is_month_end","is_month_start","is_quarter_end",
            "elapsed_days"]],
        "Cyclical":   [c for c in feat_cols if c.endswith("_sin") or c.endswith("_cos")],
        "VN Calendar":[c for c in feat_cols if any(k in c for k in [
            "tet","1111","1212","black_friday","summer","back2school",
            "womens","valentines","year_end","covid","days_to_tet"])],
        "Lag":        [c for c in feat_cols if c.startswith("lag_")],
        "Rolling":    [c for c in feat_cols if c.startswith("roll_")],
        "Traffic":    [c for c in feat_cols if any(k in c for k in
                        ["sessions","bounce","traffic","page_views","visitors","session_dur"])],
        "Promotion":  [c for c in feat_cols if any(k in c for k in
                        ["promo","discount","stackable","intensity"])],
        "Inventory":  [c for c in feat_cols if any(k in c for k in
                        ["fill_rate","stockout","stock","sell_through"])],
    }
    return groups


if __name__ == "__main__":
    print("feature_engineering.py loaded OK")
    print(f"TET_DATES: {len(TET_DATES)} dates")
