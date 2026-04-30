"""
src/models.py
=============
Model classes và training logic cho Sales Forecasting.
Pipeline: LightGBM + XGBoost + Prophet → Weighted Ensemble

Author : Team Gridbreakers
Seed   : 42
"""

import pandas as pd
import numpy as np
import warnings
from typing import List, Dict, Optional, Tuple
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
try:
    from catboost import CatBoostRegressor
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    CatBoostRegressor = None

RANDOM_SEED = 42


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Tính MAE, RMSE, R² — 3 metrics chính của cuộc thi."""
    y_pred = np.maximum(y_pred, 0)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE — bổ sung cho diagnostic."""
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# ── LightGBM ──────────────────────────────────────────────────────────────────

LGBM_PARAMS = {
    "objective":        "regression",
    "metric":           "mae",
    "n_estimators":     5000,
    "learning_rate":    0.02,
    "num_leaves":       127,
    "max_depth":        -1,
    "min_child_samples": 15,
    "feature_fraction": 0.80,
    "bagging_fraction": 0.85,
    "bagging_freq":     5,
    "lambda_l1":        0.1,
    "lambda_l2":        1.0,
    "verbose":          -1,
    "random_state":     RANDOM_SEED,
    "n_jobs":           -1,
}

XGB_PARAMS = {
    "objective":        "reg:squarederror",
    "n_estimators":     4000,
    "learning_rate":    0.02,
    "max_depth":        7,
    "subsample":        0.85,
    "colsample_bytree": 0.80,
    "min_child_weight": 5,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     RANDOM_SEED,
    "n_jobs":           -1,
    "tree_method":      "hist",
}

CATBOOST_PARAMS = {
    "loss_function": "MAE",
    "iterations": 2000,
    "learning_rate": 0.02,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "subsample": 0.85,
    "random_seed": RANDOM_SEED,
    "verbose": False,
}


class TimeSeriesCV:
    """
    TimeSeriesSplit cross-validation với walk-forward strategy.
    KHÔNG dùng random split — respect temporal order.
    """

    def __init__(self, n_splits: int = 5, test_size: int = 365):
        self.n_splits  = n_splits
        self.test_size = test_size
        self.tscv      = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

    def cross_validate_lgbm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: np.ndarray,
        params: dict = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, lgb.LGBMRegressor]:
        """CV với LightGBM. Trả về: (cv_results_df, oof_preds, best_model)."""
        if params is None:
            params = LGBM_PARAMS

        results = []
        oof     = np.zeros(len(y))
        best_model = None
        best_score = np.inf

        for fold, (tr_idx, va_idx) in enumerate(self.tscv.split(X)):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[
                    lgb.early_stopping(200, verbose=False),
                    lgb.log_evaluation(500),
                ],
            )
            print(f"  Fold {fold + 1} LGBM best_iteration={model.best_iteration_}")

            pred     = np.maximum(model.predict(X_va), 0)
            oof[va_idx] = pred
            m        = compute_metrics(y_va, pred)

            val_start = pd.Timestamp(dates[va_idx[0]]).date()
            val_end   = pd.Timestamp(dates[va_idx[-1]]).date()

            results.append({
                "fold":       fold + 1,
                "train_size": len(tr_idx),
                "val_start":  val_start,
                "val_end":    val_end,
                "best_iter":  model.best_iteration_,
                **m,
            })

            if m["MAE"] < best_score:
                best_score = m["MAE"]
                best_model = model

        return pd.DataFrame(results), oof, best_model

    def cross_validate_xgb(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: np.ndarray,
        params: dict = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, xgb.XGBRegressor]:
        """CV với XGBoost."""
        if params is None:
            params = XGB_PARAMS

        results = []
        oof     = np.zeros(len(y))
        best_model = None
        best_score = np.inf

        for fold, (tr_idx, va_idx) in enumerate(self.tscv.split(X)):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            model = xgb.XGBRegressor(**params, early_stopping_rounds=200)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=500,
            )
            print(f"  Fold {fold + 1} XGB best_iteration={model.best_iteration}")

            pred     = np.maximum(model.predict(X_va), 0)
            oof[va_idx] = pred
            m        = compute_metrics(y_va, pred)

            val_start = pd.Timestamp(dates[va_idx[0]]).date()
            val_end   = pd.Timestamp(dates[va_idx[-1]]).date()

            results.append({
                "fold":       fold + 1,
                "train_size": len(tr_idx),
                "val_start":  val_start,
                "val_end":    val_end,
                "best_iter":  model.best_iteration,
                **m,
            })

            if m["MAE"] < best_score:
                best_score = m["MAE"]
                best_model = model

        return pd.DataFrame(results), oof, best_model

    def cross_validate_catboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: np.ndarray,
        params: dict = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, Optional["CatBoostRegressor"]]:
        """CV với CatBoost nếu package khả dụng."""
        if CatBoostRegressor is None:
            raise ImportError("catboost is not installed")
        if params is None:
            params = CATBOOST_PARAMS

        results = []
        oof = np.zeros(len(y))
        best_model = None
        best_score = np.inf

        for fold, (tr_idx, va_idx) in enumerate(self.tscv.split(X)):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            model = CatBoostRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=(X_va, y_va),
                use_best_model=True,
                verbose=False,
            )

            pred = np.maximum(model.predict(X_va), 0)
            oof[va_idx] = pred
            m = compute_metrics(y_va, pred)

            val_start = pd.Timestamp(dates[va_idx[0]]).date()
            val_end = pd.Timestamp(dates[va_idx[-1]]).date()

            results.append({
                "fold": fold + 1,
                "train_size": len(tr_idx),
                "val_start": val_start,
                "val_end": val_end,
                "best_iter": getattr(model, "get_best_iteration", lambda: None)(),
                **m,
            })

            if m["MAE"] < best_score:
                best_score = m["MAE"]
                best_model = model

        return pd.DataFrame(results), oof, best_model


class FinalEnsemble:
    """
    Weighted ensemble: LightGBM + XGBoost (+ optionally Prophet).
    Weights tối ưu dựa trên validation performance.
    """

    def __init__(
        self,
        w_lgbm: float = 0.50,
        w_xgb:  float = 0.50,
        w_catboost: float = 0.0,
        w_prophet: float = 0.0,
    ):
        self.w_lgbm    = w_lgbm
        self.w_xgb     = w_xgb
        self.w_catboost = w_catboost
        self.w_prophet = w_prophet
        total = w_lgbm + w_xgb + w_catboost + w_prophet
        assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1, got {total:.4f}"

    def predict(
        self,
        lgbm_pred:    np.ndarray,
        xgb_pred:     np.ndarray,
        catboost_pred: Optional[np.ndarray] = None,
        prophet_pred: Optional[np.ndarray] = None,
        n: Optional[int] = None,
    ) -> np.ndarray:
        if n is None:
            n = min(len(lgbm_pred), len(xgb_pred))
        ensemble = self.w_lgbm * lgbm_pred[:n] + self.w_xgb * xgb_pred[:n]
        if catboost_pred is not None and self.w_catboost > 0:
            n2 = min(n, len(catboost_pred))
            ensemble[:n2] += self.w_catboost * catboost_pred[:n2]
        if prophet_pred is not None and self.w_prophet > 0:
            n2 = min(n, len(prophet_pred))
            ensemble[:n2] += self.w_prophet * prophet_pred[:n2]
        return np.maximum(ensemble, 0)

    def summary(self) -> str:
        parts = [
            f"LightGBM({self.w_lgbm:.0%})",
            f"XGBoost({self.w_xgb:.0%})",
        ]
        if self.w_catboost > 0:
            parts.append(f"CatBoost({self.w_catboost:.0%})")
        if self.w_prophet > 0:
            parts.append(f"Prophet({self.w_prophet:.0%})")
        return "Ensemble: " + " + ".join(parts)


def optimize_ensemble_weights(
    y_true: np.ndarray,
    lgbm_pred: np.ndarray,
    xgb_pred:  np.ndarray,
    catboost_pred: Optional[np.ndarray] = None,
    prophet_pred: Optional[np.ndarray] = None,
    metric: str = "MAE",
    min_weight: float = 0.05,
    regularization: float = 0.02,
) -> Dict[str, float]:
    """
    Tối ưu trọng số ensemble dựa trên validation predictions.
    Sử dụng scipy.optimize để tìm trọng số tối ưu liên tục.

    Args:
        y_true:       Giá trị thực trên validation set.
        lgbm_pred:    LightGBM predictions.
        xgb_pred:     XGBoost predictions.
        prophet_pred: Prophet predictions (None nếu không dùng).
        metric:       Metric tối ưu: 'MAE' hoặc 'RMSE'.

    Returns:
        dict với keys 'w_lgbm', 'w_xgb', 'w_prophet', và metric score.
    """
    from scipy.optimize import minimize

    use_catboost = catboost_pred is not None
    use_prophet = prophet_pred is not None

    def loss(weights):
        if use_catboost and use_prophet:
            w_l, w_x, w_c, w_p = weights
            blend = w_l * lgbm_pred + w_x * xgb_pred + w_c * catboost_pred + w_p * prophet_pred
        elif use_catboost:
            w_l, w_x, w_c = weights
            blend = w_l * lgbm_pred + w_x * xgb_pred + w_c * catboost_pred
        else:
            if use_prophet:
                w_l, w_x, w_p = weights
                blend = w_l * lgbm_pred + w_x * xgb_pred + w_p * prophet_pred
            else:
                w_l, w_x = weights
                blend = w_l * lgbm_pred + w_x * xgb_pred
        blend = np.maximum(blend, 0)
        if metric == "MAE":
            base_loss = mean_absolute_error(y_true, blend)
        else:
            base_loss = float(np.sqrt(mean_squared_error(y_true, blend)))
        prior = np.ones(len(weights), dtype=float) / len(weights)
        penalty = regularization * float(base_loss) * float(np.sum((np.asarray(weights) - prior) ** 2))
        return base_loss + penalty

    # Constraints: weights sum to 1, all >= 0
    if use_catboost and use_prophet:
        n_w = 4
        x0 = [0.25, 0.25, 0.25, 0.25]
    elif use_catboost or use_prophet:
        n_w = 3
        x0 = [1/3, 1/3, 1/3]
    else:
        n_w = 2
        x0 = [0.5, 0.5]

    constraints = {"type": "eq", "fun": lambda w: sum(w) - 1.0}
    min_weight = float(np.clip(min_weight, 0.0, 1.0 / max(n_w, 1)))
    bounds = [(min_weight, 1.0)] * n_w

    result = minimize(loss, x0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"ftol": 1e-9, "maxiter": 500})

    best_w = result.x
    if use_catboost and use_prophet:
        w_lgbm, w_xgb, w_catboost, w_prophet = best_w
    elif use_catboost:
        w_lgbm, w_xgb, w_catboost = best_w
        w_prophet = 0.0
    elif use_prophet:
        w_lgbm, w_xgb, w_prophet = best_w
        w_catboost = 0.0
    else:
        w_lgbm, w_xgb = best_w
        w_catboost = 0.0
        w_prophet = 0.0

    # Final score with optimal weights
    blend = w_lgbm * lgbm_pred + w_xgb * xgb_pred
    if use_catboost:
        blend += w_catboost * catboost_pred
    if use_prophet:
        blend += w_prophet * prophet_pred
    blend = np.maximum(blend, 0)
    m = compute_metrics(y_true, blend)

    return {
        "w_lgbm":    round(float(w_lgbm), 4),
        "w_xgb":     round(float(w_xgb), 4),
        "w_catboost": round(float(w_catboost), 4),
        "w_prophet": round(float(w_prophet), 4),
        **m,
    }


def grid_search_ensemble_weights(
    y_true: np.ndarray,
    lgbm_pred: np.ndarray,
    xgb_pred:  np.ndarray,
) -> pd.DataFrame:
    """
    Grid search trọng số LGBM/XGB với bước 0.1.
    Trả về DataFrame sorted by MAE (tốt nhất ở trên cùng).
    """
    rows = []
    for wl in np.arange(0.0, 1.1, 0.1):
        wx = round(1.0 - wl, 4)
        blend = wl * lgbm_pred + wx * xgb_pred
        m = compute_metrics(y_true, blend)
        rows.append({"w_lgbm": round(wl, 1), "w_xgb": round(wx, 1), **m})
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def train_final_lgbm(
    X: np.ndarray,
    y: np.ndarray,
    best_iter: int,
    params: dict = None,
) -> lgb.LGBMRegressor:
    """Train LightGBM trên toàn bộ training data với best_iter từ CV."""
    if params is None:
        params = LGBM_PARAMS.copy()

    final_params = {**params, "n_estimators": best_iter}
    model = lgb.LGBMRegressor(**final_params)
    model.fit(X, y)
    return model


def train_final_xgb(
    X: np.ndarray,
    y: np.ndarray,
    best_iter: int,
    params: dict = None,
) -> xgb.XGBRegressor:
    """Train XGBoost trên toàn bộ training data với best_iter từ CV."""
    if params is None:
        params = XGB_PARAMS.copy()

    final_params = {**params, "n_estimators": best_iter}
    model = xgb.XGBRegressor(**final_params)
    model.fit(X, y)
    return model


def train_final_catboost(
    X: np.ndarray,
    y: np.ndarray,
    best_iter: Optional[int] = None,
    params: dict = None,
    sample_weight: Optional[np.ndarray] = None,
):
    """Train CatBoost trên toàn bộ training data nếu package khả dụng."""
    if CatBoostRegressor is None:
        raise ImportError("catboost is not installed")
    if params is None:
        params = CATBOOST_PARAMS.copy()
    else:
        params = params.copy()

    if best_iter is not None and best_iter > 0:
        params["iterations"] = int(best_iter)

    model = CatBoostRegressor(**params)
    model.fit(X, y, sample_weight=sample_weight, verbose=False)
    return model


def predict_cogs(
    revenue_pred: np.ndarray,
    dates: pd.Series,
    historical_train: pd.DataFrame,
) -> np.ndarray:
    """
    Dự báo COGS dựa trên median COGS/Revenue ratio theo tháng từ dữ liệu lịch sử.
    Approach này ổn định hơn prediction riêng biệt vì COGS highly correlated với Revenue.
    """
    # Tính monthly COGS ratio từ training data
    hist = historical_train.copy()
    hist["month"] = hist["Date"].dt.month
    hist["cogs_ratio"] = hist["COGS"] / hist["Revenue"].replace(0, np.nan)

    monthly_ratio = hist.groupby("month")["cogs_ratio"].median().to_dict()
    global_ratio  = hist["cogs_ratio"].median()

    test_months = pd.to_datetime(dates).dt.month.values
    ratios = np.array([monthly_ratio.get(m, global_ratio) for m in test_months])
    cogs_pred = np.maximum(revenue_pred * ratios, 0)

    return cogs_pred


# ── Print helpers ─────────────────────────────────────────────────────────────

def print_cv_table(cv_df: pd.DataFrame, model_name: str = "") -> None:
    """In bảng CV results đẹp."""
    title = f"{'─'*80}"
    print(title)
    print(f"  {model_name} — {len(cv_df)}-Fold TimeSeriesSplit CV Results")
    print(title)
    print(f"  {'Fold':<5} {'Train':>8} {'Val Period':<35} {'MAE':>10} {'RMSE':>10} {'R²':>8}")
    print(f"  {'─'*5} {'─'*8} {'─'*35} {'─'*10} {'─'*10} {'─'*8}")
    for _, row in cv_df.iterrows():
        period = f"{row['val_start']} → {row['val_end']}"
        print(f"  {int(row['fold']):<5} {int(row['train_size']):>8,} "
              f"{period:<35} {row['MAE']:>10,.0f} {row['RMSE']:>10,.0f} {row['R2']:>8.3f}")
    print(f"  {'─'*5} {'─'*8} {'─'*35} {'─'*10} {'─'*10} {'─'*8}")
    print(f"  {'MEAN':<5} {'':<8} {'':<35} "
          f"{cv_df['MAE'].mean():>10,.0f} {cv_df['RMSE'].mean():>10,.0f} "
          f"{cv_df['R2'].mean():>8.3f}")
    print(f"  {'STD':<5} {'':<8} {'':<35} "
          f"{cv_df['MAE'].std():>10,.0f} {cv_df['RMSE'].std():>10,.0f} "
          f"{cv_df['R2'].std():>8.3f}")
    print(title)


if __name__ == "__main__":
    print("models.py loaded OK")
    print(f"LightGBM version: {lgb.__version__}")
    print(f"XGBoost version:  {xgb.__version__}")
