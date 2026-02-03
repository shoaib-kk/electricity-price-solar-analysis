
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.stattools import adfuller
import time
import warnings
import time_utils
import Metrics_utils


def fit_arima(series: pd.Series, order: tuple, max_iterations: int = 50):
    """fits ARIMA models with convergence handling."""

    # ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
    # warnings.warn("Maximum Likelihood optimization failed to "
    # observed that sometimes model may not converge
    try:    
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            model = ARIMA(series, order=order)
            model_fit = model.fit(method_kwargs={"maxiter": max_iterations})
            warned = any(issubclass(warn.category, ConvergenceWarning) for warn in w)
            converged = bool(model_fit.mle_retvals.get("converged", True))

            if warned or not converged:
                print(f"ARIMA{order} did not converge properly.")
                return None
            return model_fit
    except Exception as e:
        print(f"ARIMA{order} fitting failed: {e}")
        return None
    

def grid_search_arima(series, p_values, q_values) -> tuple[tuple, ARIMA]:
    """Grid search ARMA(p,0,q) on a pre-differenced series using AIC."""
    best_aic = float("inf")
    best_order = None
    best_model = None

    # figure out which combination of p, d, q yields the lowest AIC
    for p in p_values:
        for q in q_values:
            model_fit = fit_arima(series, order=(p, 0, q))
            if model_fit is None:
                continue
            aic = model_fit.aic
            print(f"Tested ARIMA({p},0,{q}) - AIC:{aic:.2f}")
            if aic < best_aic:
                best_aic = aic
                best_order = (p, 0, q)
                best_model = model_fit

    return best_order, best_model


def _transform_rrp_to_log_and_diff(rrp: pd.Series) -> tuple[pd.Series, dict]:
    """Log-transform RRP with a positive shift and difference once.
    Returns the differenced log series plus transform_info needed to
    invert forecasts back to the original price scale.
    """

    shift = 1 - min(0.0, rrp.min())
    rrp_log = np.log(rrp + shift)
    rrp_log_diff = rrp_log.diff().dropna()
    transform_info = {"shift": shift, "last_log_level": rrp_log.iloc[-1]}
    return rrp_log_diff, transform_info


def _invert_diff_log_forecast(diff_forecast: pd.Series, transform_info: dict) -> pd.Series:
    """Invert a forecast made on the differenced log scale."""

    last_log_level = transform_info["last_log_level"]
    shift = transform_info["shift"]
    log_level_forecast = last_log_level + diff_forecast.cumsum()
    return np.exp(log_level_forecast) - shift


def baseline_TSA(df, p_value: int | None = None, q_value: int | None = None):
    """Performs Time Series Analysis using ARIMA model to act as a baseline,
    ignores most features besides RRP series."""

    # Ensure we have a DatetimeIndex; if SETTLEMENTDATE is present
    # as a column, use it as the index (mirrors run_baseline_models).
    if "SETTLEMENTDATE" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("SETTLEMENTDATE")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex; clean data first.")

    rrp = df["RRP"].resample("30min").mean().ffill().dropna()

    # Log-transform with a positive shift to handle negative prices,
    # then difference once to safely fix d=0 in ARIMA.
    # Check stationarity below to make this explicit rather than
    # guessing.
    rrp_log_diff, transform_info = _transform_rrp_to_log_and_diff(rrp)

    # Check stationarity with ADF test
    try:
        p_log = adfuller(np.log(rrp + transform_info["shift"]).dropna(), autolag="AIC")[1]
    except Exception:
        p_log = np.nan
    try:
        p_diff = adfuller(rrp_log_diff.dropna(), autolag="AIC")[1]
    except Exception:
        p_diff = np.nan
    print(f"ADF p-values: log RRP={p_log:.4f}, diff(log RRP)={p_diff:.4f}")

    best_model = None
    best_order = None

    if p_value is not None and q_value is not None:
        print(f"Fitting ARIMA({p_value},0,{q_value}) model...")
        start_time = time.time()
        model = fit_arima(rrp_log_diff, order=(p_value, 0, q_value))
        if model is None:
            raise RuntimeError(f"ARIMA({p_value},0,{q_value}) model fitting failed.")

        best_model = model
        best_order = (p_value, 0, q_value)
        elapsed_time = time.time() - start_time
        print(f"Model fitted in {elapsed_time:.2f} seconds\n")

    else:
        # Restrict p and q to 0,1,2 to keep the baseline model simple
        # and avoid overfitting and long runtimes for little gain.
        # takes almost 2 minutes to run on my machine even with these small ranges.
        start_time = time.time()
        best_order, best_model = grid_search_arima(
            rrp_log_diff,
            p_values=range(0, 3),
            q_values=range(0, 3),
        )
        elapsed_time = time.time() - start_time
        if best_order is None or best_model is None:
            raise RuntimeError("Grid search did not produce a valid ARIMA model.")
        print(f"Fitting ARIMA({best_order[0]},0,{best_order[2]}) model...")
        print(f"Model fitted in {elapsed_time:.2f} seconds\n")

    print(best_model.summary())

    print("\nGenerating forecast...")

    # Forecast on the differenced log scale, then invert the
    # transformation back to the original price scale.
    diff_forecast = best_model.forecast(steps=48)
    price_forecast = _invert_diff_log_forecast(diff_forecast, transform_info)

    print("Forecasted RRP for next 24 hours (48 half-hour intervals):")
    print(price_forecast)

def evaluate_point_forecast(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> None:
    """Print simple MAE/RMSE metrics for a forecast.

    sjuo sMAPE because percentage errors become unstable and misleading when prices approach zero or
    go negative which happens in this market .
    """

    metrics = Metrics_utils.compute_mae_rmse_smape(y_true, y_pred)
    mae = metrics["mae"]
    rmse = metrics["rmse"]

    # Cost-weighted MAE: weight errors by the absolute true price so that
    # mistakes made at very high prices count more than mistakes at low prices.
    cost_weighted_mae = Metrics_utils.compute_cost_weighted_mae(y_true, y_pred)

    print(f"\nBaseline evaluation ({label}):")
    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    if np.isfinite(cost_weighted_mae):
        print(f"  Cost-weighted MAE (|price|-weighted): {cost_weighted_mae:.2f}")
    print(
        "  Note: sMAPE is not reported because it becomes unstable "
        "and misleading when prices are near zero or negative."
    )

def run_baseline_models(df: pd.DataFrame, horizon_minutes: int = 30) -> None:
    """Compute simple persistence and seasonal-naive baselines for RRP.
    forecast RRP at t + h using only information available
    at time t, then compare against realised RRP_{t+h} across the
    historical sample.
    """

    if "SETTLEMENTDATE" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("SETTLEMENTDATE")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex; clean data first.")

    rrp = df["RRP"].sort_index()

    # Work on a regular 30-minute series to keep things comparable
    rrp = rrp.resample("30min").mean().ffill().dropna()

    # Infer time step and derive horizon and daily period in steps
    step_minutes = time_utils.infer_step_minutes(rrp.index, fallback_minutes=30.0)
    ratio = horizon_minutes / step_minutes
    horizon_steps = max(1, int(round(ratio)))
    if abs(ratio - horizon_steps) > 1e-3:
        raise ValueError(
            f"Requested horizon {horizon_minutes} minutes is not an integer "
            f"multiple of inferred step {step_minutes:.3f} minutes."
        )
    steps_per_day = int(round(1440.0 / step_minutes))

    # Build the h-step-ahead target: RRP_{t+h}
    rrp_target = rrp.shift(-horizon_steps)
    mask = rrp_target.notna()

    # Keep these as aligned Series; convert to numpy only when
    # passing into metric functions 
    y_true_series = rrp_target[mask]
    current_rrp_series = rrp[mask]

    print(
        f"Baseline design: horizon={horizon_minutes} minutes, "
        f"step={step_minutes:.2f} minutes, horizon_steps={horizon_steps}, "
        f"steps_per_day={steps_per_day}"
    )

    # Baseline 1: persistence y_hat_{t+h} = RRP_t
    persistence_pred = current_rrp_series.copy()
    evaluate_point_forecast(
        y_true_series.values,
        persistence_pred.values,
        label="Persistence",
    )
    Metrics_utils.evaluate_direction_accuracy(
        y_true_series.values,
        persistence_pred.values,
        current_rrp_series.values,
        label="Persistence",
    )

    # Baseline 2: seasonal naive y_hat_{t+h} = RRP_{t+h-steps_per_day}
    rrp_target_seasonal = rrp_target.shift(steps_per_day)
    seasonal_mask = mask & rrp_target_seasonal.notna()
    if seasonal_mask.any():
        # Use the original resampled series for alignment rather than
        # indexing into a masked numpy array.
        y_true_seasonal = rrp_target[seasonal_mask]
        y_pred_seasonal = rrp_target_seasonal[seasonal_mask]
        current_rrp_seasonal = rrp[seasonal_mask]
        evaluate_point_forecast(
            y_true_seasonal.values,
            y_pred_seasonal.values,
            label="Seasonal naive (1-day)",
        )
        Metrics_utils.evaluate_direction_accuracy(
            y_true_seasonal.values,
            y_pred_seasonal.values,
            current_rrp_seasonal.values,
            label="Seasonal naive (1-day)",
        )

def main():
    file_path = "CLEANED_PRICE_AND_DEMAND_VIC1_TRAIN.csv"
    df = pd.read_csv(file_path, parse_dates=["SETTLEMENTDATE"])

    print("Initial data info:")
    print(df.info())

    print("\n" + "=" * 70)
    print("Running ARIMA analysis for baseline comparison")
    print("=" * 70)
    baseline_TSA(df, p_value = 1, q_value = 2)
    print("=" * 70 + "\n")

    print("\n" + "=" * 70)
    print("Running naive baseline models (persistence, seasonal)")
    print("=" * 70)
    run_baseline_models(df, horizon_minutes=30)
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
