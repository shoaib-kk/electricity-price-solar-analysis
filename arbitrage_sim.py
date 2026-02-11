from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from Battery_model import Battery
from time_utils import infer_step_minutes
from Quantile_regression import (
    ConformalQuantileBundle,
    compute_split_conformal_interval,
    load_cleaned_datasets,
    manage_train_test_split,
    predict_conformal,
    predict_quantiles,
    train_quantile_model,
)
from logging_utils import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)

@dataclass
class SimulationMetrics:
    total_profit: float
    terminal_value: float
    total_profit_with_terminal: float
    max_drawdown: float
    grid_throughput_kwh: float
    equivalent_cycles: float
    

def _validate_prices(prices: pd.Series) -> pd.Series:
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series indexed by time.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must have a DatetimeIndex.")
    if not np.isfinite(prices.values).all():
        raise ValueError("prices contains NaN or infinite values.")
    return prices.sort_index()


def max_drawdown(cumulative_profit: pd.Series) -> float:
    """Basically computes the max distance between your lowest point
     and the highest point you've been at before that. """
    if not cumulative_profit.index.is_unique:
        raise ValueError("cumulative_profit index must be unique for drawdown computation.")
    running_max = cumulative_profit.cummax()
    drawdown = running_max - cumulative_profit

    return float(drawdown.max())


def always_charge_discharge_policy(timestamp, price_kwh, battery) -> tuple[str, float]:
    if battery.soc < 0.5:
        logger.debug(f"Timestamp {timestamp}: Charging because SoC {battery.soc} < 0.5")
        return "charge", battery.max_power_kw
    else:
        logger.debug(f"Timestamp {timestamp}: Discharging because SoC {battery.soc} >= 0.5")
        return "discharge", battery.max_power_kw


def baseline_threshold_policy(training_prices_kwh: pd.Series, low_quantile=0.3, high_quantile=0.7):
    low_threshold = float(training_prices_kwh.quantile(low_quantile))
    high_threshold = float(training_prices_kwh.quantile(high_quantile))

    def policy(timestamp, price_kwh, battery) -> tuple[str, float]:
        if price_kwh < low_threshold:
            logger.debug(f"Timestamp {timestamp}: Charging because price {price_kwh} < low_threshold {low_threshold}")
            return "charge", battery.max_power_kw
        elif price_kwh > high_threshold:
            logger.debug(f"Timestamp {timestamp}: Discharging because price {price_kwh} > high_threshold {high_threshold}")
            return "discharge", battery.max_power_kw
        else:
            logger.debug(f"Timestamp {timestamp}: Holding because price {price_kwh} is between thresholds {low_threshold} and {high_threshold}")
            return "hold", 0.0

    return policy, {"low_quantile": low_quantile, "high_quantile": high_quantile}
def quantile_regression_policy(model: ConformalQuantileBundle, alpha_low=0.1, alpha_high=0.9):
    def policy(timestamp, price_kwh, battery) -> tuple[str, float]:
        q_low = model.predict_quantile(timestamp, alpha_low)
        q_high = model.predict_quantile(timestamp, alpha_high)

        if price_kwh < q_low:
            logger.debug(f"Timestamp {timestamp}: Charging because price {price_kwh} < q_low {q_low}")
            return "charge", battery.max_power_kw
        elif price_kwh > q_high:
            logger.debug(f"Timestamp {timestamp}: Discharging because price {price_kwh} > q_high {q_high}")
            return "discharge", battery.max_power_kw
        else:
            logger.debug(f"Timestamp {timestamp}: Holding because price {price_kwh} is between quantiles {q_low} and {q_high}")
            return "hold", 0.0

    return policy, {"alpha_low": alpha_low, "alpha_high": alpha_high}

def run_arbitrage_simulation(prices: pd.Series, battery: Battery, dt_hours: float, policy: Callable, fee_rate: float = 0.01,
                             degradation_rate_per_kwh: float = 0.0001) -> SimulationMetrics:    
    prices = _validate_prices(prices)
    records = []
    cumulative_profit = 0.0

    for timestamp, price_kwh in prices.items():
        price_kwh = float(price_kwh)


        action, power_kw = policy(timestamp, price_kwh, battery)
        energy_bought, energy_sold, new_soc = battery.step(action, power_kw, dt_hours)

        buy_fee = fee_rate * energy_bought * price_kwh
        sell_fee = fee_rate * energy_sold * price_kwh
        transaction_cost = buy_fee + sell_fee
        degradation_cost = degradation_rate_per_kwh * (energy_bought + energy_sold)
        profit = energy_sold * price_kwh - energy_bought * price_kwh - transaction_cost - degradation_cost
        
        cumulative_profit += profit

        records.append({
            "timestamp": timestamp,
            "price_kwh": price_kwh,
            "action": action,
            "power_kw": power_kw,
            "energy_bought_kwh": energy_bought,
            "energy_sold_kwh": energy_sold,
            "new_soc": new_soc,
            "profit": profit,
            "cumulative_profit": cumulative_profit
        })

    actions_df = pd.DataFrame.from_records(records).set_index("timestamp")

    last_price = actions_df["price_kwh"].iloc[-1]
    
    # how much value I have left in my battery if I were to sell it at the end 
    terminal_value = battery.soc_kwh * last_price
    total_profit_with_terminal = cumulative_profit + terminal_value

    metrics = SimulationMetrics(
        total_profit=cumulative_profit,
        terminal_value=terminal_value,
        total_profit_with_terminal=total_profit_with_terminal,
        max_drawdown=max_drawdown(actions_df["cumulative_profit"]),
        grid_throughput_kwh=actions_df["energy_bought_kwh"].sum() + actions_df["energy_sold_kwh"].sum(),
        equivalent_cycles=(actions_df["energy_bought_kwh"].sum() + actions_df["energy_sold_kwh"].sum()) / (2 * battery.capacity_kwh))
    
    return actions_df, metrics 




def mwh_to_kwh(price_mwh: float) -> float:
    return price_mwh / 1000.0

def series_mwh_to_kwh(prices_mwh: pd.Series) -> pd.Series:
    return prices_mwh.apply(mwh_to_kwh)


def filter_prices_by_period(
    prices: pd.Series,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.Series:
    """
    Slice your price series to a time window.
    Raise if empty.
    """
    if prices is None or prices.empty:
        raise ValueError("Input price series is empty or None.")
    
    filtered = prices
    if start is not None:
        start = pd.Timestamp(start)
        filtered = filtered[filtered.index >= start]
    if end is not None:
        end = pd.Timestamp(end)
        filtered = filtered[filtered.index <= end]
    if filtered.empty:
        raise ValueError("Filtered price series is empty.")

    return filtered


def compute_valid_horizons(step_minutes: float, candidate_horizons: list[int]) -> list[int]:
    valid_horizons = []
    for horizon in candidate_horizons:
        ratio = horizon / step_minutes 

        # Allow a small numerical tolerance when checking if ratio is an integer -> if the horizon is a multiple of the step size
        if abs(ratio - round(ratio)) < 0.001:
            valid_horizons.append(horizon)
        else:
            logger.warning(f"Candidate horizon {horizon} minutes is not an integer multiple of step {step_minutes:.3f} minutes and will be skipped.")
    if not valid_horizons:
        raise ValueError("No valid horizons found.")
    return valid_horizons


def compute_decision_times_from_conformal(
    conformal_dfs: dict[int, pd.DataFrame],
) -> pd.DatetimeIndex:
    """
    conformal dfs are indexed by forecast_time = decision_time + horizon
    For each horizon, shift the index back by that horizon to get decision times, then take the union across horizons to get all unique decision times you need to have price forecasts for.
    """
    decision_times = pd.DatetimeIndex([])
    for horizon, df in conformal_dfs.items():
        horizon_td = pd.Timedelta(minutes=horizon)
        decision_times = decision_times.union(df.index - horizon_td)
    return decision_times


def align_prices_to_decision_times(
    prices: pd.Series,
    decision_times: pd.DatetimeIndex,
) -> pd.Series:

    aligned_prices = prices[prices.index.isin(decision_times)]
    if aligned_prices.empty:
        raise ValueError("No Aligned prices found")
    return aligned_prices


def train_quantile_models_for_horizon(
    horizon_minutes: int,
    quantiles: list[float] = [0.05, 0.5, 0.95],
    early_stopping_rounds: int = 50,
):
    """
      1) manage_train_test_split(forecast_horizon_minutes=horizon_minutes)
      2) train_quantile_model(...)
      3) model + splits
    """

    X_train_df, y_train, X_val_df, y_val, X_test_df, y_true_test, current_rrp_test = manage_train_test_split(forecast_horizon_minutes = horizon_minutes)
    models = {}
    for q in quantiles:
        model = train_quantile_model(X_train_df, y_train, X_val_df, y_val, quantile=q, early_stopping_rounds=early_stopping_rounds)
        models[q] = model
    return models, X_train_df, y_train, X_val_df, y_val, X_test_df, y_true_test, current_rrp_test

def build_conformal_df_for_horizon(
    *,
    horizon_minutes: int,
    models,
    X_val_df: pd.DataFrame,
    y_val: pd.Series,
    X_test_df: pd.DataFrame,
    y_true_test: pd.Series,
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
    target_coverage: float = 0.90,
) -> tuple[pd.DataFrame, float]:
    """
    Build a conformal interval DataFrame for one horizon.
    """

    # create a series of the predicted low and high quantiles on the validation set 
    q_low_val = pd.Series(models[quantile_low].predict(X_val_df), index=y_val.index)
    q_high_val = pd.Series(models[quantile_high].predict(X_val_df), index=y_val.index)

    # calculate nonconformity scores on the validation set using the true values and predicted quantiles
    nonconformity = np.maximum(q_low_val - y_val, y_val - q_high_val)
    nonconformity = np.maximum(nonconformity, 0.0)
    nonconformity = nonconformity.to_numpy()
    n_cal = nonconformity.shape[0]

    # the smallest number such that at least 90% of the validation points have nonconformity scores <= q_statistic
    q_statistic = np.quantile(nonconformity, target_coverage * (1 + 1/n_cal))


    # create a series of the predicted low and high quantiles on the test set 
    q_low_test = pd.Series(models[quantile_low].predict(X_test_df), index=y_true_test.index)
    q_high_test = pd.Series(models[quantile_high].predict(X_test_df), index=y_true_test.index)


    conformal_lower = q_low_test - q_statistic
    conformal_upper = q_high_test + q_statistic

    conformal_df = pd.DataFrame({
        "lower_conformal": conformal_lower,

        "median": (q_low_test + q_high_test) / 2,
        "upper_conformal": conformal_upper
    })

    return conformal_df, q_statistic

