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