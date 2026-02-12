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
    profit_per_cycle: float = 0.0
    profit_drawdown_ratio: float = 0.0
def apply_fees_and_cooldown(policy, fee_rate = 0.005, cooldown_minutes = 0):
    """
    Wrap a policy to apply fees and cooldowns.
 assume cooldown just means skip actions for a certain number of minutes after a charge or discharge action.
    use a wrapped policy to avoid throwing in the extra parameters into other policies 
    """

    action_properties = {"last_action_time": None, "last_action_type": None}

    def wrapped_policy(timestamp, price_kwh, battery) -> tuple[str, float]:

        if action_properties["last_action_time"] is not None and (timestamp - action_properties["last_action_time"]).total_seconds() < cooldown_minutes * 60:
            logger.debug(f"Timestamp {timestamp}: In cooldown period after {action_properties['last_action_type']} at {action_properties['last_action_time']}. Holding.")
            return "hold", 0.0

        action, power_kw = policy(timestamp, price_kwh, battery)

        if action in ["charge", "discharge"]:
            action_properties["last_action_time"] = timestamp
            action_properties["last_action_type"] = action

        return action, power_kw

    return wrapped_policy

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

    params = {"threshold_soc": 0.5}
    def policy(timestamp, price_kwh, battery) -> tuple[str, float]:
        if battery.soc < params["threshold_soc"]:
            logger.debug(f"Timestamp {timestamp}: Charging because SoC {battery.soc} < {params['threshold_soc']}")
            return "charge", battery.max_power_kw
        else:
            logger.debug(f"Timestamp {timestamp}: Discharging because SoC {battery.soc} >= {params['threshold_soc']}")
            return "discharge", battery.max_power_kw
    return policy, params


def baseline_threshold_policy(training_prices_kwh: pd.Series, low_quantile=0.3, high_quantile=0.7):
    low_threshold = float(training_prices_kwh.quantile(low_quantile))
    high_threshold = float(training_prices_kwh.quantile(high_quantile))
    params = {"low_quantile": low_quantile, "high_quantile": high_quantile}
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

    return policy, params
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
        equivalent_cycles=(actions_df["energy_bought_kwh"].sum() + actions_df["energy_sold_kwh"].sum()) / (2 * battery.capacity_kwh),
        profit_drawdown_ratio=cumulative_profit / max_drawdown(actions_df["cumulative_profit"]) if max_drawdown(actions_df["cumulative_profit"]) > 0 else float("inf"),
        profit_per_cycle=cumulative_profit / ((actions_df["energy_bought_kwh"].sum() + actions_df["energy_sold_kwh"].sum()) / (2 * battery.capacity_kwh)) if (actions_df["energy_bought_kwh"].sum() + actions_df["energy_sold_kwh"].sum()) > 0 else float("inf")
        
        
        )
    
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
    Slice price series to a time window.
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

        "middle": (q_low_test + q_high_test) / 2,
        "upper_conformal": conformal_upper
    })

    return conformal_df, q_statistic

def build_multi_horizon_conformal_forecasts(
    horizons_minutes: list[int],
    *,
    quantiles: list[float] = [0.05, 0.5, 0.95],
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
    target_coverage: float = 0.90,
    early_stopping_rounds: int = 50,
) -> tuple[dict[int, pd.DataFrame], dict[int, float]]:
    """
    Train + conformalize for all horizons.
    """
    conformal_dfs = {}
    q_by_horizon = {}
    for horizon in horizons_minutes:
        logger.info(f"Training quantile models for horizon {horizon} minutes...")
        models, X_train_df, y_train, X_val_df, y_val, X_test_df, y_true_test, current_rrp_test = train_quantile_models_for_horizon(horizon_minutes=horizon,
                                                                                                                                   quantiles=quantiles,
                                                                                                                                   early_stopping_rounds=early_stopping_rounds)
        
        conformal_df, q_statistic = build_conformal_df_for_horizon(
            horizon_minutes=horizon,
            models=models,
            X_val_df=X_val_df,
            y_val=y_val,
            X_test_df=X_test_df,
            y_true_test=y_true_test,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
            target_coverage=target_coverage
        )

        conformal_dfs[horizon] = conformal_df
        q_by_horizon[horizon] = q_statistic

    return conformal_dfs, q_by_horizon



def make_precomputed_forecast_policy(
    conformal_df: pd.DataFrame,
    *,
    horizon_minutes: int,
    fee_rate: float = 0.01,
    charge_min_ev_per_kwh: float = 0.0,
    discharge_min_ev_per_kwh: float = 0.0,
    ev_scale: float = 0.02,
    power_kw: float | None = None,
    soc_buffer: float = 0.01,
    cooldown_minutes: int = 0,
):
    """
    Policy that uses a single precomputed conformal_df.

    soc_buffer is  there so you don't charge when you are too close to max capacity 
    and dont discharge when youre too close to 0, since the EV calculations might be less reliable at the extremes 

    ev_scale helps determine whether or not to charge or discharge at all. It will control how agressive the policy is 

"""

    def policy(timestamp: pd.Timestamp, price_kwh: float, battery: Battery) -> tuple[str, float]:
        soc = battery.soc 
        forecast_time = timestamp + pd.Timedelta(minutes=horizon_minutes)
        if forecast_time not in conformal_df.index:
            logger.warning(f"Forecast time {forecast_time} not in conformal_df index. Holding by default.")
            return "hold", 0.0
        
        row = conformal_df.loc[forecast_time]

        lower = row["lower_conformal"]
        upper = row["upper_conformal"]
        median = row["median"]

        action = "hold"
        ev_charge = max(0.0, (lower - price_kwh) * ev_scale)
        ev_discharge = max(0.0, (price_kwh - upper) * ev_scale)
        power = 0.0
        if ev_charge >= charge_min_ev_per_kwh * price_kwh and soc < 1.0 - soc_buffer:
            action = "charge"
            power = ev_charge if power_kw is None else min(ev_charge, power_kw)
        elif ev_discharge >= discharge_min_ev_per_kwh * price_kwh and soc > soc_buffer:
            action = "discharge"
            power = ev_discharge if power_kw is None else min(ev_discharge, power_kw)
        logger.debug(f"Timestamp {timestamp}: Forecast for {forecast_time} is (lower: {lower}, "
                     f"median: {median}, upper: {upper}). Price is {price_kwh}. EV_charge: {ev_charge}, "
                     f"EV_discharge: {ev_discharge}. Chosen action: {action} with power {power}. SoC: {soc}")

        return action, power

    params = {
        "horizon_minutes": horizon_minutes,
        "fee_rate": fee_rate,
        "charge_min_ev_per_kwh": charge_min_ev_per_kwh,
        "discharge_min_ev_per_kwh": discharge_min_ev_per_kwh,
        "ev_scale": ev_scale,
        "power_kw": power_kw,
        "soc_buffer": soc_buffer,
        "cooldown_minutes": cooldown_minutes,
    }
    return policy, params


def make_multi_horizon_precomputed_policy(
    conformal_dfs: dict[int, pd.DataFrame],
    *,
    horizons_minutes: list[int],
    fee_rate: float = 0.01,
    charge_min_ev_per_kwh: float = 0.0,
    discharge_min_ev_per_kwh: float = 0.0,
    ev_scale: float = 0.02,
    power_kw: float | None = None,
    soc_buffer: float = 0.01,
    cooldown_minutes: int = 0,
):
    """
    Policy that chooses the best action across multiple horizons (precomputed dfs).
    """
    
    def policy(timestamp: pd.Timestamp, price_kwh: float, battery: Battery) -> tuple[str, float]:
        action = "hold"
        soc = battery.soc
        power = 0.0
        best_horizon = None
        best_ev = -float("inf")

        for horizon in horizons_minutes:
            forecast_time = timestamp + pd.Timedelta(minutes=horizon)
            if forecast_time not in conformal_dfs[horizon].index:
                logger.warning(f"Forecast time {forecast_time} not in conformal_df index for horizon {horizon}. Skipping this horizon.")
                continue
            row = conformal_dfs[horizon].loc[forecast_time]
            lower_h = row["lower_conformal"]
            upper_h = row["upper_conformal"]

            ev_charge = max(0.0, (lower_h - price_kwh) * ev_scale)
            ev_discharge = max(0.0, (price_kwh - upper_h) * ev_scale)

            if lower_h - price_kwh >= charge_min_ev_per_kwh * price_kwh and soc < 1.0 - soc_buffer:
                if ev_charge > best_ev:
                    best_ev = ev_charge
                    action = "charge"
                    power = ev_charge if power_kw is None else min(ev_charge, power_kw)
                    best_horizon = horizon

            if price_kwh - upper_h >= discharge_min_ev_per_kwh * price_kwh and soc > soc_buffer:
                if ev_discharge > best_ev:
                    best_ev = ev_discharge
                    action = "discharge"
                    power = ev_discharge if power_kw is None else min(ev_discharge, power_kw)
                    best_horizon = horizon            


        if best_horizon is None:
            logger.warning(f"No actionable forecasts found for timestamp {timestamp} across all horizons. Holding by default.")
            return "hold", 0.0

        return action, power
    params = {
        "horizons_minutes": horizons_minutes,
        "fee_rate": fee_rate,
        "charge_min_ev_per_kwh": charge_min_ev_per_kwh,
        "discharge_min_ev_per_kwh": discharge_min_ev_per_kwh,
        "ev_scale": ev_scale,
        "power_kw": power_kw,
        "soc_buffer": soc_buffer,
        "cooldown_minutes": cooldown_minutes,
    }
    return policy, params




def make_battery_from_defaults(
    *,
    capacity_kwh: float = 100.0,
    max_power_kw: float = 50.0,
    charge_efficiency: float = 0.9,
    discharge_efficiency: float = 0.9,
    initial_soc: float = 0.5,
) -> Battery:
    """
    Convenience factory so each policy run starts with a fresh identical battery.
    """
    return Battery(
        capacity_kwh=capacity_kwh,
        max_power_kw=max_power_kw,
        charge_efficiency=charge_efficiency,
        discharge_efficiency=discharge_efficiency,
        initial_soc=initial_soc,
    )


def run_policy_once(
    name: str,
    prices_kwh: pd.Series,
    *,
    policy_fn: Callable,
    policy_params: dict,
    dt_hours: float,
    fee_rate: float,
    degradation_rate_per_kwh: float,
):
    """
    Run one policy on one aligned price series and print metrics."""
    battery = make_battery_from_defaults()
    actions_df, metrics = run_arbitrage_simulation(
        prices=prices_kwh,
        battery=battery,
        dt_hours=dt_hours,
        policy=policy_fn(**policy_params),
        fee_rate=fee_rate,
        degradation_rate_per_kwh=degradation_rate_per_kwh
    )
    logger.info(f"Policy {name} - Total Profit: ${metrics.total_profit:.2f}, Terminal Value: ${metrics.terminal_value:.2f}, "
                f"Total Profit with Terminal: ${metrics.total_profit_with_terminal:.2f}, Max Drawdown: ${metrics.max_drawdown:.2f}, "
                f"Grid Throughput: {metrics.grid_throughput_kwh:.2f} kWh, Equivalent Cycles: {metrics.equivalent_cycles:.4f}")
    return actions_df, metrics


def build_price_series_mwh(test_df: pd.DataFrame) -> pd.Series:
    """
    Extract the test price series in $/MWh with DatetimeIndex.
    """
    if "SETTLEMENTDATE" not in test_df.columns:
        raise ValueError("Expected 'SETTLEMENTDATE' column in test data.")
    if "RRP" not in test_df.columns:
        raise ValueError("Expected 'RRP' column in test data.")

    series_mwh = pd.Series(
        test_df["RRP"].to_numpy(),
        index=pd.DatetimeIndex(test_df["SETTLEMENTDATE"]),
        name="RRP",
    )
    return _validate_prices(series_mwh)

def main_experiment():
    pass

if __name__ == "__main__":
    main_experiment()
