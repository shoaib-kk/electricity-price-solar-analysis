from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from Battery_model import Battery
from Quantile_regression import (
    manage_train_test_split,

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
    max_equity_drawdown: float 
    grid_throughput_kwh: float
    equivalent_cycles: float
    profit_per_cycle: float = 0.0
    profit_drawdown_ratio: float = 0.0



def _conformal_q_from_scores(scores: np.ndarray, target_coverage: float) -> float:
    """Compute the conformal q statistic from nonconformity scores and target coverage level"""
    if scores.size == 0:
        raise ValueError("No nonconformity scores to compute conformal q.")
    alpha = 1.0 - target_coverage
    n = scores.size
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    sorted_scores = np.sort(scores)
    return float(sorted_scores[k - 1])

def _validate_prices(prices: pd.Series) -> pd.Series:
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series indexed by time.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must have a DatetimeIndex.")
    if not np.isfinite(prices.values).all():
        raise ValueError("prices contains NaN or infinite values.")
    return prices.sort_index()


def max_drawdown(series: pd.Series) -> float:
    """"Drawdown is just the max drop from a previous peak"""
    if not series.index.is_unique:
        raise ValueError("Series index must be unique for drawdown computation.")
    running_max = series.cummax()
    drawdown = running_max - series
    return float(drawdown.max())


def baseline_threshold_policy(training_prices_kwh: pd.Series, low_quantile=0.3, high_quantile=0.7):
    "policy where you charge when price is below low_quantile and discharge when above high_quantile of training price distribution"
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
            logger.debug(f"Timestamp {timestamp}: Holding because price {price_kwh} is between thresholds")
            return "hold", 0.0

    params = {"low_quantile": low_quantile, "high_quantile": high_quantile, 
              "low_threshold": low_threshold, "high_threshold": high_threshold}
    return policy, params


def wrap_spike_discharge_policy(
    policy: Callable,
    training_prices_kwh: pd.Series,
    spike_price_quantile: float = 0.99,
    spike_discharge_power: float | None = None,
    soc_buffer: float = 0.01,
) -> tuple[Callable, dict]:
    spike_threshold_kwh = float(training_prices_kwh.quantile(spike_price_quantile))

    def wrapped_policy(timestamp, price_kwh, battery) -> tuple[str, float]:
        if price_kwh >= spike_threshold_kwh and battery.soc > soc_buffer:
            power = battery.max_power_kw if spike_discharge_power is None else spike_discharge_power
            return "discharge", power
        return policy(timestamp, price_kwh, battery)

    params = {
        "spike_price_quantile": spike_price_quantile,
        "spike_discharge_power": spike_discharge_power,
        "soc_buffer": soc_buffer,
        "spike_threshold_kwh": spike_threshold_kwh,
    }
    return wrapped_policy, params





def mwh_to_kwh(price_mwh: float) -> float:
    return price_mwh / 1000.0

def series_mwh_to_kwh(prices_mwh: pd.Series) -> pd.Series:
    return prices_mwh.apply(mwh_to_kwh)


def filter_prices_by_period(
    prices: pd.Series,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.Series:
    
    """"Creates a certain price period based on the start and end dates """
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



def train_quantile_models_for_horizon(
    horizon_minutes: int,
    quantiles: list[float] = [0.05, 0.5, 0.95],
    early_stopping_rounds: int = 50,
):
    X_train_df, y_train, X_val_df, y_val, X_test_df, y_true_test, current_rrp_test = manage_train_test_split(
        forecast_horizon_minutes=horizon_minutes
    )
    models = train_quantile_model(
        X_train_df,
        y_train,
        quantiles=quantiles,
        X_val=X_val_df,
        y_val=y_val,
        early_stopping_rounds=early_stopping_rounds,
    )

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
    rolling_window: int | None = None,
    min_warmup: int = 200,
) -> tuple[pd.DataFrame, float]:
    """"Build a DataFrame containing conformal prediction intervals and median forecasts for a specific horizon."""

    q_low_val = pd.Series(models[quantile_low].predict(X_val_df), index=y_val.index)
    q_high_val = pd.Series(models[quantile_high].predict(X_val_df), index=y_val.index)

    # conformity scores are how far the true value is outside our predicted interval for NEM prices
    nonconformity = np.maximum(q_low_val - y_val, y_val - q_high_val)
    nonconformity = np.maximum(nonconformity, 0.0).to_numpy()
    
    if nonconformity.size == 0:
        raise ValueError

    # q stat is juts the quantile of the conformity scores needed to correct nonconformity 
    q_statistic_static = _conformal_q_from_scores(nonconformity, target_coverage)

    q_low_test = pd.Series(models[quantile_low].predict(X_test_df), index=y_true_test.index)
    q_high_test = pd.Series(models[quantile_high].predict(X_test_df), index=y_true_test.index)

    if 0.5 in models:
        q_median_test = pd.Series(models[0.5].predict(X_test_df), index=y_true_test.index)
    else:
        q_median_test = (q_low_test + q_high_test) / 2

    test_index = y_true_test.index.sort_values()
    y_true_test = y_true_test.loc[test_index]
    q_low_test = q_low_test.loc[test_index]
    q_high_test = q_high_test.loc[test_index]
    q_median_test = q_median_test.loc[test_index]

    if rolling_window is None or rolling_window <= 0:
        # if no rolling window is specified, just use the static q statistic for all test points to create conformal intervals
        conformal_lower = q_low_test - q_statistic_static
        conformal_upper = q_high_test + q_statistic_static
        q_stat_series = pd.Series(q_statistic_static, index=y_true_test.index)
    else:
        nonconf_window = deque(maxlen=rolling_window)
        lower_vals = []
        upper_vals = []
        q_values = []
        for ts in y_true_test.index:
            window_n = len(nonconf_window)
            q_t = q_statistic_static if window_n < min_warmup else _conformal_q_from_scores(np.array(nonconf_window), target_coverage)
            
            lower_vals.append(q_low_test.loc[ts] - q_t)
            upper_vals.append(q_high_test.loc[ts] + q_t)
            q_values.append(q_t)

            y_t = y_true_test.loc[ts]
            nonconf_t = max(q_low_test.loc[ts] - y_t, y_t - q_high_test.loc[ts], 0.0)
            nonconf_window.append(nonconf_t)

        conformal_lower = pd.Series(lower_vals, index=y_true_test.index)
        conformal_upper = pd.Series(upper_vals, index=y_true_test.index)
        q_stat_series = pd.Series(q_values, index=y_true_test.index)

    conformal_df = pd.DataFrame({
        "lower_conformal": conformal_lower,
        "median": q_median_test,
        "upper_conformal": conformal_upper,
        "q_statistic": q_stat_series,
    })

    return conformal_df, q_statistic_static


def build_multi_horizon_conformal_forecasts(
    horizons_minutes: list[int],
    *,
    quantiles: list[float] = [0.05, 0.5, 0.95],
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
    target_coverage: float = 0.90,
    early_stopping_rounds: int = 50,
    rolling_window: int | None = None,
    min_warmup: int = 200,
) -> tuple[dict[int, pd.DataFrame], dict[int, float]]:
    """"Build conformal forecasts for multiple horizons and return a dictionary of DataFrames and q statistics indexed by horizon."""
    conformal_dfs = {}
    q_by_horizon = {}
    for horizon in horizons_minutes:
        logger.info(f"Training quantile models for horizon {horizon} minutes...")
        models, X_train_df, y_train, X_val_df, y_val, X_test_df, y_true_test, current_rrp_test = train_quantile_models_for_horizon(
            horizon_minutes=horizon,
            quantiles=quantiles,
            early_stopping_rounds=early_stopping_rounds
        )
        
        conformal_df, q_statistic_static = build_conformal_df_for_horizon(
            horizon_minutes=horizon,
            models=models,
            X_val_df=X_val_df,
            y_val=y_val,
            X_test_df=X_test_df,
            y_true_test=y_true_test,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
            target_coverage=target_coverage,
            rolling_window=rolling_window,
            min_warmup=min_warmup,
        )

        conformal_dfs[horizon] = conformal_df
        q_by_horizon[horizon] = q_statistic_static

    return conformal_dfs, q_by_horizon


def make_aggressive_precomputed_forecast_policy(
    conformal_df: pd.DataFrame,
    *,
    horizon_minutes: int,
    fee_rate: float = 0.01,
    cost_per_kwh: float = 0.0,
    min_signal_charge_aud_per_kwh: float = 0.0,
    min_signal_discharge_aud_per_kwh: float = 0.0,
    power_kw: float | None = None,
    soc_buffer: float = 0.01,
):
    """
    Aggressive forecast-based policy:
    - No interval-width filtering
    - No confidence scaling
    - Full power when signal exceeds threshold
    """

    def policy(timestamp: pd.Timestamp, price_kwh: float, battery) -> tuple[str, float]:
        soc = battery.soc
        forecast_time = timestamp + pd.Timedelta(minutes=horizon_minutes)

        if forecast_time not in conformal_df.index:
            return "hold", 0.0

        row = conformal_df.loc[forecast_time]
        median = row["median"]

        # round-trip fee cost
        round_trip_fee_cost = 2 * fee_rate * abs(price_kwh)
        total_cost = cost_per_kwh + round_trip_fee_cost

        signal_charge = median - price_kwh - total_cost
        signal_discharge = price_kwh - median - total_cost

        action = "hold"
        power = 0.0

        # Use full power if threshold exceeded
        full_power = battery.max_power_kw if power_kw is None else power_kw

        if signal_charge >= min_signal_charge_aud_per_kwh and soc < 1.0 - soc_buffer:
            action = "charge"
            power = full_power

        elif signal_discharge >= min_signal_discharge_aud_per_kwh and soc > soc_buffer:
            action = "discharge"
            power = full_power

        return action, power

    params = {
        "horizon_minutes": horizon_minutes,
        "fee_rate": fee_rate,
        "cost_per_kwh": cost_per_kwh,
        "min_signal_charge_aud_per_kwh": min_signal_charge_aud_per_kwh,
        "min_signal_discharge_aud_per_kwh": min_signal_discharge_aud_per_kwh,
        "power_kw": power_kw,
        "soc_buffer": soc_buffer,
    }

    return policy, params


def make_multi_horizon_aggressive_policy(
    conformal_dfs: dict[int, pd.DataFrame],
    *,
    horizons_minutes: list[int],
    fee_rate: float = 0.01,
    cost_per_kwh: float = 0.0,
    min_signal_charge_aud_per_kwh: float = 0.0,
    min_signal_discharge_aud_per_kwh: float = 0.0,
    power_kw: float | None = None,
    soc_buffer: float = 0.01,
    horizon_discount_minutes: float | None = None,
    edge_k: float = 0.1,
    edge_buffer_aud_per_kwh: float = 0.0,
    min_margin_aud_per_kwh: float = 0.0,
    min_hold_steps: int = 0,
    min_switch_delta_aud_per_kwh: float = 0.0,
    min_power_frac: float = 0.05,
    fallback_target_soc: float = 0.5,
    fallback_band: float = 0.05,
    fallback_power_frac: float = 0.1,
    fallback_price_bias: float = 0.0,
    collect_diagnostics: bool = False,
):
    """
    Best-opportunity policy:
    - For each horizon: risk-adjusted edge vs width and costs
    - Pick the single horizon with max |edge/(width+eps)|
    - Require |edge| > k*width + fees + buffer + min_margin
    - Power scales with margin above the gate; drops micro-trades
    - Optional cooldown and hysteresis to reduce churn
    - Optional diagnostics capture per step (action gating, margins, horizons)
    """
    last_trade_action = "hold"
    last_trade_edge = 0.0
    steps_since_trade_change = max(min_hold_steps, 0)

    debug_records = [] if collect_diagnostics else None

    def policy(timestamp: pd.Timestamp, price_kwh: float, battery: Battery) -> tuple[str, float]:
        nonlocal last_trade_action, last_trade_edge, steps_since_trade_change
        soc = battery.soc

        best_score = float("-inf")
        best_signed_edge = None
        best_required_edge = None
        best_width = None
        best_action = "hold"
        best_horizon = None
        best_discount = None
        best_total_cost = None

        for horizon in horizons_minutes:
            forecast_time = timestamp + pd.Timedelta(minutes=horizon)
            if forecast_time not in conformal_dfs[horizon].index:
                continue

            row = conformal_dfs[horizon].loc[forecast_time]
            lower_h = row["lower_conformal"]
            upper_h = row["upper_conformal"]
            median_h = row["median"] if "median" in row else (lower_h + upper_h) / 2

            round_trip_fee_cost = 2 * fee_rate * abs(price_kwh)
            total_cost = cost_per_kwh + round_trip_fee_cost

            signed_edge = median_h - price_kwh  # positive favors charge, negative favors discharge
            width = float(upper_h - lower_h)
            gate = edge_k * width + edge_buffer_aud_per_kwh + total_cost + min_margin_aud_per_kwh

            if signed_edge > 0:
                action_dir = "charge"
                edge_excess = signed_edge - gate
            else:
                action_dir = "discharge"
                edge_excess = -signed_edge - gate

            if edge_excess <= 0:
                continue

            if horizon_discount_minutes is None:
                discount_h = 1.0
            else:
                discount_h = 1.0 / (1.0 + horizon / horizon_discount_minutes)

            score = (abs(signed_edge) / (width + 1e-9)) * discount_h
            if score > best_score:
                best_score = score
                best_signed_edge = signed_edge
                best_required_edge = gate
                best_width = width
                best_action = action_dir
                best_horizon = horizon
                best_discount = discount_h
                best_total_cost = total_cost

        final_action = "hold"
        final_power = 0.0
        hold_reason = None

        if best_action == "hold" or best_signed_edge is None or best_required_edge is None or best_width is None:
            hold_reason = "no_horizon_passed_gate"
            steps_since_trade_change += 1

            full_power = battery.max_power_kw if power_kw is None else power_kw
            fallback_power = full_power * fallback_power_frac

            if soc > fallback_target_soc + fallback_band and soc > soc_buffer and price_kwh >= fallback_price_bias:
                final_action = "discharge"
                final_power = fallback_power
                hold_reason = "fallback_discharge_to_target"
                last_trade_action = "discharge"
                last_trade_edge = 0.0
            elif soc < fallback_target_soc - fallback_band and soc < 1.0 - soc_buffer:
                final_action = "charge"
                final_power = fallback_power
                hold_reason = "fallback_charge_to_target"
                last_trade_action = "charge"
                last_trade_edge = 0.0
        else:
            full_power = battery.max_power_kw if power_kw is None else power_kw

            action = best_action
            if action == "charge" and (best_signed_edge < min_signal_charge_aud_per_kwh or soc >= 1.0 - soc_buffer):
                hold_reason = "charge_blocked_signal_or_soc"
                action = "hold"
            if action == "discharge" and (-best_signed_edge < min_signal_discharge_aud_per_kwh or soc <= soc_buffer):
                hold_reason = "discharge_blocked_signal_or_soc"
                action = "hold"

            steps_since_trade_change += 1
            if action in {"charge", "discharge"}:
                if last_trade_action in {"charge", "discharge"} and action != last_trade_action:
                    if steps_since_trade_change <= min_hold_steps:
                        hold_reason = "cooldown_hold"
                        action = "hold"
                    elif abs(best_signed_edge - last_trade_edge) < min_switch_delta_aud_per_kwh:
                        hold_reason = "switch_delta_hold"
                        action = "hold"

                if action in {"charge", "discharge"}:
                    if action != last_trade_action:
                        steps_since_trade_change = 0
                    last_trade_action = action
                    last_trade_edge = best_signed_edge

                    margin = abs(best_signed_edge) - best_required_edge
                    strength = max(0.0, min(1.0, margin / (abs(best_signed_edge) + 1e-6)))
                    power_raw = strength * full_power
                    min_power = min_power_frac * full_power
                    if power_raw < min_power:
                        hold_reason = "below_min_power"
                        action = "hold"
                    else:
                        final_action = action
                        final_power = max(min_power, min(power_raw, full_power))

        if collect_diagnostics:
            debug_records.append(
                {
                    "timestamp": timestamp,
                    "price_kwh": price_kwh,
                    "soc": soc,
                    "best_horizon": best_horizon,
                    "best_width": best_width,
                    "best_signed_edge": best_signed_edge,
                    "required_edge": best_required_edge,
                    "edge_margin": (abs(best_signed_edge) - best_required_edge) if (best_signed_edge is not None and best_required_edge is not None) else None,
                    "score": None if best_score == float("-inf") else best_score,
                    "discount": best_discount,
                    "total_cost": best_total_cost,
                    "action": final_action,
                    "hold_reason": hold_reason,
                    "power_kw": final_power,
                    "steps_since_trade_change": steps_since_trade_change,
                    "last_trade_action": last_trade_action,
                }
            )

        return final_action, final_power

    params = {
        "conformal_dfs": conformal_dfs,
        "horizons_minutes": horizons_minutes,
        "fee_rate": fee_rate,
        "cost_per_kwh": cost_per_kwh,
        "min_signal_charge_aud_per_kwh": min_signal_charge_aud_per_kwh,
        "min_signal_discharge_aud_per_kwh": min_signal_discharge_aud_per_kwh,
        "power_kw": power_kw,
        "soc_buffer": soc_buffer,
        "horizon_discount_minutes": horizon_discount_minutes,
        "edge_k": edge_k,
        "edge_buffer_aud_per_kwh": edge_buffer_aud_per_kwh,
        "min_margin_aud_per_kwh": min_margin_aud_per_kwh,
        "min_hold_steps": min_hold_steps,
        "min_switch_delta_aud_per_kwh": min_switch_delta_aud_per_kwh,
                "min_power_frac": min_power_frac,
                "fallback_target_soc": fallback_target_soc,
                "fallback_band": fallback_band,
                "fallback_power_frac": fallback_power_frac,
                "fallback_price_bias": fallback_price_bias,
        "collect_diagnostics": collect_diagnostics,
    }

    if collect_diagnostics:
        policy.debug_records = debug_records
    else:
        policy.debug_records = None

    return policy, params

def make_multi_horizon_precomputed_policy(
    conformal_dfs: dict[int, pd.DataFrame],
    *,
    horizons_minutes: list[int],
    fee_rate: float = 0.01,
    cost_per_kwh: float = 0.0,
    min_signal_charge_aud_per_kwh: float = 0.0,
    min_signal_discharge_aud_per_kwh: float = 0.0,
    ev_scale: float = 0.02,
    power_kw: float | None = None,
    soc_buffer: float = 0.01,
    strength_k: float = 0.1,
    min_activity: float = 1e-4,
    confidence_cap: float | None = 10.0,
    use_interval_confidence: bool = True,
    c: float = 0.01,
    price_epsilon: float = 1.0,
):
    """"Policy function that aggregates signals from multiple precomputed conformal forecast intervals 
    across different horizons to make charging/discharging decisions"""
    def policy(timestamp: pd.Timestamp, price_kwh: float, battery: Battery) -> tuple[str, float]:
        action = "hold"
        soc = battery.soc
        power = 0.0
        
        aggregate_charge = 0.0
        aggregate_discharge = 0.0
        weight_sum = 0.0
        
        for horizon in horizons_minutes:
            forecast_time = timestamp + pd.Timedelta(minutes=horizon)
            if forecast_time not in conformal_dfs[horizon].index:
                continue
            
            row = conformal_dfs[horizon].loc[forecast_time]
            lower_h = row["lower_conformal"]
            upper_h = row["upper_conformal"]
            median_h = row["median"] if "median" in row else (lower_h + upper_h) / 2
            
            round_trip_fee_cost = 2 * fee_rate * abs(price_kwh)
            total_cost = cost_per_kwh + round_trip_fee_cost
            
            signal_charge = max(0.0, median_h - price_kwh - total_cost)
            signal_discharge = max(0.0, price_kwh - median_h - total_cost)
            
            discount_h = 1 / (1 + horizon / 60)
            
            if use_interval_confidence:
                interval_width = upper_h - lower_h
                scale = max(abs(median_h), abs(price_kwh), price_epsilon)
                relative_width_h = interval_width / scale
                # treat width as a penalty rather than a reward; cap near 1 for tight intervals
                confidence_h = 1.0 / (1.0 + relative_width_h / max(c, 1e-9))
                if confidence_cap is not None:
                    confidence_h = min(confidence_h, confidence_cap)
            else:
                confidence_h = 1.0
            
            weight_h = discount_h * confidence_h
            
            aggregate_charge += signal_charge * weight_h
            aggregate_discharge += signal_discharge * weight_h
            weight_sum += weight_h

        if weight_sum < 1e-9:
            return "hold", 0.0
            
        net = aggregate_charge - aggregate_discharge
        net_norm = net / weight_sum
        activity = aggregate_charge + aggregate_discharge

        if activity >= min_activity:
            if net_norm >= min_signal_charge_aud_per_kwh and soc < 1.0 - soc_buffer:
                action = "charge"
            elif net_norm <= -min_signal_discharge_aud_per_kwh and soc > soc_buffer:
                action = "discharge"

            if action in {"charge", "discharge"}:
                strength = abs(net_norm) / (abs(net_norm) + strength_k)
                strength = max(0.0, min(1.0, strength))
                power = strength * (power_kw if power_kw is not None else battery.max_power_kw)
                power = max(0.0, power)

        return action, power
    
    params = {
        "horizons_minutes": horizons_minutes,
        "fee_rate": fee_rate,
        "cost_per_kwh": cost_per_kwh,
        "min_signal_charge_aud_per_kwh": min_signal_charge_aud_per_kwh,
        "min_signal_discharge_aud_per_kwh": min_signal_discharge_aud_per_kwh,
        "ev_scale": ev_scale,
        "power_kw": power_kw,
        "soc_buffer": soc_buffer,
        "strength_k": strength_k,
        "min_activity": min_activity,
        "confidence_cap": confidence_cap,
        "use_interval_confidence": use_interval_confidence,
        "c": c,
        "price_epsilon": price_epsilon,
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
    """Convenience factory for fresh battery instances."""
    return Battery(
        capacity_kwh=capacity_kwh,
        max_power_kw=max_power_kw,
        charge_efficiency=charge_efficiency,
        discharge_efficiency=discharge_efficiency,
        initial_soc=initial_soc,
    )
def run_arbitrage_simulation(
    prices: pd.Series, 
    battery: Battery, 
    dt_hours: float, 
    policy: Callable, 
    fee_rate: float = 0.01,
    degradation_cost_per_kwh: float = 0.0,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None
) -> tuple[pd.DataFrame, SimulationMetrics]:
    """Run a simulation of battery arbitrage given a price series, battery model, and policy function. 
    Returns a DataFrame of actions and a SimulationMetrics summary."""
    prices = _validate_prices(prices)
    prices = filter_prices_by_period(prices, start=start_date, end=end_date)
    records = []
    cumulative_profit = 0.0

    for timestamp, price_kwh in prices.items():
        price_kwh = float(price_kwh)

        action, power_kw = policy(timestamp, price_kwh, battery)
        energy_bought, energy_sold, new_soc = battery.step(action, power_kw, dt_hours)

        # NEM prices might be negative so have to avoid ending up with a negative buy fee or sell fee
        buy_fee = fee_rate * energy_bought * abs(price_kwh)
        sell_fee = fee_rate * energy_sold * abs(price_kwh)
        transaction_cost = buy_fee + sell_fee
        degradation_cost = degradation_cost_per_kwh * (energy_bought + energy_sold)
        
        profit = energy_sold * price_kwh - energy_bought * price_kwh - transaction_cost - degradation_cost
        cumulative_profit += profit
        equity = cumulative_profit + battery.soc_kwh * price_kwh

        records.append({
            "timestamp": timestamp,
            "price_kwh": price_kwh,
            "action": action,
            "power_kw": power_kw,
            "energy_bought_kwh": energy_bought,
            "energy_sold_kwh": energy_sold,
            "new_soc": new_soc,
            "profit": profit,
            "cumulative_profit": cumulative_profit,
            "equity": equity,
        })

    actions_df = pd.DataFrame.from_records(records).set_index("timestamp")

    last_price = actions_df["price_kwh"].iloc[-1]
    
    terminal_value = battery.soc_kwh * last_price * (1 - fee_rate)
    total_profit_with_terminal = cumulative_profit + terminal_value

    cash_drawdown = max_drawdown(actions_df["cumulative_profit"])
    equity_drawdown = max_drawdown(actions_df["equity"])
    
    total_throughput = actions_df["energy_bought_kwh"].sum() + actions_df["energy_sold_kwh"].sum()
    equivalent_cycles = total_throughput / (2 * battery.capacity_kwh) if battery.capacity_kwh > 0 else 0.0

    metrics = SimulationMetrics(
        total_profit=cumulative_profit,
        terminal_value=terminal_value,
        total_profit_with_terminal=total_profit_with_terminal,
        max_drawdown=cash_drawdown,
        max_equity_drawdown=equity_drawdown,
        grid_throughput_kwh=total_throughput,
        equivalent_cycles=equivalent_cycles,
        profit_drawdown_ratio=cumulative_profit / equity_drawdown if equity_drawdown > 0 else float("inf"),
        profit_per_cycle=cumulative_profit / equivalent_cycles if equivalent_cycles > 0 else float("inf")
    )
    
    return actions_df, metrics 
def run_policy(
    name: str,
    prices_kwh: pd.Series,
    *,
    policy_constructor: Callable,
    policy_params: dict,
    dt_hours: float,
    fee_rate: float,
    degradation_cost_per_kwh: float,
) -> tuple[pd.DataFrame, SimulationMetrics]:
    battery = make_battery_from_defaults()
    policy_callable, _ = policy_constructor(**policy_params)
    
    actions_df, metrics = run_arbitrage_simulation(
        prices=prices_kwh,
        battery=battery,
        dt_hours=dt_hours,
        policy=policy_callable,
        fee_rate=fee_rate,
        degradation_cost_per_kwh=degradation_cost_per_kwh
    )
    
    logger.info(
        f"Policy {name} - "
        f"Total Profit: ${metrics.total_profit:.2f}, "
        f"Terminal Value: ${metrics.terminal_value:.2f}, "
        f"Profit+Terminal: ${metrics.total_profit_with_terminal:.2f}, "
        f"Cash Drawdown: ${metrics.max_drawdown:.2f}, "
        f"Equity Drawdown: ${metrics.max_equity_drawdown:.2f}, "
        f"Cycles: {metrics.equivalent_cycles:.2f}"
    )
    return actions_df, metrics


def build_price_series_mwh(test_df: pd.DataFrame) -> pd.Series:
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


def main():
    pass

if __name__ == "__main__":
    main()