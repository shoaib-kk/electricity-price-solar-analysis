
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import time
import warnings

def fit_arima(series: pd.Series, order: tuple, max_iterations = 50):
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


def baseline_TSA(df, p_value = None, q_value = None):
    """Performs Time Series Analysis using ARIMA model to act as a baseline,
    ignores most features besides RRP series."""

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex"
"                        clean data first.")

    rrp = df["RRP"].resample("30min").mean().ffill().dropna()

    # Log-transform RRP with a positive shift to handle
    # negative prices without dropping data.
    shift = 1 - min(0.0, rrp.min())
    rrp_log = np.log(rrp + shift)

    # Manually difference the log series and fix d=0 in ARIMA,
    # so basically fitting an ARMA(p,q) model to the stationary series.
    rrp_log_diff = rrp_log.diff().dropna()

    if p_value is not None and q_value is not None:
        print(f"Fitting ARIMA({p_value},0,{q_value}) model...")
        start_time = time.time()
        model = fit_arima(rrp_log_diff, order=(p_value, 0, q_value))
        if model is None:
            raise RuntimeError (f"ARIMA({p_value},0,{q_value}) model fitting failed.")
            
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
        print(f"Fitting ARIMA({best_order[0]},0,{best_order[2]}) model...")
        print(f"Model fitted in {elapsed_time:.2f} seconds\n")
    print(best_model.summary())

    print("\nGenerating forecast...")

    # Forecast on the differenced log scale, then invert the differencing
    # and log-transform back to the original price scale.
    diff_forecast = best_model.forecast(steps=48) 
    last_log_level = rrp_log.iloc[-1]
    log_level_forecast = last_log_level + diff_forecast.cumsum()
    price_forecast = np.exp(log_level_forecast) - shift

    print("Forecasted RRP for next 24 hours (48 half-hour intervals):")
    print(price_forecast)


def main():
    file_path = "CLEANED_PRICE_AND_DEMAND_VIC1_TRAIN.csv"
    df = pd.read_csv(file_path, parse_dates=["SETTLEMENTDATE"])

    print("Initial data info:")
    print(df.info())

    print("\n" + "=" * 70)
    print("Running ARIMA analysis for baseline comparison")
    print("=" * 70)
    baseline_TSA(df)
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
