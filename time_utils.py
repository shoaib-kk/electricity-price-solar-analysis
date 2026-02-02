import pandas as pd


def infer_step_minutes(index, fallback_minutes: float = 5.0) -> float:
    """Infer the time step of a DatetimeIndex in minutes.

    Uses the mode of first differences. Falls back to `fallback_minutes`
    if no differences are available.
    """

    if isinstance(index, pd.DatetimeIndex):
        diffs = index.to_series().diff().dropna()
    else:
        # Best-effort: coerce to a Series and compute diffs
        diffs = pd.Series(index).diff().dropna()

    if diffs.empty:
        return float(fallback_minutes)

    delta = diffs.mode()[0]
    if isinstance(delta, pd.Timedelta):
        return float(delta.total_seconds() / 60.0)

    raise TypeError("infer_step_minutes expects a DatetimeIndex or time-like sequence.")
