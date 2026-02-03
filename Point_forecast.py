
import pandas as pd
import numpy as np
import time
import xgboost as xgb
import time_utils
import Metrics_utils
from logging_utils import setup_logging
import logging 

setup_logging()


def load_cleaned_datasets():
    train_path = "CLEANED_PRICE_AND_DEMAND_VIC1_TRAIN.csv"
    test_path = "CLEANED_PRICE_AND_DEMAND_VIC1_TEST.csv"

    train = pd.read_csv(train_path, parse_dates=["SETTLEMENTDATE"])
    test = pd.read_csv(test_path, parse_dates=["SETTLEMENTDATE"])

    return train, test
def get_feature_columns(train_df, drop_cols):
    feature_cols = []
    for col in train_df.columns:
        if col in drop_cols:
            continue
        if np.issubdtype(train_df[col].dtype, np.number):
            feature_cols.append(col)

    feature_cols = sorted(feature_cols)
    return feature_cols

def build_feature_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon_minutes: int = 30,
):
    """Build X, y for h-step-ahead point forecasting from engineered features.
    Target: RRP at time t + h (true forecast), using features at time t.
    """

    if "RRP" not in train_df.columns:
        raise ValueError("Expected column 'RRP' in cleaned training data.")
    if "SETTLEMENTDATE" not in train_df.columns:
        raise ValueError("Expected 'SETTLEMENTDATE' column in cleaned training data.")

    # sort them in-place.
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df.sort_values("SETTLEMENTDATE", inplace=True)
    test_df.sort_values("SETTLEMENTDATE", inplace=True)

    # Infer step size in minutes from the training timestamps 
    step_minutes = time_utils.infer_step_minutes(train_df["SETTLEMENTDATE"], fallback_minutes=5.0)
    ratio = horizon_minutes / step_minutes
    horizon_steps = max(1, int(round(ratio)))

    if abs(ratio - horizon_steps) > 0.001:
        raise ValueError(
            f"Requested horizon {horizon_minutes} minutes is not an integer "
            f"multiple of inferred step {step_minutes:.3f} minutes."
        )
    # 1440 minutes in a day
    #steps_per_day = int(round(1440.0 / step_minutes))

    # Define h-step-ahead forecast target
    train_df["RRP_target"] = train_df["RRP"].shift(-horizon_steps)
    test_df["RRP_target"] = test_df["RRP"].shift(-horizon_steps)

    # Drop rows that don't have a valid future target value; keep
    # rows even if some feature columns contain NaNs.
    train_df = train_df.dropna(subset=["RRP_target"])
    test_df = test_df.dropna(subset=["RRP_target"])

    y_train = train_df["RRP_target"].values
    y_test = test_df["RRP_target"].values

    drop_cols = {"SETTLEMENTDATE", "REGION", "PERIODTYPE", "RRP", "TOTALDEMAND", "RRP_target"}

    feature_cols = get_feature_columns(train_df, drop_cols)
    
    # Ensure all chosen feature columns exist in test as well
    missing_in_test = [c for c in feature_cols if c not in test_df.columns]
    if missing_in_test:
        raise ValueError(f"Feature columns missing in test set: {missing_in_test}")

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    n_train_nans = int(np.isnan(X_train).sum())
    n_test_nans = int(np.isnan(X_test).sum())
    total_train = X_train.size
    total_test = X_test.size
    pct_train = (n_train_nans / total_train * 100.0) if total_train else 0.0
    pct_test = (n_test_nans / total_test * 100.0) if total_test else 0.0

    # just in case even tho should be zero 
    logging.info(
        f"NaNs in X_train: {n_train_nans} ({pct_train:.4f}% of entries), "
        f"NaNs in X_test: {n_test_nans} ({pct_test:.4f}% of entries)"
    )

    # Also report per-feature NaN counts when there are any
    if n_train_nans or n_test_nans:
        train_nan_per_feature = train_df[feature_cols].isna().sum()
        test_nan_per_feature = test_df[feature_cols].isna().sum()
        logging.info("NaNs per feature (train):")
        logging.info(train_nan_per_feature[train_nan_per_feature > 0].sort_values(ascending=False))
        logging.info("NaNs per feature (test):")
        logging.info(test_nan_per_feature[test_nan_per_feature > 0].sort_values(ascending=False))

    # For return/direction metrics also need the current price at time t
    current_rrp_test = test_df["RRP"].values

    return X_train, y_train, X_test, y_test, feature_cols, current_rrp_test #, steps_per_day


def fit_xgboost_regressor(X_train, y_train, X_val, y_val, feature_cols=None):
    """Fit an XGBoost regressor with early stopping with xgb.train.

    Uses a time-based validation split and stops boosting when
    validation RMSE stops improving, safer than
    running a fixed number of trees.
    """

    # X_all_train / X_val are built from feature_cols in a fixed order;
    # keep those names on the DMatrices for interpretability.
    # Assume X_train/X_val share the same column ordering.
    if feature_cols is not None and X_train.ndim == 2 and len(feature_cols) == X_train.shape[1]:
        feature_names = list(feature_cols)
    elif X_train.ndim == 2:
        feature_names = [f"f{i}" for i in range(X_train.shape[1])]
    else:
        feature_names = None

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = (
        xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
        if X_val is not None and len(X_val) > 0
        else None
    )

    parameters = {
        "objective": "reg:squarederror",
        "max_depth": 5,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "rmse",
        "seed": 42,
    }

    evals = [(dtrain, "train")]
    if dval is not None:
        evals.append((dval, "val"))

    start = time.time()
    boosted = xgb.train(
        parameters,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50 if dval is not None else None,
        verbose_eval=False,
    )
    elapsed = time.time() - start
    print(f"XGBoost model fitted in {elapsed:.2f} seconds")

    if dval is not None:
        y_val_pred = boosted.predict(dval)
        val_mae = float(np.mean(np.abs(y_val - y_val_pred)))
        val_rmse = float(np.sqrt(np.mean((y_val - y_val_pred) ** 2)))
        print("Validation performance (early stopping used):")
        print(f"  MAE : {val_mae:.2f}")
        print(f"  RMSE: {val_rmse:.2f}")

    return boosted


def evaluate_point_forecast(y_true, y_pred, y_current=None, label="Model"):
    """Print basic regression metrics for a forecast."""

    metrics = Metrics_utils.compute_mae_rmse_smape(y_true, y_pred)
    mae = metrics["mae"]
    rmse = metrics["rmse"]

    logging.info(f"\nPoint forecast evaluation on TEST set ({label}):")
    logging.info(f"  MAE  : {mae:.2f}")
    logging.info(f"  RMSE : {rmse:.2f}")
    logging.info(
        "  Note: sMAPE is not reported because it becomes unstable "
        "and misleading when prices are near zero or negative."
    )

    # Optional: report behaviour on "returns" and direction, which is
    # more relevant for arbitrage decisions than raw price error.
    if y_current is not None:
        Metrics_utils.evaluate_direction_accuracy(
            y_true,
            y_pred,
            y_current,
            label=label,
        )


def _get_feature_names_for_matrix(X, feature_cols=None):
    """Derive feature names for a DMatrix given columns and shape."""

    if feature_cols is not None and X.ndim == 2 and len(feature_cols) == X.shape[1]:
        return list(feature_cols)
    if X.ndim == 2:
        return [f"f{i}" for i in range(X.shape[1])]
    return None

def main():
    train_df, test_df = load_cleaned_datasets()
    forecast_horizon_minutes = 30
    (X_all_train,y_all_train, X_test, y_test, feature_cols, current_rrp_test, #steps_per_day,
    ) = build_feature_matrices(train_df, test_df, horizon_minutes=forecast_horizon_minutes)

    n_train = X_all_train.shape[0]
    val_start = int(n_train * 0.8)
    X_train, X_val = X_all_train[:val_start], X_all_train[val_start:]
    y_train, y_val = y_all_train[:val_start], y_all_train[val_start:]

    print(f"Horizon: {forecast_horizon_minutes} minutes ahead")
    print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Validation:    {X_val.shape[0]} samples")
    print(f"Test data:     {X_test.shape[0]} samples, {X_test.shape[1]} features")

    booster = fit_xgboost_regressor(X_train, y_train, X_val, y_val, feature_cols=feature_cols)

    feature_names = _get_feature_names_for_matrix(X_test, feature_cols=feature_cols)
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)
    y_pred = booster.predict(dtest)
    evaluate_point_forecast(y_test, y_pred, y_current=current_rrp_test, label="XGBoost")


if __name__ == "__main__":
    main()

# Key evaluation metrics:
# - MAE: 29.65
# - RMSE: 272.43
# - sMAPE: 55.76%
# - Directional accuracy on returns > 1 AUD/MWh: 64.77%
#
# For battery arbitrage, directional accuracy is more significant than raw error,
# since profitability depends on correctly predicting price movements rather than
# exact prices. A directional accuracy of 64.77% indicates outperformance of a random
# model (>50%), but this metric does not account for the magnitude of gains or losses.
#
# As a result, directional accuracy alone is insufficient to infer profitability;
# a trading  simulation is required to translate forecast performance
# into expected arbitrage returns.
