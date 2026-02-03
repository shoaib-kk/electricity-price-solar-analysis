import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_pinball_loss
import Metrics_utils
import Point_forecast
import logging 
from logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def load_cleaned_datasets():
    train_path = "CLEANED_PRICE_AND_DEMAND_VIC1_TRAIN.csv"
    test_path = "CLEANED_PRICE_AND_DEMAND_VIC1_TEST.csv"

    train = pd.read_csv(train_path, parse_dates=["SETTLEMENTDATE"])
    test = pd.read_csv(test_path, parse_dates=["SETTLEMENTDATE"])

    return train, test


def validate_features(X: pd.DataFrame) -> pd.DataFrame:
    """Validate and filter the feature matrix for quantile regression.

    - Keeps only numeric columns.
    - Drops columns that are entirely NaN.
    - Leaves remaining NaNs as-is light GBM can handle them.
    Returns a cleaned feature DataFrame.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    # Keep numeric columns only.
    X_num = X.select_dtypes(include=[np.number])

    # Replace inf / -inf with NaN so LightGBM sees them as missing
    # rather than exploding gradients
    X_num = X_num.replace([np.inf, -np.inf], np.nan)

    # Drop columns that are completely missing.
    before_cols = X_num.shape[1]
    X_num = X_num.dropna(axis=1, how="all")
    after_cols = X_num.shape[1]
    if after_cols < before_cols:
        logging.info(f"validate_features: dropped {before_cols - after_cols} all-NaN feature columns.")

    return X_num

def build_quantile_model(quantile: float):
    """Build LightGBM model for quantile regression."""
    params = {
        "boosting_type": "gbdt",
        "n_estimators": 5000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "objective": "quantile",
        "alpha": quantile,
        "metric": "quantile",
        "verbosity": -1,
        "random_state": 42,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
   }
    model = lgb.LGBMRegressor(**params)
    return model


def train_quantile_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    quantiles: list[float] = [0.05, 0.5, 0.95],
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    early_stopping_rounds: int = 50,
):
    """Train LightGBM quantile regression models for specified quantiles.
    If X_val / y_val are provided, use a later-in-time validation split
    with early stopping based on quantile loss
    """

    models: dict[float, lgb.LGBMRegressor] = {}

    use_validation = False

    if X_val is not None and y_val is not None and len(X_val) > 0:
        use_validation = True

    callbacks = []
    if use_validation and early_stopping_rounds is not None:
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

    for quantile in quantiles:
        model = build_quantile_model(quantile)

        if use_validation:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="quantile",
                callbacks=callbacks,
            )
        else:
            model.fit(X_train, y_train)

        models[quantile] = model

    return models

def predict_quantiles(models: dict[float, lgb.LGBMRegressor], X_test: pd.DataFrame):
    """Predict quantiles using trained model."""
    y_pred = {}
    for quantile, model in models.items():
        y_pred[quantile] = model.predict(X_test)
    return y_pred


def compute_split_conformal_interval(
    models: dict[float, lgb.LGBMRegressor],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    y_pred_quantiles_raw: pd.DataFrame,
    alpha_low: float = 0.05,
    alpha_high: float = 0.95,
    target_coverage: float = 0.90,
) -> tuple[pd.Series, pd.Series, float]:
    
    """Compute (upper - lower)% intervals based on lower%/upper% quantile models.
                (     90     )% for (5%, 95%) quantiles.

    Uses the validation set to build nonconformity scores
    s_i = max(q_low(x_i) - y_i, y_i - q_high(x_i))
    
    The calibration offset Q is defined as the (1 - alpha)-quantile of these scores,
    where alpha = 1 - target_coverage.

    Final conformal band on test is:
        [q_low_test - Q, q_high_test + Q].

    Returns (lower_conf, upper_conf, Q).
    """

    if alpha_low not in models or alpha_high not in models:
        raise KeyError("Both lower and upper quantile models must be available.")

    # Predictions on validation set
    q_low_val = pd.Series(models[alpha_low].predict(X_val), index=y_val.index)
    q_high_val = pd.Series(models[alpha_high].predict(X_val), index=y_val.index)

    nonconformity = np.maximum(q_low_val - y_val, y_val - q_high_val)
    nonconformity = nonconformity.to_numpy()

    n_cal = nonconformity.shape[0]
    if n_cal == 0:
        raise ValueError("Validation set is empty; cannot compute conformal intervals.")

    alpha = 1.0 - target_coverage
    q_level = 1.0 - alpha
    Q = float(np.quantile(nonconformity, q_level))

    # Base quantile predictions on test
    q_low_test = y_pred_quantiles_raw[alpha_low]
    q_high_test = y_pred_quantiles_raw[alpha_high]

    lower_conf = q_low_test - Q
    upper_conf = q_high_test + Q

    return lower_conf, upper_conf, Q

def summarise_uncertainty_intervals(y_pred_quantiles: pd.DataFrame, quantiles: list[float] = [0.05, 0.5, 0.95]) :
    """Summarise uncertainty intervals from quantile predictions."""
    cols = y_pred_quantiles.columns

    if len(quantiles) != 3:
        raise ValueError("summarise_uncertainty_intervals requires exactly three quantiles.")
    for quantile in quantiles:
        if quantile not in cols:
            raise KeyError(f"y_pred_quantiles must have column {quantile}.")

    lower_bound = y_pred_quantiles[quantiles[0]]
    median = y_pred_quantiles[quantiles[1]]
    upper_bound = y_pred_quantiles[quantiles[2]]

    summary = pd.DataFrame(
        {
            "lower_bound": lower_bound,
            "median": median,
            "upper_bound": upper_bound,
        }
    )
    return summary

def manage_train_test_split(forecast_horizon_minutes: int = 30):
    # Load cleaned train/test datasets
    train_df, test_df = load_cleaned_datasets()

    (
        X_all_train,
        y_all_train,
        X_test,
        y_test,
        feature_cols,
        current_rrp_test,
    ) = Point_forecast.build_feature_matrices(
        train_df,
        test_df,
        horizon_minutes=forecast_horizon_minutes,
    )

    # Wrap arrays back into DataFrames/Series for LightGBM
    X_all_df = pd.DataFrame(X_all_train, columns=feature_cols)
    X_test_df = pd.DataFrame(X_test, columns=feature_cols)
    y_all = pd.Series(y_all_train, name="RRP_target")
    y_true_test = pd.Series(y_test, name="RRP_target")

    n_train_total = X_all_df.shape[0]
    val_start = int(n_train_total * 0.8)

    # Time-based split: earlier data for training, later slice for validation
    X_train_df = X_all_df.iloc[:val_start].copy()
    X_val_df = X_all_df.iloc[val_start:].copy()
    y_train = y_all.iloc[:val_start].copy()
    y_val = y_all.iloc[val_start:].copy()

    # Validate and clean features using train only 
    X_train_df = validate_features(X_train_df)

    X_test_df = X_test_df[X_train_df.columns]
    X_val_df = X_val_df[X_train_df.columns]

    return X_train_df, y_train, X_val_df, y_val, X_test_df, y_true_test, current_rrp_test

def metrics_raw(y_pred_quantiles_raw, y_true_test, quantiles=[0.05, 0.5, 0.95]):
    
    # Check quantile crossing before any repair -> 0.05 < 0.5 < 0.95 ideally but sometimes
    # quantiles cross due to estimation errors.
    lower_q_raw = y_pred_quantiles_raw[quantiles[0]].values
    middle_q_raw = y_pred_quantiles_raw[quantiles[1]].values
    upper_q_raw = y_pred_quantiles_raw[quantiles[2]].values
    cross_lower_mid = float(np.mean(lower_q_raw > middle_q_raw))
    cross_mid_upper = float(np.mean(middle_q_raw > upper_q_raw))

    logging.info(
        "\nQuantile crossing rate (raw, before repair): "
        f"{quantiles[0]}> {quantiles[1]} = {cross_lower_mid * 100:.2f}%, "
        f"{quantiles[1]}> {quantiles[2]} = {cross_mid_upper * 100:.2f}%"
    )
    # Metrics for raw (unrepaired) quantiles
    logging.info("\nRAW quantile pinball losses and marginal coverage on TEST set (no repair):")
    for q in quantiles:
        loss_raw = mean_pinball_loss(y_true_test, y_pred_quantiles_raw[q], alpha=q)
        coverage_raw = float(np.mean(y_true_test <= y_pred_quantiles_raw[q]))
        logging.info(
            f"  q={q:.2f}: pinball loss = {loss_raw:.4f}, "
            f"coverage = {coverage_raw * 100:.1f}% (share of y_true <= q̂)"
        )

    lower_q_raw = y_pred_quantiles_raw[quantiles[0]]
    upper_q_raw = y_pred_quantiles_raw[quantiles[2]]

    raw_interval_coverage = float(
        np.mean((y_true_test >= lower_q_raw) & (y_true_test <= upper_q_raw))
    )
    mean_width_raw = float(np.mean(upper_q_raw - lower_q_raw))
    logging.info(
        f"\nRAW interval coverage for [{quantiles[0]}, {quantiles[2]}]: {raw_interval_coverage * 100:.2f}% "
        f"(target ≈ {(quantiles[2] - quantiles[0]) * 100:.0f}%)"
    )
    logging.info(f"RAW mean interval width [{quantiles[0]}, {quantiles[2]}]: {mean_width_raw:.2f} AUD/MWh")

    # Quick MAE for the median (q=0.5) quantile as a point forecast for a point of comparison
    if 0.5 in y_pred_quantiles_raw.columns:
        mae_median = float(
            np.mean(np.abs(y_true_test.to_numpy() - y_pred_quantiles_raw[0.5].to_numpy()))
        )
        print(f"Median quantile (q=0.5) MAE: {mae_median:.2f} AUD/MWh")


def directional_performance(y_pred_quantiles_raw, current_rrp_test, y_true_test):
    """ Directional performance for the median (0.5) quantile. Use the RAW median
     to preserve its interpretation as the model's median forecast.
     basically was the predicted change in price the right sign up or down """
    
    if 0.5 in y_pred_quantiles_raw.columns:
        print("\nDirectional accuracy for median quantile (0.5), RAW predictions:")
        Metrics_utils.evaluate_direction_accuracy(
            y_true=y_true_test.to_numpy(),
            y_pred=y_pred_quantiles_raw[0.5].to_numpy(),
            y_current=current_rrp_test,
            label="LightGBM quantile (0.5, raw)",
        )


def run_conformal_calibration(
    models: dict[float, lgb.LGBMRegressor],
    X_val_df: pd.DataFrame,
    y_val: pd.Series,
    y_true_test: pd.Series,
    y_pred_quantiles_raw: pd.DataFrame,
    alpha_low: float = 0.05,
    alpha_high: float = 0.95,
    target_coverage: float = 0.90,
    quantiles: list[float] = [0.05, 0.5, 0.95],
) -> None:
    """Compute and report split-conformal intervals using a helper.
    """

    lower_conf, upper_conf, Q = compute_split_conformal_interval(models, X_val_df, y_val,y_pred_quantiles_raw,
                                                                 alpha_low=alpha_low, alpha_high=alpha_high,
                                                                 target_coverage=target_coverage,
    )

    interval_covered_conf = float(
        np.mean((y_true_test >= lower_conf) & (y_true_test <= upper_conf))
    )
    mean_width_conf = float(np.mean(upper_conf - lower_conf))

    print("\nSplit-conformal 90% interval calibration (based on validation set):")
    print(f"  Nonconformity quantile Q (target {target_coverage * 100:.0f}%): {Q:.4f}")
    print(
        f"  CONFORMAL interval coverage [{quantiles[0]}, {quantiles[2]}]: {interval_covered_conf * 100:.2f}% "
        f"(target {target_coverage * 100:.0f}%)"
    )
    print(
        f"  CONFORMAL mean interval width [{quantiles[0]}, {quantiles[2]}]: {mean_width_conf:.2f} AUD/MWh"
    )

    conformal_summary = pd.DataFrame(
        {
            "lower_conformal": lower_conf,
            "median": y_pred_quantiles_raw[0.5],
            "upper_conformal": upper_conf,
        }
    )
    print("\nFirst few split-conformal intervals (lower/median/upper):")
    print(conformal_summary.head())



def main():
    """End-to-end LightGBM quantile regression pipeline.

    Reuses the same engineered features, h-step-ahead target (RRP_target),
    and cleaning / NaN handling as the XGBoost point-forecast pipeline.
    """

    forecast_horizon_minutes = 30

    X_train_df, y_train, X_val_df, y_val, X_test_df, y_true_test, current_rrp_test = manage_train_test_split(forecast_horizon_minutes=forecast_horizon_minutes)
    quantiles = [0.05, 0.5, 0.95]

    print(
        f"Training LightGBM quantile models for horizon {forecast_horizon_minutes} minutes... "
        f"(train={len(y_train)}, val={len(y_val)})"
    )

    models = train_quantile_model(
        X_train_df,
        y_train,
        quantiles=quantiles,
        X_val=X_val_df,
        y_val=y_val,
        early_stopping_rounds=50,
    )

    # Predict quantiles on the test set (raw, no calibration/repair yet)
    y_pred_dict = predict_quantiles(models, X_test_df)
    y_pred_quantiles_raw = pd.DataFrame(y_pred_dict, index=y_true_test.index)
    metrics_raw(y_pred_quantiles_raw, y_true_test, quantiles=quantiles)

    # Split conformal calibration for the interval [lower quantile, upper quantile]
    try:
        run_conformal_calibration(
            models,
            X_val_df,
            y_val,
            y_true_test,
            y_pred_quantiles_raw,
            alpha_low=0.05,
            alpha_high=0.95,
            target_coverage=0.90,
        )
    except Exception as exc:
        logging.warning(f"\nSplit-conformal calibration skipped due to error: {exc}")

    directional_performance(y_pred_quantiles_raw, current_rrp_test, y_true_test)


if __name__ == "__main__":
    main()

