import numpy as np


def compute_mae_rmse_smape(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE and sMAPE for forecast vs truth.

    Returns a dict with keys: 'mae', 'rmse', 'smape'.
    sMAPE is returned in percent units (0-100) or np.nan if undefined.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.abs(y_true) + np.abs(y_pred)
        smape = 2.0 * np.abs(y_true - y_pred) / denom
        smape = smape[np.isfinite(smape)]
        smape_val = float(np.mean(smape)) * 100 if smape.size > 0 else np.nan

    return {"mae": mae, "rmse": rmse, "smape": smape_val}


def compute_cost_weighted_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAE weighted by |y_true| so high-price errors count more."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.abs(y_true)
        errors = np.abs(y_true - y_pred)
        mask = np.isfinite(weights) & np.isfinite(errors) & (weights > 0)
        if not np.any(mask):
            return float("nan")
        return float(np.sum(errors[mask] * weights[mask]) / np.sum(weights[mask]))


def compute_direction_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_current: np.ndarray,
    move_threshold: float = 1.0,
) -> tuple[float, int]:
    """Compute direction accuracy on returns above a move threshold.
    Returns (direction_accuracy_pct, n_eval).
    """

    true_ret = np.asarray(y_true, dtype=float) - np.asarray(y_current, dtype=float)
    pred_ret = np.asarray(y_pred, dtype=float) - np.asarray(y_current, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        true_dir = np.sign(true_ret)
        pred_dir = np.sign(pred_ret)
        mask = (
            np.isfinite(true_dir)
            & np.isfinite(pred_dir)
            & (np.abs(true_ret) >= move_threshold)
        )
        n_eval = int(mask.sum())
        if n_eval <= 0:
            return float("nan"), 0

        direction_acc = float(np.mean(true_dir[mask] == pred_dir[mask]) * 100.0)
        return direction_acc, n_eval


def evaluate_direction_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_current: np.ndarray,
    label: str = "Model",
    move_threshold: float = 1.0,
) -> None:
    """Print direction accuracy on returns above a move threshold.
    Uses compute_direction_accuracy to calculate the metric and
    standardises the printed output across scripts.
    """

    direction_acc, n_eval = compute_direction_accuracy(
        y_true, y_pred, y_current, move_threshold=move_threshold
    )

    if n_eval <= 0 or not np.isfinite(direction_acc):
        print(
            f"  Direction accuracy on returns (> {move_threshold:.1f}) [{label}]: "
            "no cases above threshold"
        )
        return

    print(
        f"  Direction accuracy on returns (> {move_threshold:.1f}) [{label}]: "
        f"{direction_acc:.2f}% over {n_eval} cases"
    )
