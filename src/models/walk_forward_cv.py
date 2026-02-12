import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.model_config import (
    CV_MIN_TRAIN_DAYS,
    CV_STEP_DAYS,
    CV_VAL_WINDOW_DAYS,
    HOLDOUT_DATE,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CVFold:
    fold_idx: int
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


@dataclass
class WalkForwardResult:
    fold_metrics: list[dict]
    oof_indices: np.ndarray
    oof_predictions: np.ndarray
    oof_actuals: np.ndarray
    summary: dict


def generate_cv_folds(
    game_dates: pd.Series,
    holdout_date: str = HOLDOUT_DATE,
    val_window_days: int = CV_VAL_WINDOW_DAYS,
    step_days: int = CV_STEP_DAYS,
    min_train_days: int = CV_MIN_TRAIN_DAYS,
) -> list[CVFold]:
    holdout_ts = pd.Timestamp(holdout_date)
    data_start = game_dates.min()
    first_val_start = data_start + pd.Timedelta(days=min_train_days)

    folds: list[CVFold] = []
    i = 0
    while True:
        val_start = first_val_start + pd.Timedelta(days=i * step_days)
        val_end = val_start + pd.Timedelta(days=val_window_days)

        if val_end > holdout_ts:
            break

        train_end = val_start
        folds.append(CVFold(
            fold_idx=len(folds),
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
        ))
        i += 1

    logger.info(
        "Generated %d walk-forward CV folds | "
        "data_start=%s | first_val_start=%s | holdout=%s",
        len(folds),
        data_start.date(),
        first_val_start.date(),
        holdout_ts.date(),
    )
    return folds


def aggregate_cv_metrics(fold_metrics: list[dict]) -> dict:
    if not fold_metrics:
        return {"n_folds": 0}

    numeric_keys: list[str] = []
    for key, value in fold_metrics[0].items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric_keys.append(key)

    summary: dict = {"n_folds": len(fold_metrics)}

    for key in numeric_keys:
        values = [m[key] for m in fold_metrics if key in m]
        if not values:
            continue
        arr = np.array(values, dtype=float)
        summary[f"{key}_mean"] = float(np.mean(arr))
        summary[f"{key}_std"] = float(np.std(arr))
        summary[f"{key}_min"] = float(np.min(arr))
        summary[f"{key}_max"] = float(np.max(arr))

    return summary
