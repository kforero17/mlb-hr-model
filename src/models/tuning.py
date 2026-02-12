import logging

import optuna
import pandas as pd

from config.model_config import (
    TUNING_BRIER_WEIGHT,
    TUNING_CV_FOLDS,
    TUNING_N_TRIALS,
)
from src.models.train_model import run_walk_forward_cv

logger = logging.getLogger(__name__)


def build_search_space(trial: optuna.Trial) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "feature_fraction_bynode": trial.suggest_float("feature_fraction_bynode", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 500),
        "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 1e-4, 10.0, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 5.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 5.0, log=True),
    }


def _create_objective(
    df: pd.DataFrame,
    n_cv_folds: int,
    brier_weight: float,
) -> callable:
    def objective(trial: optuna.Trial) -> float:
        params = build_search_space(trial)
        cv_result = run_walk_forward_cv(df, params_override=params, max_folds=n_cv_folds)

        if cv_result.summary.get("n_folds", 0) == 0:
            return float("-inf")

        pr_auc_mean = cv_result.summary["pr_auc_mean"]
        brier_score_mean = cv_result.summary["brier_score_mean"]
        score = pr_auc_mean - brier_weight * brier_score_mean

        return score

    return objective


def run_tuning(
    df: pd.DataFrame,
    n_trials: int = TUNING_N_TRIALS,
    n_cv_folds: int = TUNING_CV_FOLDS,
    brier_weight: float = TUNING_BRIER_WEIGHT,
) -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize", study_name="lgbm_hr_tuning")
    objective = _create_objective(df, n_cv_folds, brier_weight)
    study.optimize(objective, n_trials=n_trials)

    logger.info(
        "Tuning complete — best trial: %d | best score: %.6f | best params: %s",
        study.best_trial.number,
        study.best_value,
        study.best_params,
    )

    return study.best_params
