import json
import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from config.feature_config import FEATURE_MATRIX_PATH
from config.model_config import (
    CATEGORICAL_FEATURES,
    EARLY_STOPPING_ROUNDS,
    HOLDOUT_DATE,
    LIGHTGBM_PARAMS,
    MAX_BOOST_ROUNDS,
    MODEL_PATH,
    MODELS_DIR,
    TRAIN_TEST_SPLIT_DATE,
    TUNED_PARAMS_PATH,
    VALIDATION_FRACTION,
)
from src.models.evaluation import (
    apply_calibrator,
    backtest_betting_strategy,
    compute_bucket_hit_rates,
    compute_daily_precision_recall_at_k,
    compute_multi_threshold_report,
    compute_probability_distribution,
    evaluate_game_level_composition,
    evaluate_model,
    fit_calibrator,
    save_evaluation_report,
)
from src.models.walk_forward_cv import (
    WalkForwardResult,
    aggregate_cv_metrics,
    generate_cv_folds,
)

logger = logging.getLogger(__name__)

METADATA_COLUMNS: list[str] = [
    "batter", "game_pk", "game_date", "pitcher", "home_team", "away_team",
]

TARGET_COLUMN: str = "is_hr"

ENGINEERED_PREFIXES: list[str] = [
    "batter_hr_rate_", "batter_barrel_rate_", "batter_avg_exit_velo_",
    "batter_avg_launch_angle_", "batter_k_rate_", "batter_bb_rate_",
    "batter_batting_avg_", "batter_fly_ball_rate_", "batter_pull_rate_",
    "batter_hard_hit_rate_", "batter_sweet_spot_", "batter_avg_xslg_",
    "batter_avg_xwoba_", "batter_hr_rate_vs_", "batter_hr_streak_",
    "pitcher_hr_allowed_rate_", "pitcher_barrel_rate_against_",
    "pitcher_avg_exit_velo_against_", "pitcher_k_rate_",
    "pitcher_bb_rate_", "pitcher_whip_proxy_",
    "pitcher_gb_rate_", "pitcher_fb_rate_",
    "pitcher_fastball_pct_", "pitcher_breaking_pct_",
    "pitcher_offspeed_pct_",
    "games_since_last_hr",
]

FEATURE_COLUMNS: list[str] = [
    "inning", "outs_when_up", "score_diff", "is_home",
    "runners_on_base", "pa_number_in_game",
    "month", "day_of_week",
    "stand", "p_throws", "platoon", "platoon_advantage",
    "park_hr_factor", "park_hr_factor_handedness",
    "elevation_ft", "roof_type",
    "temp_f", "wind_speed_mph", "wind_dir_deg", "humidity_pct",
    "wind_out_to_cf", "air_density_index",
    "batting_order_pos", "batting_order_avg",
    "team_runs_per_game", "expected_pas",
    "batter_days_since_prev_game", "pitcher_days_since_prev_game",
]


def load_feature_matrix() -> pd.DataFrame:
    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    hr_rate = df[TARGET_COLUMN].mean()
    logger.info(
        "Loaded feature matrix: %d rows x %d columns | "
        "dates %s to %s | HR rate: %.4f",
        df.shape[0], df.shape[1],
        df["game_date"].min().date(), df["game_date"].max().date(),
        hr_rate,
    )
    return df


def time_based_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_date = pd.Timestamp(TRAIN_TEST_SPLIT_DATE)

    train_full = df[df["game_date"] < split_date].copy()
    test = df[df["game_date"] >= split_date].copy()

    train_full = train_full.sort_values("game_date").reset_index(drop=True)
    val_size = int(len(train_full) * VALIDATION_FRACTION)
    train = train_full.iloc[:-val_size].copy()
    val = train_full.iloc[-val_size:].copy()

    logger.info(
        f"Train: {len(train):,} rows | "
        f"{train['game_date'].min().date()} to {train['game_date'].max().date()}"
    )
    logger.info(
        f"Val:   {len(val):,} rows | "
        f"{val['game_date'].min().date()} to {val['game_date'].max().date()}"
    )
    logger.info(
        f"Test:  {len(test):,} rows | "
        f"{test['game_date'].min().date()} to {test['game_date'].max().date()}"
    )
    return train, val, test


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_cols: list[str] = []
    for col in df.columns:
        if col in FEATURE_COLUMNS:
            feature_cols.append(col)
        elif any(col.startswith(p) for p in ENGINEERED_PREFIXES):
            feature_cols.append(col)
    return feature_cols


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = _select_feature_columns(df)
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy()

    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")

    return X, y


def compute_scale_pos_weight(y: pd.Series) -> float:
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    weight = n_neg / n_pos
    logger.info(
        f"Class balance: {n_neg:,} neg / {n_pos:,} pos | scale_pos_weight={weight:.2f}"
    )
    return weight


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    categorical_features: list[str] | None = None,
    params_override: dict | None = None,
) -> lgb.Booster:
    if categorical_features is None:
        categorical_features = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    train_ds = lgb.Dataset(
        X_train, label=y_train, categorical_feature=categorical_features,
    )
    val_ds = lgb.Dataset(
        X_val, label=y_val, categorical_feature=categorical_features, reference=train_ds,
    )

    params = {**LIGHTGBM_PARAMS}
    if params_override:
        params.update(params_override)
    params["scale_pos_weight"] = compute_scale_pos_weight(y_train)

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS),
        lgb.log_evaluation(100),
    ]

    model = lgb.train(
        params,
        train_ds,
        num_boost_round=MAX_BOOST_ROUNDS,
        valid_sets=[train_ds, val_ds],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    logger.info(
        f"Training complete | best iteration: {model.best_iteration} | "
        f"best val score: {model.best_score['val'][LIGHTGBM_PARAMS['metric']]:.6f}"
    )
    return model


def save_model(model: lgb.Booster, path: Path | None = None) -> None:
    path = path or MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def load_model(path: Path | None = None) -> lgb.Booster:
    path = path or MODEL_PATH
    return joblib.load(path)


def load_tuned_params() -> dict | None:
    if not TUNED_PARAMS_PATH.exists():
        logger.info("No tuned params found at %s — using defaults", TUNED_PARAMS_PATH)
        return None
    with open(TUNED_PARAMS_PATH) as f:
        params = json.load(f)
    logger.info("Loaded tuned params from %s: %s", TUNED_PARAMS_PATH, params)
    return params


def run_walk_forward_cv(
    df: pd.DataFrame,
    params_override: dict | None = None,
    max_folds: int | None = None,
) -> WalkForwardResult:
    holdout_ts = pd.Timestamp(HOLDOUT_DATE)
    pre_holdout = df[df["game_date"] < holdout_ts].copy()

    folds = generate_cv_folds(pre_holdout["game_date"])
    if max_folds is not None and len(folds) > max_folds:
        folds = folds[-max_folds:]
    if not folds:
        logger.warning("No walk-forward CV folds generated — skipping CV")
        return WalkForwardResult(
            fold_metrics=[], oof_indices=np.array([]),
            oof_predictions=np.array([]), oof_actuals=np.array([]),
            summary={"n_folds": 0},
        )

    all_fold_metrics: list[dict] = []
    all_oof_indices: list[np.ndarray] = []
    all_oof_predictions: list[np.ndarray] = []
    all_oof_actuals: list[np.ndarray] = []

    for fold in folds:
        fold_train = pre_holdout[pre_holdout["game_date"] < fold.train_end]
        fold_val = pre_holdout[
            (pre_holdout["game_date"] >= fold.val_start)
            & (pre_holdout["game_date"] < fold.val_end)
        ]

        if len(fold_val) == 0:
            continue

        fold_train = fold_train.sort_values("game_date").reset_index(drop=True)
        es_size = max(int(len(fold_train) * VALIDATION_FRACTION), 1)
        es_train = fold_train.iloc[:-es_size]
        es_val = fold_train.iloc[-es_size:]

        X_train, y_train = prepare_features(es_train)
        X_es_val, y_es_val = prepare_features(es_val)
        X_val, y_val = prepare_features(fold_val)

        logger.info(
            "Fold %d: train=%d (es_val=%d) | val=%d | %s to %s",
            fold.fold_idx, len(es_train), len(es_val), len(fold_val),
            fold.val_start.date(), fold.val_end.date(),
        )

        model = train_lightgbm(X_train, y_train, X_es_val, y_es_val, params_override=params_override)
        y_pred = model.predict(X_val)

        fold_metrics = evaluate_model(y_val.values, y_pred)
        all_fold_metrics.append(fold_metrics)
        all_oof_indices.append(fold_val.index.values)
        all_oof_predictions.append(y_pred)
        all_oof_actuals.append(y_val.values)

    summary = aggregate_cv_metrics(all_fold_metrics)

    logger.info(
        "Walk-forward CV complete — %d folds | PR-AUC: %.4f ± %.4f | ROC-AUC: %.4f ± %.4f",
        summary["n_folds"],
        summary.get("pr_auc_mean", 0), summary.get("pr_auc_std", 0),
        summary.get("roc_auc_mean", 0), summary.get("roc_auc_std", 0),
    )

    return WalkForwardResult(
        fold_metrics=all_fold_metrics,
        oof_indices=np.concatenate(all_oof_indices),
        oof_predictions=np.concatenate(all_oof_predictions),
        oof_actuals=np.concatenate(all_oof_actuals),
        summary=summary,
    )


def main() -> None:
    logger.info("Starting model training pipeline")

    df = load_feature_matrix()

    tuned_params = load_tuned_params()

    cv_result = run_walk_forward_cv(df, params_override=tuned_params)

    train, val, test = time_based_split(df)

    X_train, y_train = prepare_features(train)
    X_val, y_val = prepare_features(val)
    X_test, y_test = prepare_features(test)

    logger.info("Feature count: %d", X_train.shape[1])

    model = train_lightgbm(X_train, y_train, X_val, y_val, params_override=tuned_params)

    y_pred_proba = model.predict(X_test)

    if len(cv_result.oof_actuals) > 0:
        calibrator = fit_calibrator(cv_result.oof_actuals, cv_result.oof_predictions)
        logger.info(
            "Calibrator fitted on %d pooled OOF predictions from %d CV folds",
            len(cv_result.oof_actuals), cv_result.summary["n_folds"],
        )
    else:
        y_val_pred = model.predict(X_val)
        calibrator = fit_calibrator(y_val.values, y_val_pred)

    y_pred_calibrated = apply_calibrator(calibrator, y_pred_proba)

    pa_metrics = evaluate_model(y_test, y_pred_proba)
    logger.info("PA-level test metrics (raw): %s", pa_metrics)

    pa_metrics_calibrated = evaluate_model(y_test, y_pred_calibrated)
    logger.info("PA-level test metrics (calibrated): %s", pa_metrics_calibrated)

    game_metrics = evaluate_game_level_composition(test, y_pred_proba)
    logger.info("Game-level test metrics (raw): %s", game_metrics)

    game_metrics_calibrated = evaluate_game_level_composition(test, y_pred_calibrated)
    logger.info("Game-level test metrics (calibrated): %s", game_metrics_calibrated)

    prob_dist_raw = compute_probability_distribution(y_pred_proba)
    prob_dist_calibrated = compute_probability_distribution(y_pred_calibrated)
    multi_threshold_df = compute_multi_threshold_report(y_test.values, y_pred_calibrated)

    game_dates_test = test["game_date"].values

    daily_pk = compute_daily_precision_recall_at_k(
        y_test.values, y_pred_calibrated, game_dates_test,
    )
    logger.info("Daily precision/recall at K: %s", daily_pk)

    bucket_df = compute_bucket_hit_rates(y_test.values, y_pred_calibrated)

    backtest = backtest_betting_strategy(y_test.values, y_pred_calibrated)
    logger.info("Backtest results: %s", backtest)

    betting_metrics = {**daily_pk, **backtest}

    importances = model.feature_importance(importance_type="gain")
    feature_names = model.feature_name()
    save_evaluation_report(
        pa_metrics, feature_names, importances, y_test.values, y_pred_proba,
        game_metrics=game_metrics,
        calibrated_proba=y_pred_calibrated,
        calibrated_metrics=pa_metrics_calibrated,
        game_metrics_calibrated=game_metrics_calibrated,
        betting_metrics=betting_metrics,
        bucket_df=bucket_df,
        game_dates=game_dates_test,
        cv_summary=cv_result.summary,
        cv_fold_metrics=cv_result.fold_metrics,
        multi_threshold_df=multi_threshold_df,
        prob_dist_raw=prob_dist_raw,
        prob_dist_calibrated=prob_dist_calibrated,
    )

    save_model(model)
    joblib.dump(calibrator, MODELS_DIR / "calibrator.joblib")
    logger.info("Calibrator saved to %s", MODELS_DIR / "calibrator.joblib")

    logger.info("Training pipeline complete")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
