import json
import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from config.feature_config import STARTER_GAME_MATRIX_PATH
from config.model_config import (
    HOLDOUT_DATE,
    STARTER_K_CATEGORICAL_FEATURES,
    STARTER_K_EARLY_STOPPING_ROUNDS,
    STARTER_K_EVALUATION_DIR,
    STARTER_K_LGBM_PARAMS,
    STARTER_K_MAX_BOOST_ROUNDS,
    STARTER_K_MODEL_PATH,
    STARTER_K_MODELS_DIR,
    STARTER_K_OVER_UNDER_LINES,
    TOP_N_FEATURES,
    TRAIN_TEST_SPLIT_DATE,
    VALIDATION_FRACTION,
)
from src.models.starter_k_evaluation import (
    compute_baselines,
    evaluate_k_count,
    evaluate_over_under,
    fit_dispersion,
    save_starter_k_report,
)
from src.models.walk_forward_cv import (
    WalkForwardResult,
    aggregate_cv_metrics,
    generate_cv_folds,
)

logger = logging.getLogger(__name__)

TARGET_COLUMN: str = "n_k_against"

METADATA_COLUMNS: list[str] = [
    "pitcher", "game_pk", "game_date", "home_team", "away_team", "opposing_team",
]

ENGINEERED_PREFIXES: list[str] = [
    "pitcher_k_rate_", "pitcher_bb_rate_", "pitcher_whip_proxy_",
    "pitcher_hr_allowed_rate_", "pitcher_barrel_rate_against_",
    "pitcher_avg_exit_velo_against_",
    "pitcher_swstr_rate_", "pitcher_chase_rate_induced_",
    "pitcher_zone_rate_", "pitcher_avg_fastball_velo_",
    "pitcher_fastball_pct_", "pitcher_breaking_pct_", "pitcher_offspeed_pct_",
    "pitcher_gb_rate_", "pitcher_fb_rate_",
    "pitcher_bf_count_",
    "pitcher_avg_bf_", "pitcher_avg_pitch_count_", "pitcher_pct_6plus_innings_",
    "opp_team_k_rate_", "opp_team_whiff_rate_", "opp_team_chase_rate_",
    "ix_",
]

STATIC_FEATURES: list[str] = [
    "park_k_factor", "park_k_factor_handedness",
    "elevation_ft", "roof_type",
    "temp_f", "wind_speed_mph", "humidity_pct", "air_density_index",
    "month", "day_of_week", "is_home", "p_throws", "opp_lineup_pct_lhb",
    "pitcher_days_since_prev_game",
]


def load_starter_game_matrix() -> pd.DataFrame:
    df = pd.read_parquet(STARTER_GAME_MATRIX_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    target_mean = df[TARGET_COLUMN].mean()
    logger.info(
        "Loaded starter game matrix: %d rows x %d columns | "
        "dates %s to %s | %s mean: %.2f",
        df.shape[0], df.shape[1],
        df["game_date"].min().date(), df["game_date"].max().date(),
        TARGET_COLUMN, target_mean,
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


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_cols: list[str] = []
    for col in df.columns:
        if col in STATIC_FEATURES:
            feature_cols.append(col)
        elif any(col.startswith(p) for p in ENGINEERED_PREFIXES):
            feature_cols.append(col)
    return feature_cols


def prepare_features(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy()

    for col in STARTER_K_CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")

    return X, y


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params_override: dict | None = None,
) -> lgb.Booster:
    categorical_features = [
        c for c in STARTER_K_CATEGORICAL_FEATURES if c in X_train.columns
    ]

    train_ds = lgb.Dataset(
        X_train, label=y_train, categorical_feature=categorical_features,
    )
    val_ds = lgb.Dataset(
        X_val, label=y_val, categorical_feature=categorical_features,
        reference=train_ds,
    )

    params = {**STARTER_K_LGBM_PARAMS}
    if params_override:
        params.update(params_override)

    callbacks = [
        lgb.early_stopping(STARTER_K_EARLY_STOPPING_ROUNDS),
        lgb.log_evaluation(100),
    ]

    model = lgb.train(
        params,
        train_ds,
        num_boost_round=STARTER_K_MAX_BOOST_ROUNDS,
        valid_sets=[train_ds, val_ds],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    metric_name = params.get("metric", "poisson")
    logger.info(
        f"Training complete | best iteration: {model.best_iteration} | "
        f"best val score: {model.best_score['val'][metric_name]:.6f}"
    )
    return model


def save_model(model: lgb.Booster, path: Path | None = None) -> None:
    path = path or STARTER_K_MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def load_model(path: Path | None = None) -> lgb.Booster:
    path = path or STARTER_K_MODEL_PATH
    return joblib.load(path)


def run_walk_forward_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
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

        X_train, y_train = prepare_features(es_train, feature_cols)
        X_es_val, y_es_val = prepare_features(es_val, feature_cols)
        X_val, y_val = prepare_features(fold_val, feature_cols)

        logger.info(
            "Fold %d: train=%d (es_val=%d) | val=%d | %s to %s",
            fold.fold_idx, len(es_train), len(es_val), len(fold_val),
            fold.val_start.date(), fold.val_end.date(),
        )

        model = train_lightgbm(
            X_train, y_train, X_es_val, y_es_val,
            params_override=params_override,
        )
        lambda_pred = model.predict(X_val)

        fold_metrics = evaluate_k_count(y_val.values, lambda_pred)
        all_fold_metrics.append(fold_metrics)
        all_oof_indices.append(fold_val.index.values)
        all_oof_predictions.append(lambda_pred)
        all_oof_actuals.append(y_val.values)

    summary = aggregate_cv_metrics(all_fold_metrics)

    logger.info(
        "Walk-forward CV complete — %d folds | MAE: %.4f +/- %.4f | RMSE: %.4f +/- %.4f",
        summary["n_folds"],
        summary.get("mae_mean", 0), summary.get("mae_std", 0),
        summary.get("rmse_mean", 0), summary.get("rmse_std", 0),
    )

    return WalkForwardResult(
        fold_metrics=all_fold_metrics,
        oof_indices=np.concatenate(all_oof_indices) if all_oof_indices else np.array([]),
        oof_predictions=np.concatenate(all_oof_predictions) if all_oof_predictions else np.array([]),
        oof_actuals=np.concatenate(all_oof_actuals) if all_oof_actuals else np.array([]),
        summary=summary,
    )


def main() -> None:
    logger.info("Starting starter-K model training pipeline")

    df = load_starter_game_matrix()

    feature_cols = select_feature_columns(df)
    logger.info("Selected %d feature columns", len(feature_cols))

    cv_result = run_walk_forward_cv(df, feature_cols)

    train, val, test = time_based_split(df)

    X_train, y_train = prepare_features(train, feature_cols)
    X_val, y_val = prepare_features(val, feature_cols)
    X_test, y_test = prepare_features(test, feature_cols)

    logger.info("Feature count: %d", X_train.shape[1])

    model = train_lightgbm(X_train, y_train, X_val, y_val)

    lambda_pred = model.predict(X_test)

    count_metrics = evaluate_k_count(y_test.values, lambda_pred)
    logger.info("Test count metrics: %s", count_metrics)

    val_lambda = model.predict(X_val)
    dispersion = fit_dispersion(y_val.values, val_lambda)
    logger.info("Fitted dispersion on validation set: %.4f", dispersion)

    ou_metrics = evaluate_over_under(
        y_test.values, lambda_pred, STARTER_K_OVER_UNDER_LINES, dispersion,
    )
    logger.info("Over/under metrics: %s", ou_metrics)

    baselines = compute_baselines(test)
    logger.info("Baseline metrics: %s", baselines)

    importances = model.feature_importance(importance_type="gain")
    feature_names = model.feature_name()

    save_starter_k_report(
        count_metrics=count_metrics,
        ou_metrics_df=ou_metrics,
        baselines=baselines,
        feature_names=feature_names,
        importances=importances,
        y_true=y_test.values,
        lambda_pred=lambda_pred,
        dispersion=dispersion,
        output_dir=STARTER_K_EVALUATION_DIR,
    )

    save_model(model)

    dispersion_path = STARTER_K_MODELS_DIR / "dispersion.json"
    dispersion_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dispersion_path, "w") as f:
        json.dump({"dispersion": dispersion}, f, indent=2)
    logger.info("Dispersion saved to %s", dispersion_path)

    logger.info("Starter-K training pipeline complete")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
