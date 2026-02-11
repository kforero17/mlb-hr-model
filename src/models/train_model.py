import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

from config.feature_config import FEATURE_MATRIX_PATH
from config.model_config import (
    CATEGORICAL_FEATURES,
    EARLY_STOPPING_ROUNDS,
    LIGHTGBM_PARAMS,
    MAX_BOOST_ROUNDS,
    MODEL_PATH,
    MODELS_DIR,
    TRAIN_TEST_SPLIT_DATE,
    VALIDATION_FRACTION,
)
from src.models.evaluation import evaluate_model, save_evaluation_report

logger = logging.getLogger(__name__)

METADATA_COLUMNS: list[str] = [
    "batter", "game_pk", "game_date", "pitcher", "home_team", "away_team",
]
TARGET_COLUMN: str = "hit_hr"


def load_feature_matrix() -> pd.DataFrame:
    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    hr_rate = df[TARGET_COLUMN].mean()
    logger.info(
        f"Loaded feature matrix: {df.shape[0]:,} rows x {df.shape[1]} columns | "
        f"dates {df['game_date'].min().date()} to {df['game_date'].max().date()} | "
        f"HR rate: {hr_rate:.4f}"
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


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    exclude = set(METADATA_COLUMNS) | {TARGET_COLUMN}
    feature_cols = [c for c in df.columns if c not in exclude]
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


def main() -> None:
    logger.info("Starting model training pipeline")

    df = load_feature_matrix()
    train, val, test = time_based_split(df)

    X_train, y_train = prepare_features(train)
    X_val, y_val = prepare_features(val)
    X_test, y_test = prepare_features(test)

    logger.info(f"Feature count: {X_train.shape[1]}")

    model = train_lightgbm(X_train, y_train, X_val, y_val)

    y_pred_proba = model.predict(X_test)

    metrics = evaluate_model(y_test, y_pred_proba)
    logger.info(f"Test metrics: {metrics}")

    importances = model.feature_importance(importance_type="gain")
    feature_names = model.feature_name()
    save_evaluation_report(
        metrics, feature_names, importances, y_test.values, y_pred_proba,
    )

    save_model(model)

    logger.info("Training pipeline complete")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
