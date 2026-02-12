import logging
from pathlib import Path

import pandas as pd

from config.feature_config import (
    FEATURE_MATRIX_PATH,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    ROLLING_WINDOWS,
)
from src.features.feature_engineering import (
    add_game_context_features,
    add_platoon_features,
    aggregate_to_batter_game,
    build_pa_rows,
    compute_batter_rolling_stats,
    compute_pitcher_rolling_stats,
    merge_rolling_stats_onto_pas,
)

logger = logging.getLogger(__name__)


def load_raw_data() -> pd.DataFrame:
    parquet_files = sorted(RAW_DATA_DIR.glob("statcast_*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No statcast_*.parquet files found in {RAW_DATA_DIR}"
        )

    logger.info(f"Found {len(parquet_files)} raw statcast files in {RAW_DATA_DIR}")

    frames: list[pd.DataFrame] = []
    for path in parquet_files:
        df = pd.read_parquet(path)
        logger.info(f"  Loaded {path.name}: {len(df):,} rows")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["game_date"] = pd.to_datetime(combined["game_date"])

    logger.info(
        f"Total raw data: {len(combined):,} rows, "
        f"date range {combined['game_date'].min().date()} to {combined['game_date'].max().date()}"
    )
    return combined


def build_feature_matrix() -> pd.DataFrame:
    logger.info("Starting feature matrix build")

    raw_data = load_raw_data()

    logger.info("Building PA-level rows with context features")
    pa_df = build_pa_rows(raw_data)
    logger.info(f"PA rows: {len(pa_df):,}")

    logger.info("Aggregating to batter-game level")
    batter_games = aggregate_to_batter_game(raw_data)
    logger.info(f"Batter-game rows: {len(batter_games):,}")

    logger.info(f"Computing batter rolling stats (windows={ROLLING_WINDOWS})")
    batter_games = compute_batter_rolling_stats(batter_games, windows=ROLLING_WINDOWS)

    logger.info(f"Computing pitcher rolling stats (windows={ROLLING_WINDOWS})")
    pitcher_stats = compute_pitcher_rolling_stats(raw_data, windows=ROLLING_WINDOWS)
    logger.info(f"Pitcher rolling stats rows: {len(pitcher_stats):,}")

    logger.info("Merging rolling stats onto PA rows")
    feature_df = merge_rolling_stats_onto_pas(pa_df, batter_games, pitcher_stats)
    logger.info(f"Merged rows: {len(feature_df):,}")

    logger.info("Adding platoon features")
    feature_df = add_platoon_features(feature_df)

    logger.info("Adding game context features")
    feature_df = add_game_context_features(feature_df)

    logger.info(
        f"Feature matrix complete: {feature_df.shape[0]:,} rows, "
        f"{feature_df.shape[1]} columns"
    )
    return feature_df


def main() -> None:
    feature_matrix = build_feature_matrix()

    Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
    feature_matrix.to_parquet(FEATURE_MATRIX_PATH, index=False)
    logger.info(f"Saved feature matrix to {FEATURE_MATRIX_PATH}")

    hr_rate = feature_matrix["is_hr"].mean() if "is_hr" in feature_matrix.columns else float("nan")

    date_min = feature_matrix["game_date"].min().date()
    date_max = feature_matrix["game_date"].max().date()

    logger.info(
        f"Summary: {len(feature_matrix):,} rows | "
        f"HR rate: {hr_rate:.4f} | "
        f"Date range: {date_min} to {date_max} | "
        f"Features: {feature_matrix.shape[1]}"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
