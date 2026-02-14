import logging
from pathlib import Path

import pandas as pd

from config.feature_config import (
    FEATURE_MATRIX_PATH,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    ROLLING_WINDOWS,
    STARTER_GAME_MATRIX_PATH,
)
from src.data.collect_weather import load_weather_data
from src.features.environment_features import (
    add_environment_features,
    compute_park_hr_factors,
    compute_park_k_factors,
)
from src.features.pitch_features import (
    aggregate_batter_pitch_stats,
    aggregate_pitcher_pitch_stats,
    compute_batter_pitch_rolling,
    compute_pitcher_pitch_rolling,
    merge_pitch_features_onto_pas,
)
from src.features.opportunity_features import (
    add_opportunity_features,
    compute_team_rolling_runs,
    derive_batting_order,
)
from src.features.skill_features import (
    aggregate_batter_skill_stats,
    aggregate_pitcher_skill_stats,
    compute_batter_skill_rolling,
    compute_pitcher_skill_rolling,
    merge_skill_features_onto_pas,
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
from src.features.interaction_features import add_interaction_features
from src.features.starter_game_features import build_starter_game_matrix

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

    logger.info("Computing park HR factors")
    park_factors = compute_park_hr_factors(raw_data)

    logger.info("Computing park K factors")
    park_k_factors = compute_park_k_factors(raw_data)

    logger.info("Loading weather data")
    weather = load_weather_data()

    logger.info("Adding environment features")
    feature_df = add_environment_features(feature_df, park_factors, weather, park_k_factors)

    logger.info("Deriving batting order and team run rates")
    batting_order = derive_batting_order(raw_data)
    team_runs = compute_team_rolling_runs(raw_data)

    logger.info("Adding opportunity features")
    feature_df = add_opportunity_features(feature_df, batting_order, team_runs)

    logger.info("Aggregating batter skill stats")
    batter_skill = aggregate_batter_skill_stats(raw_data)
    batter_skill = compute_batter_skill_rolling(batter_skill, windows=ROLLING_WINDOWS)

    logger.info("Aggregating pitcher skill stats")
    pitcher_skill = aggregate_pitcher_skill_stats(raw_data)
    pitcher_skill = compute_pitcher_skill_rolling(pitcher_skill, windows=ROLLING_WINDOWS)

    logger.info("Merging skill features onto PA rows")
    feature_df = merge_skill_features_onto_pas(feature_df, batter_skill, pitcher_skill)

    logger.info("Aggregating batter pitch stats")
    batter_pitch = aggregate_batter_pitch_stats(raw_data)
    batter_pitch = compute_batter_pitch_rolling(batter_pitch, windows=ROLLING_WINDOWS)

    logger.info("Aggregating pitcher pitch stats")
    pitcher_pitch = aggregate_pitcher_pitch_stats(raw_data)
    pitcher_pitch = compute_pitcher_pitch_rolling(pitcher_pitch, windows=ROLLING_WINDOWS)

    logger.info("Merging pitch features onto PA rows")
    feature_df = merge_pitch_features_onto_pas(feature_df, batter_pitch, pitcher_pitch)

    logger.info("Adding interaction features")
    feature_df = add_interaction_features(feature_df)

    logger.info(
        f"Feature matrix complete: {feature_df.shape[0]:,} rows, "
        f"{feature_df.shape[1]} columns"
    )
    return feature_df


def build_starter_matrix() -> pd.DataFrame:
    logger.info("Starting starter-game feature matrix build")
    raw_data = load_raw_data()
    starter_matrix = build_starter_game_matrix(raw_data)

    Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
    starter_matrix.to_parquet(STARTER_GAME_MATRIX_PATH, index=False)
    logger.info(f"Saved starter game matrix to {STARTER_GAME_MATRIX_PATH}")

    target_mean = starter_matrix["n_k_against"].mean()
    date_min = starter_matrix["game_date"].min().date()
    date_max = starter_matrix["game_date"].max().date()

    logger.info(
        f"Summary: {len(starter_matrix):,} rows | "
        f"Mean K: {target_mean:.2f} | "
        f"Date range: {date_min} to {date_max} | "
        f"Features: {starter_matrix.shape[1]}"
    )
    return starter_matrix


def main() -> None:
    import sys

    if "--starter" in sys.argv:
        build_starter_matrix()
        return

    feature_matrix = build_feature_matrix()

    Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
    feature_matrix.to_parquet(FEATURE_MATRIX_PATH, index=False)
    logger.info(f"Saved feature matrix to {FEATURE_MATRIX_PATH}")

    k_rate = feature_matrix["is_k"].mean() if "is_k" in feature_matrix.columns else float("nan")

    date_min = feature_matrix["game_date"].min().date()
    date_max = feature_matrix["game_date"].max().date()

    logger.info(
        f"Summary: {len(feature_matrix):,} rows | "
        f"K rate: {k_rate:.4f} | "
        f"Date range: {date_min} to {date_max} | "
        f"Features: {feature_matrix.shape[1]}"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
