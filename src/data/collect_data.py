"""
Data collection module for MLB home run prediction.

Fetches Statcast pitch-by-pitch data, player-specific stats, and weather
data across multiple seasons. Supports incremental collection — already
fetched seasons are skipped automatically.
"""

import logging
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pybaseball import statcast, statcast_batter, statcast_pitcher

from config.data_collection_config import (
    PROCESSED_DATA_DIR,
    RATE_LIMIT_DELAY,
    RAW_DATA_DIR,
    SEASONS,
    season_dates,
)
from src.data.processing import (
    process_batter_data,
    process_pitcher_data,
    process_statcast_data,
)
from src.utils.file_utils import ensure_directory, load_dataframe, save_dataframe

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 5


def _raw_path(name: str, year: int) -> str:
    return os.path.join(RAW_DATA_DIR, f"{name}_{year}.parquet")


def _processed_path(name: str, year: int) -> str:
    return os.path.join(PROCESSED_DATA_DIR, f"processed_{name}_{year}.csv")


def _already_collected(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_parquet(path)
        return len(df) > 0
    except Exception:
        return False


def _fetch_with_retry(fetch_fn, description: str) -> pd.DataFrame:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = fetch_fn()
            if df is not None and not df.empty:
                return df
            logger.warning(f"No data returned for {description} (attempt {attempt})")
        except Exception as e:
            wait = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
            logger.error(f"{description} failed (attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)
    return pd.DataFrame()


def fetch_statcast_season(year: int) -> pd.DataFrame:
    start_date, end_date = season_dates(year)
    logger.info(f"Fetching Statcast data for {year} ({start_date} to {end_date})")

    df = _fetch_with_retry(
        lambda: statcast(start_date, end_date),
        f"Statcast {year}",
    )
    if not df.empty:
        logger.info(f"Fetched {len(df):,} rows for {year}")
    return df


def fetch_player_data(
    player_ids: list[int], year: int, player_type: str
) -> pd.DataFrame:
    start_date, end_date = season_dates(year)
    logger.info(
        f"Fetching {player_type} data for {len(player_ids)} players in {year}"
    )

    fetch_fn = statcast_batter if player_type == "batter" else statcast_pitcher
    all_data: list[pd.DataFrame] = []

    for i, player_id in enumerate(player_ids, 1):
        try:
            df = fetch_fn(start_dt=start_date, end_dt=end_date, player_id=player_id)
            if df is not None and not df.empty:
                all_data.append(df)
                logger.info(
                    f"  [{i}/{len(player_ids)}] player {player_id}: {len(df)} rows"
                )
            time.sleep(RATE_LIMIT_DELAY)
        except Exception as e:
            logger.error(f"  [{i}/{len(player_ids)}] player {player_id} failed: {e}")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total {player_type} rows for {year}: {len(combined):,}")
        return combined
    return pd.DataFrame()


def _save_raw_and_processed(df: pd.DataFrame, name: str, year: int, process_fn):
    if df.empty:
        return
    save_dataframe(df, _raw_path(name, year), format="parquet")
    processed = process_fn(df)
    save_dataframe(processed, _processed_path(name, year), format="csv")


def collect_season(year: int) -> None:
    statcast_path = _raw_path("statcast", year)
    if _already_collected(statcast_path):
        logger.info(f"Season {year} already collected — skipping")
        return

    logger.info(f"{'=' * 60}")
    logger.info(f"Collecting season {year}")
    logger.info(f"{'=' * 60}")

    statcast_df = fetch_statcast_season(year)
    _save_raw_and_processed(statcast_df, "statcast", year, process_statcast_data)

    if statcast_df.empty:
        logger.warning(f"No Statcast data for {year} — skipping player collection")
        return

    batter_ids = statcast_df["batter"].unique().tolist()
    pitcher_ids = statcast_df["pitcher"].unique().tolist()

    batter_df = fetch_player_data(batter_ids, year, "batter")
    _save_raw_and_processed(batter_df, "batter", year, process_batter_data)

    pitcher_df = fetch_player_data(pitcher_ids, year, "pitcher")
    _save_raw_and_processed(pitcher_df, "pitcher", year, process_pitcher_data)


def combine_all_seasons():
    for name in ("statcast", "batter", "pitcher"):
        pattern = f"{name}_*.parquet"
        files = sorted(Path(RAW_DATA_DIR).glob(pattern))
        if not files:
            continue

        combined = pd.concat(
            [pd.read_parquet(f) for f in files], ignore_index=True
        )
        combined_path = os.path.join(RAW_DATA_DIR, f"{name}_all.parquet")
        save_dataframe(combined, combined_path, format="parquet")
        logger.info(f"Combined {len(files)} files into {combined_path} ({len(combined):,} rows)")


def main() -> None:
    ensure_directory(RAW_DATA_DIR)
    ensure_directory(PROCESSED_DATA_DIR)

    for year in SEASONS:
        collect_season(year)

    combine_all_seasons()
    logger.info("Data collection complete")


if __name__ == "__main__":
    main()
