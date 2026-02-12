import logging
import time
from pathlib import Path

import pandas as pd
import requests

from config.data_collection_config import (
    SEASONS,
    WEATHER_RATE_LIMIT_DELAY,
    season_dates,
)
from config.feature_config import WEATHER_DATA_PATH
from config.park_config import PARKS

logger = logging.getLogger(__name__)

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GAME_HOUR = 19

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
]


def fetch_weather_for_park(
    team_code: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
    }

    response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    hourly = data.get("hourly", {})
    if not hourly or "time" not in hourly:
        logger.warning(f"No hourly data returned for {team_code} ({start_date} to {end_date})")
        return pd.DataFrame()

    df = pd.DataFrame(hourly)
    df["hour"] = pd.to_datetime(df["time"]).dt.hour
    df = df[df["hour"] == GAME_HOUR].copy()

    df["home_team"] = team_code
    df["game_date"] = pd.to_datetime(df["time"]).dt.date

    result = df.rename(columns={
        "temperature_2m": "temp_c",
        "relative_humidity_2m": "humidity_pct",
        "wind_speed_10m": "wind_speed_kmh",
        "wind_direction_10m": "wind_dir_deg",
    })

    return result[["home_team", "game_date", "temp_c", "humidity_pct", "wind_speed_kmh", "wind_dir_deg"]].reset_index(drop=True)


def collect_all_weather(
    seasons: list[int] | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    if seasons is None:
        seasons = SEASONS
    if output_path is None:
        output_path = WEATHER_DATA_PATH

    all_frames: list[pd.DataFrame] = []
    total_calls = len(PARKS) * len(seasons)
    call_number = 0

    for year in seasons:
        start_date, end_date = season_dates(year)
        for team_code, park_info in PARKS.items():
            call_number += 1
            logger.info(
                f"[{call_number}/{total_calls}] Fetching weather for {team_code} "
                f"{year} ({start_date} to {end_date})"
            )
            try:
                df = fetch_weather_for_park(
                    team_code, park_info.lat, park_info.lon, start_date, end_date
                )
                if not df.empty:
                    all_frames.append(df)
                    logger.info(f"  Got {len(df)} days of weather data")
                else:
                    logger.warning(f"  No data returned for {team_code} {year}")
            except Exception as e:
                logger.warning(f"  Failed to fetch weather for {team_code} {year}: {e}")

            time.sleep(WEATHER_RATE_LIMIT_DELAY)

    if not all_frames:
        logger.warning("No weather data collected")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    combined["game_date"] = pd.to_datetime(combined["game_date"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(combined):,} weather records to {output_path}")

    return combined


def load_weather_data(path: Path | None = None) -> pd.DataFrame | None:
    if path is None:
        path = WEATHER_DATA_PATH

    if not path.exists():
        logger.warning(f"Weather data file not found: {path}")
        return None

    df = pd.read_parquet(path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    logger.info(f"Loaded {len(df):,} weather records from {path}")
    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    collect_all_weather()
