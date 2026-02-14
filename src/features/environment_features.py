import logging

import numpy as np
import pandas as pd

from config.feature_config import PLATE_APPEARANCE_EVENTS
from config.park_config import PARKS

logger = logging.getLogger(__name__)


def compute_park_hr_factors(
    raw_df: pd.DataFrame,
    min_pa_per_park_year: int = 500,
) -> pd.DataFrame:
    raw_df = raw_df.copy()
    raw_df["game_date"] = pd.to_datetime(raw_df["game_date"])

    pa_mask = raw_df["events"].notna() & raw_df["events"].isin(PLATE_APPEARANCE_EVENTS)
    pa_df = raw_df.loc[pa_mask].copy()
    pa_df["is_hr"] = (pa_df["events"] == "home_run").astype(int)

    logger.info("Computing park HR factors from %d plate appearances", len(pa_df))

    overall = _compute_overall_park_event_factors(pa_df, "is_hr", "park_hr_factor", min_pa_per_park_year)
    handedness = _compute_handedness_park_event_factors(pa_df, "is_hr", "park_hr_factor_handedness", min_pa_per_park_year)

    combined = handedness.merge(overall, on=["home_team", "year"], how="outer")

    logger.info(
        "Park HR factors computed: %d rows covering %d parks",
        len(combined),
        combined["home_team"].nunique(),
    )
    return combined


def _add_static_park_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df["elevation_ft"] = df["home_team"].map(
        {team: info.elevation_ft for team, info in PARKS.items()}
    )
    df["roof_type"] = df["home_team"].map(
        {team: info.roof for team, info in PARKS.items()}
    )
    return df


def _merge_park_event_factors(
    df: pd.DataFrame,
    park_factors_df: pd.DataFrame,
    overall_col: str,
    handedness_col: str,
) -> pd.DataFrame:
    overall_cols = ["home_team", "year", overall_col]
    overall_available = [c for c in overall_cols if c in park_factors_df.columns]
    overall = park_factors_df[overall_available].drop_duplicates()

    df = df.merge(overall, on=["home_team", "year"], how="left")

    if "stand" in df.columns:
        hand_cols = ["home_team", "year", "stand", handedness_col]
        hand_available = [c for c in hand_cols if c in park_factors_df.columns]
        hand = park_factors_df[hand_available].drop_duplicates()
        df = df.merge(hand, on=["home_team", "year", "stand"], how="left")
    else:
        df[handedness_col] = np.nan

    df[overall_col] = df[overall_col].fillna(1.0)
    df[handedness_col] = df[handedness_col].fillna(1.0)

    return df


def _merge_weather(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    weather = weather_df.copy()
    weather["game_date"] = pd.to_datetime(weather["game_date"])

    weather["temp_f"] = weather["temp_c"] * 9 / 5 + 32
    weather["wind_speed_mph"] = weather["wind_speed_kmh"] / 1.60934

    weather_merge_cols = [
        "home_team", "game_date",
        "temp_f", "humidity_pct", "wind_speed_mph", "wind_dir_deg",
    ]
    weather_subset = weather[weather_merge_cols].drop_duplicates(
        subset=["home_team", "game_date"]
    )

    df = df.merge(weather_subset, on=["home_team", "game_date"], how="left")
    return df


def _apply_dome_mask(df: pd.DataFrame) -> pd.DataFrame:
    is_dome = df["roof_type"] == "dome"
    dome_null_cols = ["temp_f", "humidity_pct", "wind_speed_mph", "wind_dir_deg"]
    for col in dome_null_cols:
        if col in df.columns:
            df.loc[is_dome, col] = np.nan
    return df


def _compute_wind_out_to_cf(df: pd.DataFrame) -> pd.DataFrame:
    cf_bearing = df["home_team"].map(
        {team: info.cf_bearing_deg for team, info in PARKS.items()}
    )

    df["wind_out_to_cf"] = -df["wind_speed_mph"] * np.cos(
        np.radians(df["wind_dir_deg"] - cf_bearing)
    )
    return df


def _compute_air_density_index(df: pd.DataFrame) -> pd.DataFrame:
    pressure_ratio = (1 - 2.2558e-5 * df["elevation_ft"]) ** 5.2559

    temp_ratio = 518.67 / (df["temp_f"] + 459.67)

    df["air_density_index"] = pressure_ratio * temp_ratio
    return df


def compute_park_k_factors(
    raw_df: pd.DataFrame,
    min_pa_per_park_year: int = 500,
) -> pd.DataFrame:
    raw_df = raw_df.copy()
    raw_df["game_date"] = pd.to_datetime(raw_df["game_date"])

    pa_mask = raw_df["events"].notna() & raw_df["events"].isin(PLATE_APPEARANCE_EVENTS)
    pa_df = raw_df.loc[pa_mask].copy()
    pa_df["is_k"] = pa_df["events"].str.startswith("strikeout").fillna(False).astype(int)

    logger.info("Computing park K factors from %d plate appearances", len(pa_df))

    overall = _compute_overall_park_event_factors(pa_df, "is_k", "park_k_factor", min_pa_per_park_year)
    handedness = _compute_handedness_park_event_factors(pa_df, "is_k", "park_k_factor_handedness", min_pa_per_park_year)

    combined = handedness.merge(overall, on=["home_team", "year"], how="outer")

    logger.info(
        "Park K factors computed: %d rows covering %d parks",
        len(combined),
        combined["home_team"].nunique(),
    )
    return combined


def _compute_overall_park_event_factors(
    pa_df: pd.DataFrame,
    event_col: str,
    factor_name: str,
    min_pa: int,
) -> pd.DataFrame:
    pa_df = pa_df.copy()
    pa_df["year"] = pa_df["game_date"].dt.year

    park_stats = (
        pa_df.groupby(["home_team", "year"])
        .agg(n_pa=(event_col, "size"), n_events=(event_col, "sum"))
        .reset_index()
    )
    park_stats["park_rate"] = park_stats["n_events"] / park_stats["n_pa"]

    league_stats = (
        pa_df.groupby("year")
        .agg(league_pa=(event_col, "size"), league_events=(event_col, "sum"))
        .reset_index()
    )
    league_stats["league_rate"] = league_stats["league_events"] / league_stats["league_pa"]

    park_stats = park_stats.merge(league_stats[["year", "league_rate"]], on="year", how="left")

    park_stats[factor_name] = np.where(
        (park_stats["league_rate"] > 0) & (park_stats["n_pa"] >= min_pa),
        park_stats["park_rate"] / park_stats["league_rate"],
        1.0,
    )

    park_stats["year"] = park_stats["year"] + 1

    return park_stats[["home_team", "year", factor_name]]


def _compute_handedness_park_event_factors(
    pa_df: pd.DataFrame,
    event_col: str,
    factor_name: str,
    min_pa: int,
) -> pd.DataFrame:
    pa_df = pa_df.copy()
    pa_df["year"] = pa_df["game_date"].dt.year

    park_hand_stats = (
        pa_df.groupby(["home_team", "year", "stand"])
        .agg(n_pa=(event_col, "size"), n_events=(event_col, "sum"))
        .reset_index()
    )
    park_hand_stats["park_rate_hand"] = park_hand_stats["n_events"] / park_hand_stats["n_pa"]

    league_hand_stats = (
        pa_df.groupby(["year", "stand"])
        .agg(league_pa=(event_col, "size"), league_events=(event_col, "sum"))
        .reset_index()
    )
    league_hand_stats["league_rate_hand"] = (
        league_hand_stats["league_events"] / league_hand_stats["league_pa"]
    )

    park_hand_stats = park_hand_stats.merge(
        league_hand_stats[["year", "stand", "league_rate_hand"]],
        on=["year", "stand"],
        how="left",
    )

    park_hand_stats[factor_name] = np.where(
        (park_hand_stats["league_rate_hand"] > 0) & (park_hand_stats["n_pa"] >= min_pa),
        park_hand_stats["park_rate_hand"] / park_hand_stats["league_rate_hand"],
        1.0,
    )

    park_hand_stats["year"] = park_hand_stats["year"] + 1

    return park_hand_stats[["home_team", "year", "stand", factor_name]]


def add_environment_features(
    df: pd.DataFrame,
    park_factors_df: pd.DataFrame,
    weather_df: pd.DataFrame | None,
    park_k_factors_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    had_year = "year" in df.columns
    df["year"] = df["game_date"].dt.year

    df = _add_static_park_metadata(df)
    df = _merge_park_event_factors(df, park_factors_df, "park_hr_factor", "park_hr_factor_handedness")

    if park_k_factors_df is not None:
        df = _merge_park_event_factors(df, park_k_factors_df, "park_k_factor", "park_k_factor_handedness")

    if weather_df is not None:
        df = _merge_weather(df, weather_df)
    else:
        for col in ["temp_f", "humidity_pct", "wind_speed_mph", "wind_dir_deg"]:
            df[col] = np.nan

    df = _apply_dome_mask(df)
    df = _compute_wind_out_to_cf(df)
    df = _compute_air_density_index(df)

    if not had_year:
        df.drop(columns=["year"], inplace=True)

    env_cols = [
        "elevation_ft", "roof_type", "park_hr_factor", "park_hr_factor_handedness",
        "temp_f", "wind_speed_mph", "wind_dir_deg", "humidity_pct",
        "wind_out_to_cf", "air_density_index",
    ]
    if park_k_factors_df is not None:
        env_cols += ["park_k_factor", "park_k_factor_handedness"]
    logger.info("Environment features added: %s", ", ".join(env_cols))
    return df
