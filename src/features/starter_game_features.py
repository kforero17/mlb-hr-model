import logging

import numpy as np
import pandas as pd

from config.feature_config import (
    MIN_BATTERS_FACED_STARTER,
    PLATE_APPEARANCE_EVENTS,
    ROLLING_WINDOWS,
)
from src.data.collect_weather import load_weather_data
from src.features.environment_features import (
    add_environment_features,
    compute_park_hr_factors,
    compute_park_k_factors,
)
from src.features.feature_engineering import (
    add_game_context_features,
    compute_pitcher_rolling_stats,
    compute_rolling_rates,
)
from src.features.pitch_features import (
    IN_ZONE_NUMBERS,
    OUT_OF_ZONE_NUMBERS,
    SWING_DESCRIPTIONS,
    SWINGING_STRIKE_DESCRIPTIONS,
    aggregate_pitcher_pitch_stats,
    compute_pitcher_pitch_rolling,
)
from src.features.skill_features import (
    aggregate_pitcher_skill_stats,
    compute_pitcher_skill_rolling,
)

logger = logging.getLogger(__name__)


def identify_starters(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    inning1 = df[df["inning"] == 1].copy()

    records = []
    for (game_pk, topbot), group in inning1.groupby(["game_pk", "inning_topbot"]):
        first_row = group.loc[group["at_bat_number"].idxmin()]
        is_home = 1 if topbot == "Top" else 0
        records.append({
            "pitcher": first_row["pitcher"],
            "game_pk": game_pk,
            "game_date": first_row["game_date"],
            "home_team": first_row["home_team"],
            "away_team": first_row["away_team"],
            "p_throws": first_row["p_throws"],
            "is_home": is_home,
        })

    starters = pd.DataFrame(records)
    logger.info("Identified %d starter appearances", len(starters))
    return starters


def aggregate_starter_game(
    raw_df: pd.DataFrame,
    starters_df: pd.DataFrame,
) -> pd.DataFrame:
    df = raw_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    pa_df = df[df["events"].notna() & df["events"].isin(PLATE_APPEARANCE_EVENTS)].copy()
    pa_df["is_k"] = pa_df["events"].str.startswith("strikeout").fillna(False).astype(int)
    pa_df["is_bb"] = pa_df["events"].isin(["walk", "hit_by_pitch", "intent_walk"]).fillna(False).astype(int)
    pa_df["is_hit"] = pa_df["events"].isin(["single", "double", "triple", "home_run"]).fillna(False).astype(int)

    starter_keys = starters_df[["pitcher", "game_pk"]].drop_duplicates()
    pa_starters = pa_df.merge(starter_keys, on=["pitcher", "game_pk"], how="inner")

    pitch_counts = (
        df.merge(starter_keys, on=["pitcher", "game_pk"], how="inner")
        .groupby(["pitcher", "game_pk"])
        .size()
        .reset_index(name="n_pitches")
    )

    grouped = pa_starters.groupby(["pitcher", "game_pk", "game_date"], sort=False)

    agg = grouped.agg(
        n_k_against=("is_k", "sum"),
        n_batters_faced=("is_k", "size"),
        n_bb_against=("is_bb", "sum"),
        n_hits_against=("is_hit", "sum"),
        max_inning=("inning", "max"),
    ).reset_index()

    agg = agg.merge(pitch_counts, on=["pitcher", "game_pk"], how="left")

    meta = starters_df[["pitcher", "game_pk", "game_date", "home_team", "away_team", "p_throws", "is_home"]].copy()
    agg = agg.merge(meta, on=["pitcher", "game_pk", "game_date"], how="left")

    agg["opposing_team"] = np.where(agg["is_home"] == 1, agg["away_team"], agg["home_team"])

    before_filter = len(agg)
    agg = agg[agg["n_batters_faced"] >= MIN_BATTERS_FACED_STARTER].copy()
    logger.info(
        "Starter game aggregation: %d -> %d rows after min BF filter (%d)",
        before_filter,
        len(agg),
        MIN_BATTERS_FACED_STARTER,
    )

    agg.sort_values(["pitcher", "game_date"], inplace=True)
    agg.reset_index(drop=True, inplace=True)
    return agg


def compute_pitcher_workload_rolling(
    starter_df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    if windows is None:
        windows = ROLLING_WINDOWS

    df = starter_df.copy()
    df.sort_values(["pitcher", "game_date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["went_6plus"] = (df["max_inning"] >= 6).astype(int)

    logger.info("Computing pitcher workload rolling stats for windows %s", windows)

    rate_columns: dict[str, str | tuple[str, str]] = {
        "avg_bf": "n_batters_faced",
        "avg_pitch_count": "n_pitches",
        "pct_6plus_innings": "went_6plus",
    }

    df = compute_rolling_rates(df, "pitcher", rate_columns, windows, "pitcher")

    df["pitcher_days_since_prev_game"] = (
        df.groupby("pitcher")["game_date"].diff().dt.days
    )

    df.drop(columns=["went_6plus"], inplace=True)

    logger.info("Pitcher workload rolling stats complete")
    return df


def compute_opposing_team_k_features(
    raw_df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    if windows is None:
        windows = ROLLING_WINDOWS

    df = raw_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    df["batting_team"] = np.where(
        df["inning_topbot"] == "Top", df["away_team"], df["home_team"]
    )

    pa_df = df[df["events"].notna() & df["events"].isin(PLATE_APPEARANCE_EVENTS)].copy()
    pa_df["is_k"] = pa_df["events"].str.startswith("strikeout").fillna(False).astype(int)

    zone = pd.to_numeric(df["zone"], errors="coerce")
    df["is_swing"] = df["description"].isin(SWING_DESCRIPTIONS).astype(int)
    df["is_whiff"] = df["description"].isin(SWINGING_STRIKE_DESCRIPTIONS).astype(int)
    df["is_out_of_zone"] = zone.isin(OUT_OF_ZONE_NUMBERS).astype(int)
    df["is_chase"] = (df["is_swing"] & df["is_out_of_zone"]).astype(int)

    pa_agg = (
        pa_df.groupby(["batting_team", "game_pk", "game_date"], sort=False)
        .agg(n_pa=("is_k", "size"), n_k=("is_k", "sum"))
        .reset_index()
    )

    pitch_agg = (
        df.groupby(["batting_team", "game_pk", "game_date"], sort=False)
        .agg(
            n_swings=("is_swing", "sum"),
            n_whiffs=("is_whiff", "sum"),
            n_chases=("is_chase", "sum"),
            n_out_of_zone=("is_out_of_zone", "sum"),
        )
        .reset_index()
    )

    team_game = pa_agg.merge(
        pitch_agg, on=["batting_team", "game_pk", "game_date"], how="outer"
    )

    team_game.sort_values(["batting_team", "game_date"], inplace=True)
    team_game.reset_index(drop=True, inplace=True)

    logger.info("Computing opposing team K rolling stats for windows %s", windows)

    rate_columns: dict[str, str | tuple[str, str]] = {
        "k_rate": ("n_k", "n_pa"),
        "whiff_rate": ("n_whiffs", "n_swings"),
        "chase_rate": ("n_chases", "n_out_of_zone"),
    }

    team_game = compute_rolling_rates(
        team_game, "batting_team", rate_columns, windows, "opp_team"
    )

    logger.info("Opposing team K rolling stats complete: %d rows", len(team_game))
    return team_game


def compute_starter_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    interactions = [
        ("ix_pitcher_k_x_opp_team_k", "pitcher_k_rate_50g", "opp_team_k_rate_50g"),
        ("ix_pitcher_swstr_x_opp_whiff", "pitcher_swstr_rate_50g", "opp_team_whiff_rate_50g"),
        ("ix_pitcher_k_x_park_k", "pitcher_k_rate_50g", "park_k_factor"),
        ("ix_pitcher_bf_x_pitcher_k", "pitcher_avg_bf_50g", "pitcher_k_rate_50g"),
    ]

    added = 0
    for name, col_a, col_b in interactions:
        if col_a not in df.columns or col_b not in df.columns:
            missing = col_a if col_a not in df.columns else col_b
            logger.warning("Skipping %s: missing %s", name, missing)
            continue
        df[name] = df[col_a] * df[col_b]
        added += 1

    logger.info("Added %d starter interaction features", added)
    return df


def compute_opp_lineup_handedness(
    raw_df: pd.DataFrame,
    starters_df: pd.DataFrame,
) -> pd.DataFrame:
    df = raw_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    pa_df = df[df["events"].notna() & df["events"].isin(PLATE_APPEARANCE_EVENTS)].copy()

    pa_df["batting_team"] = np.where(
        pa_df["inning_topbot"] == "Top", pa_df["away_team"], pa_df["home_team"]
    )

    unique_batters = (
        pa_df.groupby(["game_pk", "batting_team", "batter"], sort=False)["stand"]
        .first()
        .reset_index()
    )

    lhb_pct = (
        unique_batters.groupby(["game_pk", "batting_team"])
        .apply(lambda g: (g["stand"] == "L").mean(), include_groups=False)
        .reset_index(name="opp_lineup_pct_lhb")
    )

    starter_merge = starters_df[["pitcher", "game_pk", "game_date", "opposing_team"]].copy()
    starter_merge = starter_merge.rename(columns={"opposing_team": "batting_team"})

    result = starter_merge.merge(lhb_pct, on=["game_pk", "batting_team"], how="left")
    result = result.rename(columns={"batting_team": "opposing_team"})

    logger.info(
        "Opposing lineup handedness computed: %.1f%% matched",
        100.0 * result["opp_lineup_pct_lhb"].notna().mean(),
    )
    return result


def _merge_prefixed_columns(
    target: pd.DataFrame,
    source: pd.DataFrame,
    prefix: str,
    merge_keys: list[str],
) -> pd.DataFrame:
    stat_cols = [c for c in source.columns if c.startswith(prefix)]
    all_cols = merge_keys + stat_cols
    subset = source[[c for c in all_cols if c in source.columns]].copy()
    return target.merge(subset, on=merge_keys, how="left")


def _compute_pitcher_rolling_features(raw_df: pd.DataFrame) -> list[pd.DataFrame]:
    pitcher_rolling = compute_pitcher_rolling_stats(raw_df)

    pitcher_pitch_agg = aggregate_pitcher_pitch_stats(raw_df)
    pitcher_pitch_rolling = compute_pitcher_pitch_rolling(pitcher_pitch_agg)

    pitcher_skill_agg = aggregate_pitcher_skill_stats(raw_df)
    pitcher_skill_rolling = compute_pitcher_skill_rolling(pitcher_skill_agg)

    return [pitcher_rolling, pitcher_pitch_rolling, pitcher_skill_rolling]


def build_starter_game_matrix(raw_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting starter-game feature matrix build")

    raw_df = raw_df.copy()
    raw_df["game_date"] = pd.to_datetime(raw_df["game_date"])

    starters = identify_starters(raw_df)
    starter_df = aggregate_starter_game(raw_df, starters)
    starter_df = compute_pitcher_workload_rolling(starter_df)

    pitcher_keys = ["pitcher", "game_pk", "game_date"]
    for rolling_df in _compute_pitcher_rolling_features(raw_df):
        starter_df = _merge_prefixed_columns(
            starter_df, rolling_df, "pitcher_", pitcher_keys
        )

    opp_team_k = compute_opposing_team_k_features(raw_df)
    opp_team_k = opp_team_k.rename(columns={"batting_team": "opposing_team"})
    starter_df = _merge_prefixed_columns(
        starter_df, opp_team_k, "opp_team_",
        ["opposing_team", "game_pk", "game_date"],
    )

    park_hr_factors = compute_park_hr_factors(raw_df)
    park_k_factors = compute_park_k_factors(raw_df)
    weather = load_weather_data()
    starter_df = add_environment_features(
        starter_df, park_hr_factors, weather, park_k_factors
    )

    opp_hand = compute_opp_lineup_handedness(raw_df, starter_df)
    starter_df = _merge_prefixed_columns(
        starter_df, opp_hand, "opp_lineup_pct_lhb",
        ["pitcher", "game_pk", "game_date"],
    )

    starter_df = add_game_context_features(starter_df)
    starter_df = compute_starter_interactions(starter_df)

    logger.info(
        "Starter-game feature matrix complete: %d rows, %d columns",
        starter_df.shape[0],
        starter_df.shape[1],
    )
    return starter_df
