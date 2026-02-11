import logging

import numpy as np
import pandas as pd

from config.feature_config import (
    AL_TEAMS,
    AT_BAT_EXCLUSIONS,
    BARREL_MAX_LAUNCH_ANGLE,
    BARREL_MIN_EXIT_VELO,
    BARREL_MIN_LAUNCH_ANGLE,
    HIT_EVENTS,
    MIN_GAMES_FOR_ROLLING,
    PLATE_APPEARANCE_EVENTS,
    ROLLING_WINDOWS,
)

logger = logging.getLogger(__name__)

def _filter_plate_appearances(df: pd.DataFrame) -> pd.DataFrame:
    pa_df = df[df["events"].notna()].copy()
    pa_df = pa_df[pa_df["events"].isin(PLATE_APPEARANCE_EVENTS)].copy()
    pa_df["is_hr"] = (pa_df["events"] == "home_run").astype(int)
    pa_df["is_hit"] = pa_df["events"].isin(HIT_EVENTS).astype(int)
    pa_df["is_k"] = pa_df["events"].str.startswith("strikeout").astype(int)
    pa_df["is_bb"] = pa_df["events"].isin(["walk", "hit_by_pitch", "intent_walk"]).astype(int)
    pa_df["is_barrel"] = _is_barrel(pa_df).astype(int)
    return pa_df


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _is_barrel(df: pd.DataFrame) -> pd.Series:
    return (
        (df["launch_speed"] >= BARREL_MIN_EXIT_VELO)
        & (df["launch_angle"] >= BARREL_MIN_LAUNCH_ANGLE)
        & (df["launch_angle"] <= BARREL_MAX_LAUNCH_ANGLE)
    )


def _compute_rolling_rates(
    df: pd.DataFrame,
    group_col: str,
    rate_columns: dict[str, str | tuple[str, str]],
    windows: list[int],
    prefix: str,
) -> pd.DataFrame:
    df = df.copy()

    per_game_cols: list[str] = []
    for output_name, source in rate_columns.items():
        if isinstance(source, tuple):
            num_col, denom_col = source
            col_name = f"_tmp_{output_name}"
            df[col_name] = _safe_divide(df[num_col], df[denom_col])
        else:
            col_name = f"_tmp_{output_name}"
            df[col_name] = df[source]
        per_game_cols.append((output_name, col_name))

    for w in windows:
        for output_name, col_name in per_game_cols:
            shifted = df.groupby(group_col)[col_name].transform(
                lambda s: s.rolling(window=w, min_periods=MIN_GAMES_FOR_ROLLING)
                .mean()
                .shift(1)
            )
            df[f"{prefix}_{output_name}_{w}g"] = shifted

    tmp_cols = [col_name for _, col_name in per_game_cols]
    df.drop(columns=tmp_cols, inplace=True)
    return df


def _compute_hr_streak(series: pd.Series, window: int) -> pd.Series:
    return (
        series.rolling(window=window, min_periods=1)
        .sum()
        .shift(1)
    )


def _compute_games_since_last_hr(hr_series: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=hr_series.index)
    games_since = np.nan
    for i, had_hr in enumerate(hr_series):
        if i == 0:
            result.iloc[i] = np.nan
        else:
            result.iloc[i] = games_since
        if had_hr > 0:
            games_since = 0
        elif np.isnan(games_since):
            games_since = np.nan
        else:
            games_since += 1
    return result


def aggregate_to_batter_game(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    pa_df = _filter_plate_appearances(df)
    pa_df["is_ab_exclusion"] = pa_df["events"].isin(AT_BAT_EXCLUSIONS).astype(int)

    logger.info(
        "Aggregating %d plate-appearance rows to batter-game level",
        len(pa_df),
    )

    grouped = pa_df.groupby(["batter", "game_pk", "game_date"], sort=False)

    agg = grouped.agg(
        hit_hr=("is_hr", "max"),
        n_pa=("is_hr", "size"),
        n_hits=("is_hit", "sum"),
        n_hr=("is_hr", "sum"),
        n_k=("is_k", "sum"),
        n_bb=("is_bb", "sum"),
        n_barrel=("is_barrel", "sum"),
        n_ab_exclusions=("is_ab_exclusion", "sum"),
        avg_exit_velo=("launch_speed", "mean"),
        max_exit_velo=("launch_speed", "max"),
        avg_launch_angle=("launch_angle", "mean"),
    ).reset_index()

    agg["n_ab"] = agg["n_pa"] - agg["n_ab_exclusions"]
    agg.drop(columns=["n_ab_exclusions"], inplace=True)

    agg["hit_hr"] = (agg["hit_hr"] > 0).astype(int)

    meta = _extract_batter_game_metadata(pa_df)
    agg = agg.merge(meta, on=["batter", "game_pk", "game_date"], how="left")

    agg.sort_values(["batter", "game_date"], inplace=True)
    agg.reset_index(drop=True, inplace=True)

    logger.info("Batter-game aggregation complete: %d rows", len(agg))
    return agg


def _extract_batter_game_metadata(pa_df: pd.DataFrame) -> pd.DataFrame:
    first_rows = (
        pa_df.sort_values("at_bat_number")
        .groupby(["batter", "game_pk", "game_date"], sort=False)
        .first()
        .reset_index()
    )
    meta = first_rows[["batter", "game_pk", "game_date", "home_team", "away_team", "stand", "p_throws"]].copy()

    is_home = (
        pa_df.groupby(["batter", "game_pk", "game_date"], sort=False)["inning_topbot"]
        .agg(lambda s: (s == "Bot").mean() > 0.5)
        .astype(int)
        .reset_index()
        .rename(columns={"inning_topbot": "is_home"})
    )
    meta = meta.merge(is_home, on=["batter", "game_pk", "game_date"], how="left")
    return meta


def compute_batter_rolling_stats(
    df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    if windows is None:
        windows = ROLLING_WINDOWS

    df = df.copy()
    df.sort_values(["batter", "game_date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("Computing batter rolling stats for windows %s", windows)

    rate_columns: dict[str, str | tuple[str, str]] = {
        "hr_rate": ("n_hr", "n_pa"),
        "barrel_rate": ("n_barrel", "n_ab"),
        "avg_exit_velo": "avg_exit_velo",
        "avg_launch_angle": "avg_launch_angle",
        "k_rate": ("n_k", "n_pa"),
        "bb_rate": ("n_bb", "n_pa"),
        "batting_avg": ("n_hits", "n_ab"),
    }

    df = _compute_rolling_rates(df, "batter", rate_columns, windows, "batter")

    for w in windows:
        df[f"batter_hr_streak_{w}g"] = df.groupby("batter")["hit_hr"].transform(
            lambda s: _compute_hr_streak(s, w)
        )

    df["games_since_last_hr"] = df.groupby("batter")["hit_hr"].transform(
        _compute_games_since_last_hr
    )

    logger.info("Batter rolling stats complete")
    return df


def compute_pitcher_rolling_stats(
    df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    if windows is None:
        windows = ROLLING_WINDOWS

    pitcher_games = _aggregate_to_pitcher_game(df)
    pitcher_games = _apply_pitcher_rolling(pitcher_games, windows)
    return pitcher_games


def _aggregate_to_pitcher_game(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    pa_df = _filter_plate_appearances(df)

    logger.info(
        "Aggregating %d plate-appearance rows to pitcher-game level",
        len(pa_df),
    )

    grouped = pa_df.groupby(["pitcher", "game_pk", "game_date"], sort=False)

    agg = grouped.agg(
        n_batters_faced=("is_hr", "size"),
        n_pa_against=("is_hr", "size"),
        n_hr_against=("is_hr", "sum"),
        n_hits_against=("is_hit", "sum"),
        n_k_against=("is_k", "sum"),
        n_bb_against=("is_bb", "sum"),
        n_barrel_against=("is_barrel", "sum"),
        avg_exit_velo_against=("launch_speed", "mean"),
    ).reset_index()

    agg.sort_values(["pitcher", "game_date"], inplace=True)
    agg.reset_index(drop=True, inplace=True)

    logger.info("Pitcher-game aggregation complete: %d rows", len(agg))
    return agg


def _apply_pitcher_rolling(
    df: pd.DataFrame,
    windows: list[int],
) -> pd.DataFrame:
    logger.info("Computing pitcher rolling stats for windows %s", windows)

    rate_columns: dict[str, str | tuple[str, str]] = {
        "hr_allowed_rate": ("n_hr_against", "n_pa_against"),
        "barrel_rate_against": ("n_barrel_against", "n_pa_against"),
        "avg_exit_velo_against": "avg_exit_velo_against",
        "k_rate": ("n_k_against", "n_pa_against"),
        "bb_rate": ("n_bb_against", "n_pa_against"),
        "whip_proxy": ("_whip_num", "n_batters_faced"),
    }

    df = df.copy()
    df["_whip_num"] = df["n_hits_against"] + df["n_bb_against"]

    df = _compute_rolling_rates(df, "pitcher", rate_columns, windows, "pitcher")

    if "_whip_num" in df.columns:
        df.drop(columns=["_whip_num"], inplace=True)

    logger.info("Pitcher rolling stats complete")
    return df


def add_platoon_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["platoon"] = df["stand"] + "_vs_" + df["p_throws"]

    df["platoon_advantage"] = (
        ((df["stand"] == "L") & (df["p_throws"] == "R"))
        | ((df["stand"] == "R") & (df["p_throws"] == "L"))
    ).astype(int)

    logger.info("Platoon features added")
    return df


def add_game_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    df["month"] = df["game_date"].dt.month
    df["day_of_week"] = df["game_date"].dt.dayofweek
    df["is_dh"] = df["home_team"].isin(AL_TEAMS).astype(int)

    logger.info("Game context features added")
    return df
