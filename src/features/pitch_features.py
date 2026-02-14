import logging

import numpy as np
import pandas as pd

from config.feature_config import (
    BATTER_PITCH_FEATURES,
    FASTBALL_TYPES,
    PITCHER_PITCH_FEATURES,
    ROLLING_WINDOWS,
)
from src.features.feature_engineering import compute_rolling_rates

logger = logging.getLogger(__name__)

SWING_DESCRIPTIONS: set[str] = {
    "swinging_strike", "swinging_strike_blocked", "foul_tip",
    "foul", "foul_bunt", "hit_into_play",
    "hit_into_play_no_out", "hit_into_play_score",
    "missed_bunt", "bunt_foul_tip",
}

SWINGING_STRIKE_DESCRIPTIONS: set[str] = {
    "swinging_strike", "swinging_strike_blocked", "foul_tip",
}

CONTACT_DESCRIPTIONS: set[str] = {
    "foul", "foul_bunt", "hit_into_play",
    "hit_into_play_no_out", "hit_into_play_score",
}

IN_ZONE_NUMBERS: set[int] = {1, 2, 3, 4, 5, 6, 7, 8, 9}
OUT_OF_ZONE_NUMBERS: set[int] = {11, 12, 13, 14}


def _classify_pitch_outcomes(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df[raw_df["description"].notna()].copy()

    df["is_swing"] = df["description"].isin(SWING_DESCRIPTIONS).astype(int)
    df["is_swinging_strike"] = df["description"].isin(SWINGING_STRIKE_DESCRIPTIONS).astype(int)
    df["is_called_strike"] = (df["description"] == "called_strike").astype(int)
    df["is_contact"] = df["description"].isin(CONTACT_DESCRIPTIONS).astype(int)

    zone = pd.to_numeric(df["zone"], errors="coerce")
    df["is_in_zone"] = zone.isin(IN_ZONE_NUMBERS).astype(int)
    df["is_out_of_zone"] = zone.isin(OUT_OF_ZONE_NUMBERS).astype(int)

    df["is_chase"] = (df["is_swing"] & df["is_out_of_zone"]).astype(int)
    df["is_zone_swing"] = (df["is_swing"] & df["is_in_zone"]).astype(int)
    df["is_zone_contact"] = (df["is_contact"] & df["is_in_zone"]).astype(int)
    df["is_taken"] = (1 - df["is_swing"]).astype(int)
    df["is_fastball"] = df["pitch_type"].isin(FASTBALL_TYPES).astype(int)

    return df


def aggregate_batter_pitch_stats(raw_df: pd.DataFrame) -> pd.DataFrame:
    classified = _classify_pitch_outcomes(raw_df)

    logger.info("Aggregating batter pitch stats from %d pitches", len(classified))

    grouped = classified.groupby(["batter", "game_pk", "game_date"], sort=False)

    agg = grouped.agg(
        n_pitches_seen=("is_swing", "size"),
        n_swings=("is_swing", "sum"),
        n_swinging_strikes=("is_swinging_strike", "sum"),
        n_called_strikes=("is_called_strike", "sum"),
        n_chases=("is_chase", "sum"),
        n_out_of_zone_pitches=("is_out_of_zone", "sum"),
        n_zone_swings=("is_zone_swing", "sum"),
        n_zone_contacts=("is_zone_contact", "sum"),
        n_pitches_taken=("is_taken", "sum"),
    ).reset_index()

    agg["game_date"] = pd.to_datetime(agg["game_date"])
    agg.sort_values(["batter", "game_date"], inplace=True)
    agg.reset_index(drop=True, inplace=True)

    logger.info("Batter pitch stats aggregation complete: %d rows", len(agg))
    return agg


def compute_batter_pitch_rolling(
    df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    if windows is None:
        windows = ROLLING_WINDOWS

    df = df.copy()
    df.sort_values(["batter", "game_date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("Computing batter pitch rolling stats for windows %s", windows)

    rate_columns: dict[str, str | tuple[str, str]] = {
        "swing_rate": ("n_swings", "n_pitches_seen"),
        "whiff_rate": ("n_swinging_strikes", "n_swings"),
        "chase_rate": ("n_chases", "n_out_of_zone_pitches"),
        "zone_contact_rate": ("n_zone_contacts", "n_zone_swings"),
        "called_strike_rate": ("n_called_strikes", "n_pitches_taken"),
    }

    df = compute_rolling_rates(df, "batter", rate_columns, windows, "batter")

    logger.info("Batter pitch rolling stats complete")
    return df


def aggregate_pitcher_pitch_stats(raw_df: pd.DataFrame) -> pd.DataFrame:
    classified = _classify_pitch_outcomes(raw_df)

    logger.info("Aggregating pitcher pitch stats from %d pitches", len(classified))

    grouped = classified.groupby(["pitcher", "game_pk", "game_date"], sort=False)

    agg = grouped.agg(
        n_pitches_thrown=("is_swing", "size"),
        n_swinging_strikes_induced=("is_swinging_strike", "sum"),
        n_chases_induced=("is_chase", "sum"),
        n_out_of_zone_thrown=("is_out_of_zone", "sum"),
        n_in_zone_thrown=("is_in_zone", "sum"),
    ).reset_index()

    fastball_pitches = classified[classified["is_fastball"] == 1]
    if "release_speed" in fastball_pitches.columns:
        velo_col = "release_speed"
    elif "effective_speed" in fastball_pitches.columns:
        velo_col = "effective_speed"
    else:
        velo_col = None

    if velo_col is not None:
        fb_velo = (
            fastball_pitches.groupby(["pitcher", "game_pk", "game_date"], sort=False)[velo_col]
            .mean()
            .reset_index()
            .rename(columns={velo_col: "avg_fastball_velo"})
        )
        agg = agg.merge(fb_velo, on=["pitcher", "game_pk", "game_date"], how="left")
    else:
        agg["avg_fastball_velo"] = np.nan

    agg["game_date"] = pd.to_datetime(agg["game_date"])
    agg.sort_values(["pitcher", "game_date"], inplace=True)
    agg.reset_index(drop=True, inplace=True)

    logger.info("Pitcher pitch stats aggregation complete: %d rows", len(agg))
    return agg


def compute_pitcher_pitch_rolling(
    df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    if windows is None:
        windows = ROLLING_WINDOWS

    df = df.copy()
    df.sort_values(["pitcher", "game_date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("Computing pitcher pitch rolling stats for windows %s", windows)

    rate_columns: dict[str, str | tuple[str, str]] = {
        "swstr_rate": ("n_swinging_strikes_induced", "n_pitches_thrown"),
        "chase_rate_induced": ("n_chases_induced", "n_out_of_zone_thrown"),
        "zone_rate": ("n_in_zone_thrown", "n_pitches_thrown"),
        "avg_fastball_velo": "avg_fastball_velo",
    }

    df = compute_rolling_rates(df, "pitcher", rate_columns, windows, "pitcher")

    logger.info("Pitcher pitch rolling stats complete")
    return df


def _extract_pitch_rolling_cols(
    df: pd.DataFrame,
    prefix: str,
    feature_names: list[str],
) -> list[str]:
    cols = []
    for col in df.columns:
        if not col.startswith(f"{prefix}_"):
            continue
        for feat in feature_names:
            if col.startswith(f"{prefix}_{feat}_"):
                cols.append(col)
                break
    return cols


def merge_pitch_features_onto_pas(
    pa_df: pd.DataFrame,
    batter_pitch_df: pd.DataFrame,
    pitcher_pitch_df: pd.DataFrame,
) -> pd.DataFrame:
    batter_cols = _extract_pitch_rolling_cols(batter_pitch_df, "batter", BATTER_PITCH_FEATURES)
    batter_merge = ["batter", "game_pk", "game_date"] + batter_cols
    batter_subset = batter_pitch_df[
        [c for c in batter_merge if c in batter_pitch_df.columns]
    ].copy()

    pitcher_cols = _extract_pitch_rolling_cols(pitcher_pitch_df, "pitcher", PITCHER_PITCH_FEATURES)
    pitcher_merge = ["pitcher", "game_pk", "game_date"] + pitcher_cols
    pitcher_subset = pitcher_pitch_df[
        [c for c in pitcher_merge if c in pitcher_pitch_df.columns]
    ].copy()

    merged = pa_df.merge(
        batter_subset,
        on=["batter", "game_pk", "game_date"],
        how="left",
    )
    batter_matched = merged[batter_cols[0]].notna().sum() if batter_cols else 0
    logger.info(
        "Batter pitch merge: %d / %d PAs matched (%.1f%%)",
        batter_matched,
        len(pa_df),
        100.0 * batter_matched / max(len(pa_df), 1),
    )

    merged = merged.merge(
        pitcher_subset,
        on=["pitcher", "game_pk", "game_date"],
        how="left",
    )
    pitcher_matched = merged[pitcher_cols[0]].notna().sum() if pitcher_cols else 0
    logger.info(
        "Pitcher pitch merge: %d / %d PAs matched (%.1f%%)",
        pitcher_matched,
        len(pa_df),
        100.0 * pitcher_matched / max(len(pa_df), 1),
    )

    return merged
