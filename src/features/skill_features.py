import logging

import numpy as np
import pandas as pd

from config.feature_config import (
    BATTER_SKILL_FEATURES,
    BREAKING_TYPES,
    FASTBALL_TYPES,
    HARD_HIT_MIN_EXIT_VELO,
    OFFSPEED_TYPES,
    PITCHER_SKILL_FEATURES,
    PLATE_APPEARANCE_EVENTS,
    ROLLING_WINDOWS,
    SPRAY_ANGLE_PULL_THRESHOLD,
    SPRAY_HOME_X,
    SPRAY_HOME_Y,
    SWEET_SPOT_MAX_ANGLE,
    SWEET_SPOT_MIN_ANGLE,
)
from src.features.feature_engineering import compute_rolling_rates

logger = logging.getLogger(__name__)


def _compute_spray_angle(hc_x: pd.Series, hc_y: pd.Series) -> pd.Series:
    return np.degrees(np.arctan2(hc_x - SPRAY_HOME_X, SPRAY_HOME_Y - hc_y))


def _is_pull(stand: pd.Series, spray_angle: pd.Series) -> pd.Series:
    return (
        ((stand == "R") & (spray_angle < -SPRAY_ANGLE_PULL_THRESHOLD))
        | ((stand == "L") & (spray_angle > SPRAY_ANGLE_PULL_THRESHOLD))
    )


def _aggregate_batted_ball_stats(raw_df: pd.DataFrame) -> pd.DataFrame:
    batted = raw_df[raw_df["bb_type"].notna()].copy()

    batted["is_fly_ball"] = (batted["bb_type"] == "fly_ball").fillna(False).astype(int)
    batted["is_hard_hit"] = (batted["launch_speed"] >= HARD_HIT_MIN_EXIT_VELO).fillna(False).astype(int)
    batted["is_sweet_spot"] = (
        (batted["launch_angle"] >= SWEET_SPOT_MIN_ANGLE)
        & (batted["launch_angle"] <= SWEET_SPOT_MAX_ANGLE)
    ).fillna(False).astype(int)

    pull_mask = batted["hc_x"].notna() & batted["hc_y"].notna()
    spray_angle = _compute_spray_angle(batted["hc_x"], batted["hc_y"])
    batted["is_pull"] = 0
    batted.loc[pull_mask, "is_pull"] = _is_pull(
        batted.loc[pull_mask, "stand"], spray_angle[pull_mask]
    ).astype(int)

    grouped = batted.groupby(["batter", "game_pk", "game_date"], sort=False)

    agg = grouped.agg(
        n_batted_ball=("bb_type", "size"),
        n_fly_ball=("is_fly_ball", "sum"),
        n_pull=("is_pull", "sum"),
        n_hard_hit=("is_hard_hit", "sum"),
        n_sweet_spot=("is_sweet_spot", "sum"),
    ).reset_index()

    xstats = raw_df.groupby(["batter", "game_pk", "game_date"], sort=False).agg(
        avg_xslg=("estimated_slg_using_speedangle", "mean"),
        avg_xwoba=("estimated_woba_using_speedangle", "mean"),
    ).reset_index()

    return agg.merge(xstats, on=["batter", "game_pk", "game_date"], how="outer")


def _classify_pitch_category(pitch_type: pd.Series) -> pd.Series:
    category = pd.Series(pd.NA, index=pitch_type.index, dtype="string")
    category[pitch_type.isin(FASTBALL_TYPES)] = "fastball"
    category[pitch_type.isin(BREAKING_TYPES)] = "breaking"
    category[pitch_type.isin(OFFSPEED_TYPES)] = "offspeed"
    return category


def _aggregate_batter_pitch_type_stats(raw_df: pd.DataFrame) -> pd.DataFrame:
    pa_mask = raw_df["events"].notna() & raw_df["events"].isin(PLATE_APPEARANCE_EVENTS)
    pa_df = raw_df.loc[pa_mask].copy()

    pa_df["pitch_category"] = _classify_pitch_category(pa_df["pitch_type"])
    pa_df["is_hr"] = (pa_df["events"] == "home_run").fillna(False).astype(int)

    records = []
    for category in ("fastball", "breaking", "offspeed"):
        subset = pa_df[pa_df["pitch_category"] == category]
        grouped = subset.groupby(["batter", "game_pk", "game_date"], sort=False)
        cat_agg = grouped.agg(
            n_pa=(f"is_hr", "size"),
            n_hr=(f"is_hr", "sum"),
        ).reset_index()
        cat_agg = cat_agg.rename(columns={
            "n_pa": f"n_pa_vs_{category}",
            "n_hr": f"n_hr_vs_{category}",
        })
        records.append(cat_agg)

    result = records[0]
    for r in records[1:]:
        result = result.merge(r, on=["batter", "game_pk", "game_date"], how="outer")

    return result


def aggregate_batter_skill_stats(raw_df: pd.DataFrame) -> pd.DataFrame:
    raw_df = raw_df.copy()
    raw_df["game_date"] = pd.to_datetime(raw_df["game_date"])

    logger.info("Aggregating batter skill stats from %d raw rows", len(raw_df))

    batted_ball = _aggregate_batted_ball_stats(raw_df)
    pitch_type = _aggregate_batter_pitch_type_stats(raw_df)

    merged = batted_ball.merge(
        pitch_type, on=["batter", "game_pk", "game_date"], how="outer"
    )

    fill_cols = [c for c in merged.columns if c.startswith(("n_", "avg_"))]
    for col in fill_cols:
        if col.startswith("n_"):
            merged[col] = merged[col].fillna(0).astype(int)

    merged.sort_values(["batter", "game_date"], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    logger.info("Batter skill aggregation complete: %d rows", len(merged))
    return merged


def aggregate_pitcher_skill_stats(raw_df: pd.DataFrame) -> pd.DataFrame:
    raw_df = raw_df.copy()
    raw_df["game_date"] = pd.to_datetime(raw_df["game_date"])

    logger.info("Aggregating pitcher skill stats from %d raw rows", len(raw_df))

    batted_ball = _aggregate_pitcher_batted_ball(raw_df)
    pitch_usage = _aggregate_pitcher_pitch_usage(raw_df)

    merged = batted_ball.merge(
        pitch_usage, on=["pitcher", "game_pk", "game_date"], how="outer"
    )

    fill_cols = [c for c in merged.columns if c.startswith("n_")]
    for col in fill_cols:
        merged[col] = merged[col].fillna(0).astype(int)

    merged.sort_values(["pitcher", "game_date"], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    logger.info("Pitcher skill aggregation complete: %d rows", len(merged))
    return merged


def _aggregate_pitcher_batted_ball(raw_df: pd.DataFrame) -> pd.DataFrame:
    batted = raw_df[raw_df["bb_type"].notna()].copy()
    batted["is_ground_ball"] = (batted["bb_type"] == "ground_ball").fillna(False).astype(int)
    batted["is_fly_ball"] = (batted["bb_type"] == "fly_ball").fillna(False).astype(int)

    grouped = batted.groupby(["pitcher", "game_pk", "game_date"], sort=False)

    return grouped.agg(
        n_batted_ball=("bb_type", "size"),
        n_ground_ball=("is_ground_ball", "sum"),
        n_fly_ball=("is_fly_ball", "sum"),
    ).reset_index()


def _aggregate_pitcher_pitch_usage(raw_df: pd.DataFrame) -> pd.DataFrame:
    classified = raw_df[raw_df["pitch_type"].notna()].copy()
    classified["pitch_category"] = _classify_pitch_category(classified["pitch_type"])

    classified = classified[classified["pitch_category"].notna()]

    classified["is_fastball"] = (classified["pitch_category"] == "fastball").fillna(False).astype(int)
    classified["is_breaking"] = (classified["pitch_category"] == "breaking").fillna(False).astype(int)
    classified["is_offspeed"] = (classified["pitch_category"] == "offspeed").fillna(False).astype(int)

    grouped = classified.groupby(["pitcher", "game_pk", "game_date"], sort=False)

    return grouped.agg(
        n_pitches=("pitch_category", "size"),
        n_fastball=("is_fastball", "sum"),
        n_breaking=("is_breaking", "sum"),
        n_offspeed=("is_offspeed", "sum"),
    ).reset_index()


def compute_batter_skill_rolling(
    batter_skill_df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    if windows is None:
        windows = ROLLING_WINDOWS

    df = batter_skill_df.copy()
    df.sort_values(["batter", "game_date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("Computing batter skill rolling stats for windows %s", windows)

    rate_columns: dict[str, str | tuple[str, str]] = {
        "fly_ball_rate": ("n_fly_ball", "n_batted_ball"),
        "pull_rate": ("n_pull", "n_batted_ball"),
        "hard_hit_rate": ("n_hard_hit", "n_batted_ball"),
        "sweet_spot_pct": ("n_sweet_spot", "n_batted_ball"),
        "avg_xslg": "avg_xslg",
        "avg_xwoba": "avg_xwoba",
        "hr_rate_vs_fastball": ("n_hr_vs_fastball", "n_pa_vs_fastball"),
        "hr_rate_vs_breaking": ("n_hr_vs_breaking", "n_pa_vs_breaking"),
        "hr_rate_vs_offspeed": ("n_hr_vs_offspeed", "n_pa_vs_offspeed"),
    }

    df = compute_rolling_rates(df, "batter", rate_columns, windows, "batter")

    logger.info("Batter skill rolling stats complete")
    return df


def compute_pitcher_skill_rolling(
    pitcher_skill_df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    if windows is None:
        windows = ROLLING_WINDOWS

    df = pitcher_skill_df.copy()
    df.sort_values(["pitcher", "game_date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("Computing pitcher skill rolling stats for windows %s", windows)

    rate_columns: dict[str, str | tuple[str, str]] = {
        "gb_rate": ("n_ground_ball", "n_batted_ball"),
        "fb_rate": ("n_fly_ball", "n_batted_ball"),
        "fastball_pct": ("n_fastball", "n_pitches"),
        "breaking_pct": ("n_breaking", "n_pitches"),
        "offspeed_pct": ("n_offspeed", "n_pitches"),
    }

    df = compute_rolling_rates(df, "pitcher", rate_columns, windows, "pitcher")

    logger.info("Pitcher skill rolling stats complete")
    return df


def _extract_skill_rolling_cols(
    df: pd.DataFrame,
    prefix: str,
    feature_names: list[str],
) -> list[str]:
    skill_cols = []
    for col in df.columns:
        if not col.startswith(f"{prefix}_"):
            continue
        for feat in feature_names:
            if col.startswith(f"{prefix}_{feat}_"):
                skill_cols.append(col)
                break
    return skill_cols


def merge_skill_features_onto_pas(
    pa_df: pd.DataFrame,
    batter_skill_df: pd.DataFrame,
    pitcher_skill_df: pd.DataFrame,
) -> pd.DataFrame:
    batter_cols = _extract_skill_rolling_cols(
        batter_skill_df, "batter", BATTER_SKILL_FEATURES
    )
    batter_merge = ["batter", "game_pk", "game_date"] + batter_cols
    batter_subset = batter_skill_df[
        [c for c in batter_merge if c in batter_skill_df.columns]
    ].copy()

    pitcher_cols = _extract_skill_rolling_cols(
        pitcher_skill_df, "pitcher", PITCHER_SKILL_FEATURES
    )
    pitcher_merge = ["pitcher", "game_pk", "game_date"] + pitcher_cols
    pitcher_subset = pitcher_skill_df[
        [c for c in pitcher_merge if c in pitcher_skill_df.columns]
    ].copy()

    merged = pa_df.merge(
        batter_subset,
        on=["batter", "game_pk", "game_date"],
        how="left",
    )
    batter_matched = merged[batter_cols[0]].notna().sum() if batter_cols else 0
    logger.info(
        "Batter skill merge: %d / %d PAs matched (%.1f%%)",
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
        "Pitcher skill merge: %d / %d PAs matched (%.1f%%)",
        pitcher_matched,
        len(pa_df),
        100.0 * pitcher_matched / max(len(pa_df), 1),
    )

    return merged
