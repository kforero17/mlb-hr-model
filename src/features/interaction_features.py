import logging

import pandas as pd

logger = logging.getLogger(__name__)

_INTERACTIONS: list[tuple[str, str, str]] = [
    ("ix_barrel_x_fb_rate", "batter_barrel_rate_50g", "pitcher_fb_rate_50g"),
    ("ix_hr_vs_fb_x_fb_pct", "batter_hr_rate_vs_fastball_50g", "pitcher_fastball_pct_50g"),
    ("ix_hard_hit_x_hr_allowed", "batter_hard_hit_rate_50g", "pitcher_hr_allowed_rate_50g"),
    ("ix_pull_x_park_hand", "batter_pull_rate_50g", "park_hr_factor_handedness"),
    ("ix_barrel_x_elevation", "batter_barrel_rate_50g", "elevation_ft"),
    ("ix_hard_hit_x_wind_out", "batter_hard_hit_rate_50g", "wind_out_to_cf"),
    ("ix_barrel_x_air_density", "batter_barrel_rate_50g", "air_density_index"),
    ("ix_hr_streak_x_barrel", "batter_hr_streak_50g", "batter_barrel_rate_50g"),
    ("ix_platoon_x_hr_rate", "platoon_advantage", "batter_hr_rate_50g"),
]

_ELEVATION_DIVISOR = 5280.0


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    added = 0
    for name, col_a, col_b in _INTERACTIONS:
        if col_a not in df.columns or col_b not in df.columns:
            logger.warning(
                "Skipping %s: missing %s",
                name,
                col_a if col_a not in df.columns else col_b,
            )
            continue

        a = df[col_a]
        b = df[col_b]

        if name == "ix_barrel_x_elevation":
            b = b / _ELEVATION_DIVISOR
        elif name == "ix_barrel_x_air_density":
            b = 1.0 - b

        df[name] = a * b
        added += 1

    logger.info("Added %d interaction features", added)
    return df
