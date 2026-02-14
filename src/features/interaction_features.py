import logging

import pandas as pd

logger = logging.getLogger(__name__)

_INTERACTIONS: list[tuple[str, str, str]] = [
    ("ix_batter_k_x_pitcher_k", "batter_k_rate_50g", "pitcher_k_rate_50g"),
    ("ix_whiff_x_swstr", "batter_whiff_rate_50g", "pitcher_swstr_rate_50g"),
    ("ix_k_vs_breaking_x_brk_pct", "batter_k_rate_vs_breaking_50g", "pitcher_breaking_pct_50g"),
    ("ix_chase_x_chase_induced", "batter_chase_rate_50g", "pitcher_chase_rate_induced_50g"),
    ("ix_platoon_x_pitcher_k", "platoon_advantage", "pitcher_k_rate_50g"),
    ("ix_batter_k_x_velo", "batter_k_rate_50g", "pitcher_avg_fastball_velo_50g"),
    ("ix_zone_contact_x_zone_rate", "batter_zone_contact_rate_50g", "pitcher_zone_rate_50g"),
    ("ix_batter_k_x_park_k", "batter_k_rate_50g", "park_k_factor"),
    ("ix_chase_x_offspeed", "batter_chase_rate_50g", "pitcher_offspeed_pct_50g"),
]

_VELO_CENTER = 90.0
_VELO_SCALE = 10.0


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

        if name == "ix_batter_k_x_velo":
            b = (b - _VELO_CENTER) / _VELO_SCALE

        df[name] = a * b
        added += 1

    logger.info("Added %d interaction features", added)
    return df
