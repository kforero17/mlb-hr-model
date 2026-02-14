import numpy as np
import pandas as pd
import pytest

from src.features.interaction_features import add_interaction_features


class TestAddInteractionFeatures:

    def _make_base_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "batter_k_rate_50g": [0.25, 0.18],
            "pitcher_k_rate_50g": [0.22, 0.30],
            "batter_whiff_rate_50g": [0.28, 0.20],
            "pitcher_swstr_rate_50g": [0.12, 0.15],
            "batter_k_rate_vs_breaking_50g": [0.30, 0.22],
            "pitcher_breaking_pct_50g": [0.35, 0.40],
            "batter_chase_rate_50g": [0.32, 0.25],
            "pitcher_chase_rate_induced_50g": [0.30, 0.28],
            "platoon_advantage": [1.0, 0.0],
            "pitcher_avg_fastball_velo_50g": [95.0, 90.0],
            "batter_zone_contact_rate_50g": [0.82, 0.88],
            "pitcher_zone_rate_50g": [0.48, 0.52],
            "park_k_factor": [1.05, 0.95],
            "pitcher_offspeed_pct_50g": [0.15, 0.20],
        })

    def test_adds_all_nine_features(self):
        df = self._make_base_df()

        result = add_interaction_features(df)

        ix_cols = [c for c in result.columns if c.startswith("ix_")]
        assert len(ix_cols) == 9

    def test_velo_interaction_normalizes(self):
        df = self._make_base_df()

        result = add_interaction_features(df)

        expected = 0.25 * (95.0 - 90.0) / 10.0
        assert result["ix_batter_k_x_velo"].iloc[0] == pytest.approx(expected)

    def test_nan_propagation(self):
        df = self._make_base_df()
        df.loc[0, "batter_k_rate_50g"] = np.nan

        result = add_interaction_features(df)

        assert pd.isna(result["ix_batter_k_x_pitcher_k"].iloc[0])
        assert not pd.isna(result["ix_batter_k_x_pitcher_k"].iloc[1])

    def test_skips_features_with_missing_columns(self):
        df = pd.DataFrame({
            "batter_k_rate_50g": [0.25],
            "pitcher_k_rate_50g": [0.22],
        })

        result = add_interaction_features(df)

        ix_cols = [c for c in result.columns if c.startswith("ix_")]
        assert len(ix_cols) == 1
        assert "ix_batter_k_x_pitcher_k" in ix_cols
