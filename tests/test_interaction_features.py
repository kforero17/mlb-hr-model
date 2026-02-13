import numpy as np
import pandas as pd
import pytest

from src.features.interaction_features import add_interaction_features


class TestAddInteractionFeatures:

    def _make_base_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "batter_barrel_rate_50g": [0.10, 0.20],
            "pitcher_fb_rate_50g": [0.40, 0.50],
            "batter_hr_rate_vs_fastball_50g": [0.05, 0.10],
            "pitcher_fastball_pct_50g": [0.60, 0.70],
            "batter_hard_hit_rate_50g": [0.35, 0.40],
            "pitcher_hr_allowed_rate_50g": [0.03, 0.04],
            "batter_pull_rate_50g": [0.45, 0.50],
            "park_hr_factor_handedness": [1.1, 0.9],
            "elevation_ft": [5280.0, 0.0],
            "wind_out_to_cf": [1.0, 0.0],
            "air_density_index": [0.9, 1.0],
            "batter_hr_streak_50g": [3.0, 0.0],
            "platoon_advantage": [1.0, 0.0],
            "batter_hr_rate_50g": [0.04, 0.02],
        })

    def test_adds_all_nine_features(self):
        df = self._make_base_df()

        result = add_interaction_features(df)

        ix_cols = [c for c in result.columns if c.startswith("ix_")]
        assert len(ix_cols) == 9

    def test_barrel_x_elevation_normalizes(self):
        df = self._make_base_df()

        result = add_interaction_features(df)

        expected = 0.10 * (5280.0 / 5280.0)
        assert result["ix_barrel_x_elevation"].iloc[0] == pytest.approx(expected)

    def test_barrel_x_air_density_inverts(self):
        df = self._make_base_df()

        result = add_interaction_features(df)

        expected = 0.10 * (1.0 - 0.9)
        assert result["ix_barrel_x_air_density"].iloc[0] == pytest.approx(expected)

    def test_nan_propagation(self):
        df = self._make_base_df()
        df.loc[0, "batter_barrel_rate_50g"] = np.nan

        result = add_interaction_features(df)

        assert pd.isna(result["ix_barrel_x_fb_rate"].iloc[0])
        assert not pd.isna(result["ix_barrel_x_fb_rate"].iloc[1])

    def test_skips_features_with_missing_columns(self):
        df = pd.DataFrame({
            "batter_barrel_rate_50g": [0.10],
            "pitcher_fb_rate_50g": [0.40],
        })

        result = add_interaction_features(df)

        ix_cols = [c for c in result.columns if c.startswith("ix_")]
        assert len(ix_cols) == 1
        assert "ix_barrel_x_fb_rate" in ix_cols
