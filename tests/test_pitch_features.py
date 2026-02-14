import numpy as np
import pandas as pd
import pytest

from src.features.pitch_features import (
    _classify_pitch_outcomes,
    aggregate_batter_pitch_stats,
    aggregate_pitcher_pitch_stats,
    compute_batter_pitch_rolling,
    compute_pitcher_pitch_rolling,
    merge_pitch_features_onto_pas,
)


def _make_raw_pitch_row(**overrides):
    base = {
        "batter": 100,
        "pitcher": 200,
        "game_pk": 1,
        "game_date": "2024-06-01",
        "events": np.nan,
        "description": "ball",
        "zone": 14,
        "pitch_type": "FF",
        "release_speed": 95.0,
        "stand": "R",
    }
    base.update(overrides)
    return base


class TestClassifyPitchOutcomes:

    def test_swinging_strike_classified_correctly(self):
        rows = [_make_raw_pitch_row(description="swinging_strike", zone=5)]
        raw_df = pd.DataFrame(rows)

        result = _classify_pitch_outcomes(raw_df)

        row = result.iloc[0]
        assert row["is_swing"] == 1
        assert row["is_swinging_strike"] == 1

    def test_called_strike_classified_correctly(self):
        rows = [_make_raw_pitch_row(description="called_strike", zone=3)]
        raw_df = pd.DataFrame(rows)

        result = _classify_pitch_outcomes(raw_df)

        row = result.iloc[0]
        assert row["is_called_strike"] == 1
        assert row["is_swing"] == 0

    def test_chase_requires_swing_and_out_of_zone(self):
        chase_row = _make_raw_pitch_row(description="foul", zone=12)
        no_chase_row = _make_raw_pitch_row(description="foul", zone=5)
        raw_df = pd.DataFrame([chase_row, no_chase_row])

        result = _classify_pitch_outcomes(raw_df)

        assert result.iloc[0]["is_chase"] == 1
        assert result.iloc[1]["is_chase"] == 0


class TestAggregateBatterPitchStats:

    def test_counts_swing_discipline_correctly(self):
        rows = [
            _make_raw_pitch_row(description="swinging_strike", zone=5),
            _make_raw_pitch_row(description="foul", zone=12),
            _make_raw_pitch_row(description="called_strike", zone=3),
            _make_raw_pitch_row(description="ball", zone=14),
            _make_raw_pitch_row(description="hit_into_play", zone=5),
        ]
        raw_df = pd.DataFrame(rows)

        result = aggregate_batter_pitch_stats(raw_df)

        row = result[(result["batter"] == 100) & (result["game_pk"] == 1)].iloc[0]
        assert row["n_swings"] == 3
        assert row["n_swinging_strikes"] == 1
        assert row["n_chases"] == 1


class TestAggregatePitcherPitchStats:

    def test_counts_pitch_stats_correctly(self):
        rows = [
            _make_raw_pitch_row(description="swinging_strike", zone=5, pitch_type="FF", release_speed=96.0),
            _make_raw_pitch_row(description="called_strike", zone=3, pitch_type="FF", release_speed=94.0),
            _make_raw_pitch_row(description="foul", zone=12, pitch_type="SL", release_speed=85.0),
            _make_raw_pitch_row(description="ball", zone=14, pitch_type="FF", release_speed=97.0),
        ]
        raw_df = pd.DataFrame(rows)

        result = aggregate_pitcher_pitch_stats(raw_df)

        row = result[(result["pitcher"] == 200) & (result["game_pk"] == 1)].iloc[0]
        assert row["n_pitches_thrown"] == 4
        assert row["n_swinging_strikes_induced"] == 1
        assert row["avg_fastball_velo"] == pytest.approx((96.0 + 94.0 + 97.0) / 3, abs=0.01)


class TestComputeBatterPitchRolling:

    def test_rolling_columns_created(self):
        dates = pd.date_range("2024-06-01", periods=8, freq="D")
        batter_agg_df = pd.DataFrame({
            "batter": [100] * 8,
            "game_pk": list(range(1, 9)),
            "game_date": dates,
            "n_pitches_seen": [20] * 8,
            "n_swings": [10] * 8,
            "n_swinging_strikes": [3] * 8,
            "n_called_strikes": [4] * 8,
            "n_chases": [2] * 8,
            "n_out_of_zone_pitches": [8] * 8,
            "n_zone_swings": [6] * 8,
            "n_zone_contacts": [5] * 8,
            "n_pitches_taken": [10] * 8,
        })

        result = compute_batter_pitch_rolling(batter_agg_df, windows=[5])

        expected_cols = [
            "batter_swing_rate_5g",
            "batter_whiff_rate_5g",
            "batter_chase_rate_5g",
            "batter_zone_contact_rate_5g",
            "batter_called_strike_rate_5g",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"


class TestComputePitcherPitchRolling:

    def test_rolling_columns_created(self):
        dates = pd.date_range("2024-06-01", periods=8, freq="D")
        pitcher_agg_df = pd.DataFrame({
            "pitcher": [200] * 8,
            "game_pk": list(range(1, 9)),
            "game_date": dates,
            "n_pitches_thrown": [25] * 8,
            "n_swinging_strikes_induced": [4] * 8,
            "n_chases_induced": [3] * 8,
            "n_out_of_zone_thrown": [10] * 8,
            "n_in_zone_thrown": [15] * 8,
            "avg_fastball_velo": [95.0] * 8,
        })

        result = compute_pitcher_pitch_rolling(pitcher_agg_df, windows=[5])

        expected_cols = [
            "pitcher_swstr_rate_5g",
            "pitcher_chase_rate_induced_5g",
            "pitcher_zone_rate_5g",
            "pitcher_avg_fastball_velo_5g",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"


class TestMergePitchFeaturesOntoPas:

    def _make_rolling_batter_df(self):
        batter_cols = {}
        for feat in ["swing_rate", "whiff_rate", "chase_rate", "zone_contact_rate", "called_strike_rate"]:
            for w in [15, 50]:
                batter_cols[f"batter_{feat}_{w}g"] = [0.25]
        return pd.DataFrame({
            "batter": [100],
            "game_pk": [1],
            "game_date": pd.to_datetime(["2024-06-01"]),
            **batter_cols,
        })

    def _make_rolling_pitcher_df(self):
        pitcher_cols = {}
        for feat in ["swstr_rate", "chase_rate_induced", "zone_rate", "avg_fastball_velo"]:
            for w in [15, 50]:
                pitcher_cols[f"pitcher_{feat}_{w}g"] = [0.30]
        return pd.DataFrame({
            "pitcher": [200],
            "game_pk": [1],
            "game_date": pd.to_datetime(["2024-06-01"]),
            **pitcher_cols,
        })

    def test_row_count_preserved(self):
        pa_df = pd.DataFrame({
            "batter": [100, 100, 101],
            "pitcher": [200, 201, 200],
            "game_pk": [1, 1, 1],
            "game_date": pd.to_datetime(["2024-06-01"] * 3),
            "events": ["single", "field_out", "home_run"],
        })
        batter_rolling = self._make_rolling_batter_df()
        pitcher_rolling = self._make_rolling_pitcher_df()

        result = merge_pitch_features_onto_pas(pa_df, batter_rolling, pitcher_rolling)

        assert len(result) == len(pa_df)
