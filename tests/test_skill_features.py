import numpy as np
import pandas as pd
import pytest

from src.features.skill_features import (
    _classify_pitch_category,
    _compute_spray_angle,
    _is_pull,
    aggregate_batter_skill_stats,
    aggregate_pitcher_skill_stats,
    compute_batter_skill_rolling,
    compute_pitcher_skill_rolling,
    merge_skill_features_onto_pas,
)


def _make_raw_pitch_row(**overrides):
    base = {
        "batter": 100,
        "pitcher": 200,
        "game_pk": 1,
        "game_date": "2024-06-01",
        "events": np.nan,
        "stand": "R",
        "pitch_type": "FF",
        "bb_type": np.nan,
        "launch_speed": np.nan,
        "launch_angle": np.nan,
        "hc_x": np.nan,
        "hc_y": np.nan,
        "estimated_slg_using_speedangle": np.nan,
        "estimated_woba_using_speedangle": np.nan,
    }
    base.update(overrides)
    return base


class TestAggregateBatterSkillStats:

    def test_batted_ball_counts_correct(self):
        rows = [
            _make_raw_pitch_row(events="single", bb_type="fly_ball"),
            _make_raw_pitch_row(events="field_out", bb_type="fly_ball"),
            _make_raw_pitch_row(events="field_out", bb_type="ground_ball"),
            _make_raw_pitch_row(events="field_out", bb_type="line_drive"),
        ]
        raw_df = pd.DataFrame(rows)

        result = aggregate_batter_skill_stats(raw_df)

        row = result[(result["batter"] == 100) & (result["game_pk"] == 1)].iloc[0]
        assert row["n_fly_ball"] == 2
        assert row["n_batted_ball"] == 4

    def test_hard_hit_and_sweet_spot_counts(self):
        rows = [
            _make_raw_pitch_row(events="single", bb_type="line_drive", launch_speed=100, launch_angle=15),
            _make_raw_pitch_row(events="single", bb_type="line_drive", launch_speed=96, launch_angle=5),
            _make_raw_pitch_row(events="single", bb_type="line_drive", launch_speed=80, launch_angle=20),
            _make_raw_pitch_row(events="single", bb_type="line_drive", launch_speed=70, launch_angle=40),
        ]
        raw_df = pd.DataFrame(rows)

        result = aggregate_batter_skill_stats(raw_df)

        row = result.iloc[0]
        assert row["n_hard_hit"] == 2
        assert row["n_sweet_spot"] == 2

    def test_pull_rate_uses_spray_angle_and_handedness(self):
        rows = [
            _make_raw_pitch_row(
                batter=101, events="single", bb_type="fly_ball", stand="R",
                hc_x=60, hc_y=150,
            ),
            _make_raw_pitch_row(
                batter=102, events="single", bb_type="fly_ball", stand="L",
                hc_x=190, hc_y=150,
            ),
            _make_raw_pitch_row(
                batter=103, events="single", bb_type="fly_ball", stand="R",
                hc_x=125, hc_y=100,
            ),
        ]
        raw_df = pd.DataFrame(rows)

        result = aggregate_batter_skill_stats(raw_df)

        rhb_pull = result[result["batter"] == 101]["n_pull"].iloc[0]
        lhb_pull = result[result["batter"] == 102]["n_pull"].iloc[0]
        center = result[result["batter"] == 103]["n_pull"].iloc[0]
        assert rhb_pull == 1
        assert lhb_pull == 1
        assert center == 0

    def test_pitch_type_pa_k_counts(self):
        rows = [
            _make_raw_pitch_row(events="strikeout", pitch_type="FF", bb_type="fly_ball"),
            _make_raw_pitch_row(events="single", pitch_type="FF", bb_type="line_drive"),
            _make_raw_pitch_row(events="strikeout", pitch_type="SL", bb_type="fly_ball"),
            _make_raw_pitch_row(events="field_out", pitch_type="CH", bb_type="ground_ball"),
        ]
        raw_df = pd.DataFrame(rows)

        result = aggregate_batter_skill_stats(raw_df)

        row = result.iloc[0]
        assert row["n_pa_vs_fastball"] == 2
        assert row["n_k_vs_fastball"] == 1
        assert row["n_pa_vs_breaking"] == 1
        assert row["n_k_vs_breaking"] == 1
        assert row["n_pa_vs_offspeed"] == 1
        assert row["n_k_vs_offspeed"] == 0


class TestAggregatePitcherSkillStats:

    def test_batted_ball_profile_correct(self):
        rows = [
            _make_raw_pitch_row(events="field_out", bb_type="ground_ball"),
            _make_raw_pitch_row(events="field_out", bb_type="ground_ball"),
            _make_raw_pitch_row(events="single", bb_type="fly_ball"),
            _make_raw_pitch_row(events="single", bb_type="line_drive"),
        ]
        raw_df = pd.DataFrame(rows)

        result = aggregate_pitcher_skill_stats(raw_df)

        row = result[(result["pitcher"] == 200) & (result["game_pk"] == 1)].iloc[0]
        assert row["n_ground_ball"] == 2
        assert row["n_fly_ball"] == 1
        assert row["n_batted_ball"] == 4

    def test_pitch_usage_counts_all_pitches(self):
        rows = [
            _make_raw_pitch_row(pitch_type="FF"),
            _make_raw_pitch_row(pitch_type="FF"),
            _make_raw_pitch_row(pitch_type="SL"),
            _make_raw_pitch_row(pitch_type="SL"),
            _make_raw_pitch_row(pitch_type="CH"),
            _make_raw_pitch_row(events="strikeout", pitch_type="FF"),
        ]
        raw_df = pd.DataFrame(rows)

        result = aggregate_pitcher_skill_stats(raw_df)

        row = result[(result["pitcher"] == 200) & (result["game_pk"] == 1)].iloc[0]
        assert row["n_pitches"] == 6
        assert row["n_fastball"] == 3
        assert row["n_breaking"] == 2
        assert row["n_offspeed"] == 1


class TestComputeSkillRolling:

    def test_batter_rolling_columns_created(self):
        dates = pd.date_range("2024-06-01", periods=8, freq="D")
        batter_skill_df = pd.DataFrame({
            "batter": [100] * 8,
            "game_pk": list(range(1, 9)),
            "game_date": dates,
            "n_batted_ball": [4] * 8,
            "n_fly_ball": [2] * 8,
            "n_pull": [1] * 8,
            "n_hard_hit": [2] * 8,
            "n_sweet_spot": [1] * 8,
            "avg_xslg": [0.5] * 8,
            "avg_xwoba": [0.4] * 8,
            "n_k_vs_fastball": [1] * 8,
            "n_pa_vs_fastball": [3] * 8,
            "n_k_vs_breaking": [0] * 8,
            "n_pa_vs_breaking": [2] * 8,
            "n_k_vs_offspeed": [0] * 8,
            "n_pa_vs_offspeed": [1] * 8,
        })

        result = compute_batter_skill_rolling(batter_skill_df, windows=[5])

        expected_cols = [
            "batter_fly_ball_rate_5g",
            "batter_pull_rate_5g",
            "batter_hard_hit_rate_5g",
            "batter_sweet_spot_pct_5g",
            "batter_avg_xslg_5g",
            "batter_avg_xwoba_5g",
            "batter_k_rate_vs_fastball_5g",
            "batter_k_rate_vs_breaking_5g",
            "batter_k_rate_vs_offspeed_5g",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_pitcher_rolling_columns_created(self):
        dates = pd.date_range("2024-06-01", periods=8, freq="D")
        pitcher_skill_df = pd.DataFrame({
            "pitcher": [200] * 8,
            "game_pk": list(range(1, 9)),
            "game_date": dates,
            "n_batted_ball": [5] * 8,
            "n_ground_ball": [2] * 8,
            "n_fly_ball": [2] * 8,
            "n_pitches": [20] * 8,
            "n_fastball": [10] * 8,
            "n_breaking": [6] * 8,
            "n_offspeed": [4] * 8,
        })

        result = compute_pitcher_skill_rolling(pitcher_skill_df, windows=[5])

        expected_cols = [
            "pitcher_gb_rate_5g",
            "pitcher_fb_rate_5g",
            "pitcher_fastball_pct_5g",
            "pitcher_breaking_pct_5g",
            "pitcher_offspeed_pct_5g",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"


class TestMergeSkillFeaturesOntoPas:

    def _make_rolling_batter_df(self):
        batter_cols = {}
        for feat in [
            "fly_ball_rate", "pull_rate", "hard_hit_rate", "sweet_spot_pct",
            "avg_xslg", "avg_xwoba",
            "k_rate_vs_fastball", "k_rate_vs_breaking", "k_rate_vs_offspeed",
        ]:
            for w in [15, 50]:
                batter_cols[f"batter_{feat}_{w}g"] = [0.5]
        return pd.DataFrame({
            "batter": [100],
            "game_pk": [1],
            "game_date": pd.to_datetime(["2024-06-01"]),
            **batter_cols,
        })

    def _make_rolling_pitcher_df(self):
        pitcher_cols = {}
        for feat in ["gb_rate", "fb_rate", "fastball_pct", "breaking_pct", "offspeed_pct"]:
            for w in [15, 50]:
                pitcher_cols[f"pitcher_{feat}_{w}g"] = [0.5]
        return pd.DataFrame({
            "pitcher": [200],
            "game_pk": [1],
            "game_date": pd.to_datetime(["2024-06-01"]),
            **pitcher_cols,
        })

    def test_all_skill_columns_present_after_merge(self):
        pa_df = pd.DataFrame({
            "batter": [100],
            "pitcher": [200],
            "game_pk": [1],
            "game_date": pd.to_datetime(["2024-06-01"]),
            "events": ["home_run"],
        })
        batter_rolling = self._make_rolling_batter_df()
        pitcher_rolling = self._make_rolling_pitcher_df()

        result = merge_skill_features_onto_pas(pa_df, batter_rolling, pitcher_rolling)

        n_batter_skill_cols = 9 * 2
        n_pitcher_skill_cols = 5 * 2
        expected_total = n_batter_skill_cols + n_pitcher_skill_cols
        skill_cols = [
            c for c in result.columns
            if (c.startswith("batter_") and c.endswith("g"))
            or (c.startswith("pitcher_") and c.endswith("g"))
        ]
        assert len(skill_cols) == expected_total

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

        result = merge_skill_features_onto_pas(pa_df, batter_rolling, pitcher_rolling)

        assert len(result) == len(pa_df)


class TestSprayAngleAndPull:

    def test_spray_angle_center(self):
        hc_x = pd.Series([125.42])
        hc_y = pd.Series([150.0])

        result = _compute_spray_angle(hc_x, hc_y)

        assert result.iloc[0] == pytest.approx(0.0, abs=0.01)

    def test_rhb_pull_to_left_field(self):
        hc_x = pd.Series([60.0])
        hc_y = pd.Series([150.0])
        stand = pd.Series(["R"])

        spray_angle = _compute_spray_angle(hc_x, hc_y)
        result = _is_pull(stand, spray_angle)

        assert result.iloc[0] == True

    def test_lhb_pull_to_right_field(self):
        hc_x = pd.Series([190.0])
        hc_y = pd.Series([150.0])
        stand = pd.Series(["L"])

        spray_angle = _compute_spray_angle(hc_x, hc_y)
        result = _is_pull(stand, spray_angle)

        assert result.iloc[0] == True


class TestPitchClassification:

    def test_classifies_known_types(self):
        pitch_types = pd.Series(["FF", "SL", "CH"])

        result = _classify_pitch_category(pitch_types)

        assert result.iloc[0] == "fastball"
        assert result.iloc[1] == "breaking"
        assert result.iloc[2] == "offspeed"

    def test_unknown_type_is_nan(self):
        pitch_types = pd.Series(["XX"])

        result = _classify_pitch_category(pitch_types)

        assert pd.isna(result.iloc[0])
