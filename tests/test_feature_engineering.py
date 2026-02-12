import numpy as np
import pandas as pd
import pytest

from config.feature_config import AL_TEAMS, HIT_EVENTS, PLATE_APPEARANCE_EVENTS
from src.features.feature_engineering import (
    add_game_context_features,
    add_platoon_features,
    aggregate_to_batter_game,
    build_pa_rows,
    compute_batter_rolling_stats,
    compute_pitcher_rolling_stats,
    merge_rolling_stats_onto_pas,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pitch_level_data():
    rows = []

    # Batter 1, Game 1 — 4 PAs across 6 pitches
    base = dict(
        batter=100, pitcher=900, game_pk=1, game_date="2023-06-01",
        home_team="NYY", away_team="BOS", stand="R", p_throws="L",
        inning_topbot="Bot",
    )
    rows.append({**base, "at_bat_number": 1, "events": np.nan,
                 "launch_speed": np.nan, "launch_angle": np.nan,
                 "description": "ball",
                 "inning": 1, "outs_when_up": 0, "home_score": 0, "away_score": 0})
    rows.append({**base, "at_bat_number": 1, "events": "home_run",
                 "launch_speed": 105.0, "launch_angle": 28.0,
                 "description": "hit_into_play",
                 "inning": 1, "outs_when_up": 0, "home_score": 0, "away_score": 0})
    rows.append({**base, "at_bat_number": 2, "events": np.nan,
                 "launch_speed": np.nan, "launch_angle": np.nan,
                 "description": "called_strike",
                 "inning": 3, "outs_when_up": 1, "home_score": 1, "away_score": 0})
    rows.append({**base, "at_bat_number": 2, "events": "strikeout",
                 "launch_speed": np.nan, "launch_angle": np.nan,
                 "description": "swinging_strike",
                 "inning": 3, "outs_when_up": 1, "home_score": 1, "away_score": 0})
    rows.append({**base, "at_bat_number": 3, "events": "single",
                 "launch_speed": 92.0, "launch_angle": 12.0,
                 "description": "hit_into_play",
                 "inning": 5, "outs_when_up": 0, "home_score": 2, "away_score": 1})
    rows.append({**base, "at_bat_number": 4, "events": "walk",
                 "launch_speed": np.nan, "launch_angle": np.nan,
                 "description": "ball",
                 "inning": 7, "outs_when_up": 2, "home_score": 3, "away_score": 1})

    # Batter 1, Game 2 — 2 PAs
    base2 = {**base, "game_pk": 2, "game_date": "2023-06-02", "pitcher": 901}
    rows.append({**base2, "at_bat_number": 1, "events": "field_out",
                 "launch_speed": 88.0, "launch_angle": 45.0,
                 "description": "hit_into_play",
                 "inning": 2, "outs_when_up": 0, "home_score": 0, "away_score": 1})
    rows.append({**base2, "at_bat_number": 2, "events": np.nan,
                 "launch_speed": np.nan, "launch_angle": np.nan,
                 "description": "ball",
                 "inning": 4, "outs_when_up": 1, "home_score": 0, "away_score": 2})
    rows.append({**base2, "at_bat_number": 2, "events": "field_out",
                 "launch_speed": 75.0, "launch_angle": -5.0,
                 "description": "hit_into_play",
                 "inning": 4, "outs_when_up": 1, "home_score": 0, "away_score": 2})

    # Batter 2, Game 1 — 3 PAs
    base3 = dict(
        batter=200, pitcher=900, game_pk=1, game_date="2023-06-01",
        home_team="NYY", away_team="BOS", stand="L", p_throws="L",
        inning_topbot="Top",
    )
    rows.append({**base3, "at_bat_number": 1, "events": "home_run",
                 "launch_speed": 110.0, "launch_angle": 27.0,
                 "description": "hit_into_play",
                 "inning": 1, "outs_when_up": 2, "home_score": 0, "away_score": 0})
    rows.append({**base3, "at_bat_number": 2, "events": "walk",
                 "launch_speed": np.nan, "launch_angle": np.nan,
                 "description": "ball",
                 "inning": 3, "outs_when_up": 0, "home_score": 0, "away_score": 1})
    rows.append({**base3, "at_bat_number": 3, "events": "field_out",
                 "launch_speed": 95.0, "launch_angle": 10.0,
                 "description": "hit_into_play",
                 "inning": 6, "outs_when_up": 1, "home_score": 1, "away_score": 2})

    # Batter 2, Game 2 — 3 PAs
    base4 = {**base3, "game_pk": 2, "game_date": "2023-06-02", "pitcher": 901}
    rows.append({**base4, "at_bat_number": 1, "events": "single",
                 "launch_speed": 90.0, "launch_angle": 15.0,
                 "description": "hit_into_play",
                 "inning": 1, "outs_when_up": 1, "home_score": 0, "away_score": 0})
    rows.append({**base4, "at_bat_number": 2, "events": np.nan,
                 "launch_speed": np.nan, "launch_angle": np.nan,
                 "description": "called_strike",
                 "inning": 4, "outs_when_up": 0, "home_score": 2, "away_score": 1})
    rows.append({**base4, "at_bat_number": 2, "events": "strikeout",
                 "launch_speed": np.nan, "launch_angle": np.nan,
                 "description": "swinging_strike",
                 "inning": 4, "outs_when_up": 0, "home_score": 2, "away_score": 1})
    rows.append({**base4, "at_bat_number": 3, "events": "field_out",
                 "launch_speed": 80.0, "launch_angle": 35.0,
                 "description": "hit_into_play",
                 "inning": 7, "outs_when_up": 2, "home_score": 3, "away_score": 2})

    return pd.DataFrame(rows)


@pytest.fixture
def batter_game_df():
    dates = pd.date_range("2023-06-01", periods=10, freq="D")
    return pd.DataFrame({
        "batter": [100] * 10,
        "game_pk": list(range(1, 11)),
        "game_date": dates,
        "hit_hr": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        "n_pa": [4, 4, 3, 4, 5, 3, 4, 4, 4, 3],
        "n_hits": [1, 2, 0, 1, 2, 1, 0, 1, 2, 0],
        "n_hr": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        "n_k": [1, 0, 2, 1, 0, 1, 2, 1, 0, 1],
        "n_bb": [1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        "n_barrel": [0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        "n_ab": [3, 4, 2, 4, 4, 3, 4, 3, 4, 2],
        "avg_exit_velo": [90.0, 95.0, 88.0, 92.0, 100.0, 85.0, 87.0, 91.0, 98.0, 86.0],
        "avg_launch_angle": [12.0, 25.0, 10.0, 15.0, 28.0, 8.0, 14.0, 18.0, 26.0, 11.0],
    })


# ---------------------------------------------------------------------------
# aggregate_to_batter_game
# ---------------------------------------------------------------------------

class TestAggregateToBatterGame:

    def test_produces_one_row_per_batter_per_game(self, pitch_level_data):
        result = aggregate_to_batter_game(pitch_level_data)

        assert len(result) == 4  # 2 batters x 2 games

    def test_hr_flag_is_set_correctly(self, pitch_level_data):
        result = aggregate_to_batter_game(pitch_level_data)

        b1_g1 = result[(result["batter"] == 100) & (result["game_pk"] == 1)]
        b1_g2 = result[(result["batter"] == 100) & (result["game_pk"] == 2)]

        assert b1_g1["hit_hr"].iloc[0] == 1
        assert b1_g2["hit_hr"].iloc[0] == 0

    def test_counting_stats_are_accurate(self, pitch_level_data):
        result = aggregate_to_batter_game(pitch_level_data)
        b1_g1 = result[(result["batter"] == 100) & (result["game_pk"] == 1)].iloc[0]

        assert b1_g1["n_pa"] == 4  # HR + K + single + walk
        assert b1_g1["n_hr"] == 1
        assert b1_g1["n_hits"] == 2  # HR + single
        assert b1_g1["n_k"] == 1
        assert b1_g1["n_bb"] == 1

    def test_metadata_columns_present(self, pitch_level_data):
        result = aggregate_to_batter_game(pitch_level_data)

        for col in ["home_team", "away_team", "stand", "p_throws", "is_home"]:
            assert col in result.columns


# ---------------------------------------------------------------------------
# compute_batter_rolling_stats
# ---------------------------------------------------------------------------

class TestComputeBatterRollingStats:

    def test_rolling_columns_created_with_correct_naming(self, batter_game_df):
        result = compute_batter_rolling_stats(batter_game_df, windows=[5])

        assert "batter_hr_rate_5g" in result.columns
        assert "batter_avg_exit_velo_5g" in result.columns
        assert "batter_batting_avg_5g" in result.columns

    def test_first_rows_are_nan_to_prevent_leakage(self, batter_game_df):
        result = compute_batter_rolling_stats(batter_game_df, windows=[5])

        assert pd.isna(result["batter_hr_rate_5g"].iloc[0])

    def test_games_since_last_hr_is_present(self, batter_game_df):
        result = compute_batter_rolling_stats(batter_game_df, windows=[5])

        assert "games_since_last_hr" in result.columns
        assert pd.isna(result["games_since_last_hr"].iloc[0])


# ---------------------------------------------------------------------------
# compute_pitcher_rolling_stats
# ---------------------------------------------------------------------------

class TestComputePitcherRollingStats:

    def test_produces_pitcher_game_rows_with_rolling_columns(self, pitch_level_data):
        extra_games = []
        for i, gdate in enumerate(pd.date_range("2023-06-03", periods=8, freq="D")):
            extra_games.append({
                "batter": 300 + i, "pitcher": 900, "game_pk": 10 + i,
                "game_date": str(gdate.date()),
                "home_team": "NYY", "away_team": "BOS",
                "stand": "R", "p_throws": "R",
                "inning_topbot": "Top", "at_bat_number": 1,
                "events": "field_out",
                "launch_speed": 88.0, "launch_angle": 15.0,
                "description": "hit_into_play",
            })

        combined = pd.concat([pitch_level_data, pd.DataFrame(extra_games)], ignore_index=True)
        result = compute_pitcher_rolling_stats(combined, windows=[5])

        assert "pitcher_hr_allowed_rate_5g" in result.columns
        assert "pitcher_avg_exit_velo_against_5g" in result.columns

        pitcher_rows = result[result["pitcher"] == 900]
        assert len(pitcher_rows) >= 2


# ---------------------------------------------------------------------------
# add_platoon_features
# ---------------------------------------------------------------------------

class TestAddPlatoonFeatures:

    def test_platoon_column_format(self):
        df = pd.DataFrame({"stand": ["L", "R", "L", "R"], "p_throws": ["R", "L", "L", "R"]})

        result = add_platoon_features(df)

        assert list(result["platoon"]) == ["L_vs_R", "R_vs_L", "L_vs_L", "R_vs_R"]

    def test_platoon_advantage_for_opposite_hand(self):
        df = pd.DataFrame({"stand": ["L", "R", "L", "R"], "p_throws": ["R", "L", "L", "R"]})

        result = add_platoon_features(df)

        assert list(result["platoon_advantage"]) == [1, 1, 0, 0]


# ---------------------------------------------------------------------------
# add_game_context_features
# ---------------------------------------------------------------------------

class TestAddGameContextFeatures:

    def test_month_and_day_of_week_extracted(self):
        df = pd.DataFrame({
            "game_date": ["2023-07-04", "2023-09-15"],
            "home_team": ["NYY", "LAD"],
            "is_home": [1, 0],
        })

        result = add_game_context_features(df)

        assert list(result["month"]) == [7, 9]
        assert result["day_of_week"].iloc[0] == pd.Timestamp("2023-07-04").dayofweek

    def test_is_dh_based_on_al_teams(self):
        df = pd.DataFrame({
            "game_date": ["2023-06-01", "2023-06-01"],
            "home_team": ["NYY", "LAD"],
            "is_home": [1, 0],
        })

        result = add_game_context_features(df)

        assert result["is_dh"].iloc[0] == 1  # NYY is AL
        assert result["is_dh"].iloc[1] == 0  # LAD is NL


# ---------------------------------------------------------------------------
# build_pa_rows
# ---------------------------------------------------------------------------

class TestBuildPaRows:

    def test_returns_one_row_per_plate_appearance(self, pitch_level_data):
        result = build_pa_rows(pitch_level_data)

        assert len(result) == 12

    def test_preserves_batter_and_pitcher_columns(self, pitch_level_data):
        result = build_pa_rows(pitch_level_data)

        assert "batter" in result.columns
        assert "pitcher" in result.columns
        assert result["batter"].notna().all()
        assert result["pitcher"].notna().all()

    def test_adds_pa_context_features(self, pitch_level_data):
        result = build_pa_rows(pitch_level_data)

        for col in ["is_home", "score_diff", "runners_on_base", "pa_number_in_game"]:
            assert col in result.columns

    def test_runners_on_base_defaults_to_zero_without_base_columns(self, pitch_level_data):
        result = build_pa_rows(pitch_level_data)

        assert (result["runners_on_base"] == 0).all()


# ---------------------------------------------------------------------------
# merge_rolling_stats_onto_pas
# ---------------------------------------------------------------------------

class TestMergeRollingStatsOntoPas:

    def test_merges_batter_and_pitcher_stats_onto_pas(self):
        pa_df = pd.DataFrame({
            "batter": [100, 100, 100],
            "pitcher": [900, 900, 901],
            "game_pk": [1, 1, 1],
            "game_date": pd.to_datetime(["2023-06-01"] * 3),
            "is_hr": [0, 1, 0],
        })

        batter_rolling = pd.DataFrame({
            "batter": [100],
            "game_pk": [1],
            "game_date": pd.to_datetime(["2023-06-01"]),
            "batter_hr_rate_15g": [0.1],
        })

        pitcher_rolling = pd.DataFrame({
            "pitcher": [900, 901],
            "game_pk": [1, 1],
            "game_date": pd.to_datetime(["2023-06-01", "2023-06-01"]),
            "pitcher_hr_allowed_rate_15g": [0.05, 0.08],
        })

        result = merge_rolling_stats_onto_pas(pa_df, batter_rolling, pitcher_rolling)

        assert len(result) == 3
        assert (result["batter_hr_rate_15g"] == 0.1).all()
        assert list(result["pitcher_hr_allowed_rate_15g"]) == [0.05, 0.05, 0.08]
