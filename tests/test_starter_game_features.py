import numpy as np
import pandas as pd
import pytest

from src.features.starter_game_features import (
    aggregate_starter_game,
    compute_opp_lineup_handedness,
    compute_opposing_team_k_features,
    compute_pitcher_workload_rolling,
    compute_starter_interactions,
    identify_starters,
)


def _make_raw_row(**overrides):
    base = {
        "pitcher": 200,
        "batter": 100,
        "game_pk": 1,
        "game_date": "2024-06-01",
        "at_bat_number": 1,
        "inning": 1,
        "inning_topbot": "Top",
        "events": np.nan,
        "home_team": "NYY",
        "away_team": "BOS",
        "p_throws": "R",
        "stand": "R",
        "description": "called_strike",
        "zone": 5,
        "pitch_type": "FF",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# identify_starters
# ---------------------------------------------------------------------------

class TestIdentifyStarters:

    def test_identifies_home_and_away_starters(self):
        rows = [
            _make_raw_row(pitcher=200, inning_topbot="Top", at_bat_number=1, events="field_out"),
            _make_raw_row(pitcher=200, inning_topbot="Top", at_bat_number=2, events="single"),
            _make_raw_row(pitcher=300, inning_topbot="Bot", at_bat_number=3, events="strikeout"),
            _make_raw_row(pitcher=300, inning_topbot="Bot", at_bat_number=4, events="walk"),
        ]
        raw_df = pd.DataFrame(rows)

        result = identify_starters(raw_df)

        assert len(result) == 2
        home_starter = result[result["is_home"] == 1]
        away_starter = result[result["is_home"] == 0]
        assert home_starter["pitcher"].iloc[0] == 200
        assert away_starter["pitcher"].iloc[0] == 300

    def test_handles_multiple_games(self):
        rows = [
            _make_raw_row(game_pk=1, pitcher=200, inning_topbot="Top", at_bat_number=1, events="field_out"),
            _make_raw_row(game_pk=1, pitcher=300, inning_topbot="Bot", at_bat_number=2, events="single"),
            _make_raw_row(game_pk=2, pitcher=400, inning_topbot="Top", at_bat_number=1,
                          events="field_out", game_date="2024-06-02", home_team="LAD", away_team="SF"),
            _make_raw_row(game_pk=2, pitcher=500, inning_topbot="Bot", at_bat_number=2,
                          events="strikeout", game_date="2024-06-02", home_team="LAD", away_team="SF"),
        ]
        raw_df = pd.DataFrame(rows)

        result = identify_starters(raw_df)

        assert len(result) == 4

    def test_picks_first_pitcher_by_at_bat_number(self):
        rows = [
            _make_raw_row(pitcher=200, inning_topbot="Top", at_bat_number=2, events="field_out"),
            _make_raw_row(pitcher=201, inning_topbot="Top", at_bat_number=1, events="single"),
        ]
        raw_df = pd.DataFrame(rows)

        result = identify_starters(raw_df)

        assert result["pitcher"].iloc[0] == 201


# ---------------------------------------------------------------------------
# aggregate_starter_game
# ---------------------------------------------------------------------------

def _make_starter_game_raw(pitcher, game_pk, game_date, n_pa, n_k, home_team="NYY", away_team="BOS"):
    events_list = ["strikeout"] * n_k + ["field_out"] * (n_pa - n_k)
    rows = []
    for i, ev in enumerate(events_list):
        rows.append(_make_raw_row(
            pitcher=pitcher,
            batter=100 + i,
            game_pk=game_pk,
            game_date=game_date,
            at_bat_number=i + 1,
            inning=1 + i // 3,
            inning_topbot="Top",
            events=ev,
            home_team=home_team,
            away_team=away_team,
            description="swinging_strike" if ev == "strikeout" else "hit_into_play",
        ))
    return rows


class TestAggregateStarterGame:

    def test_counts_strikeouts_and_batters_faced(self):
        rows = _make_starter_game_raw(pitcher=200, game_pk=1, game_date="2024-06-01", n_pa=18, n_k=6)
        rows += [_make_raw_row(pitcher=200, game_pk=1, at_bat_number=0, inning_topbot="Top", events="field_out")]
        raw_df = pd.DataFrame(rows)

        starters_df = pd.DataFrame([{
            "pitcher": 200, "game_pk": 1, "game_date": pd.Timestamp("2024-06-01"),
            "home_team": "NYY", "away_team": "BOS", "p_throws": "R", "is_home": 1,
        }])

        result = aggregate_starter_game(raw_df, starters_df)

        assert len(result) == 1
        assert result["n_k_against"].iloc[0] == 6
        assert result["n_batters_faced"].iloc[0] == 19

    def test_filters_below_min_bf(self):
        rows = _make_starter_game_raw(pitcher=200, game_pk=1, game_date="2024-06-01", n_pa=10, n_k=3)
        raw_df = pd.DataFrame(rows)

        starters_df = pd.DataFrame([{
            "pitcher": 200, "game_pk": 1, "game_date": pd.Timestamp("2024-06-01"),
            "home_team": "NYY", "away_team": "BOS", "p_throws": "R", "is_home": 1,
        }])

        result = aggregate_starter_game(raw_df, starters_df)

        assert len(result) == 0

    def test_sets_opposing_team(self):
        rows = _make_starter_game_raw(pitcher=200, game_pk=1, game_date="2024-06-01", n_pa=18, n_k=5)
        raw_df = pd.DataFrame(rows)

        starters_df = pd.DataFrame([{
            "pitcher": 200, "game_pk": 1, "game_date": pd.Timestamp("2024-06-01"),
            "home_team": "NYY", "away_team": "BOS", "p_throws": "R", "is_home": 1,
        }])

        result = aggregate_starter_game(raw_df, starters_df)

        assert result["opposing_team"].iloc[0] == "BOS"


# ---------------------------------------------------------------------------
# compute_pitcher_workload_rolling
# ---------------------------------------------------------------------------

class TestComputePitcherWorkloadRolling:

    def _make_starter_df(self, n_games=10, pitcher=200):
        dates = pd.date_range("2024-04-01", periods=n_games, freq="5D")
        return pd.DataFrame({
            "pitcher": [pitcher] * n_games,
            "game_pk": list(range(1, n_games + 1)),
            "game_date": dates,
            "n_batters_faced": [25] * n_games,
            "n_pitches": [95] * n_games,
            "max_inning": [7] * n_games,
        })

    def test_produces_rolling_columns(self):
        df = self._make_starter_df(n_games=10)

        result = compute_pitcher_workload_rolling(df, windows=[5])

        assert "pitcher_avg_bf_5g" in result.columns
        assert "pitcher_avg_pitch_count_5g" in result.columns
        assert "pitcher_pct_6plus_innings_5g" in result.columns

    def test_first_row_is_nan_due_to_shift(self):
        df = self._make_starter_df(n_games=10)

        result = compute_pitcher_workload_rolling(df, windows=[5])

        assert pd.isna(result["pitcher_avg_bf_5g"].iloc[0])

    def test_days_since_prev_game(self):
        df = self._make_starter_df(n_games=4)

        result = compute_pitcher_workload_rolling(df, windows=[5])

        assert pd.isna(result["pitcher_days_since_prev_game"].iloc[0])
        assert result["pitcher_days_since_prev_game"].iloc[1] == 5

    def test_went_6plus_column_dropped(self):
        df = self._make_starter_df(n_games=6)

        result = compute_pitcher_workload_rolling(df, windows=[5])

        assert "went_6plus" not in result.columns


# ---------------------------------------------------------------------------
# compute_opposing_team_k_features
# ---------------------------------------------------------------------------

class TestComputeOpposingTeamKFeatures:

    def _make_team_raw(self, n_games=8):
        rows = []
        base_date = pd.Timestamp("2024-04-01")
        for g in range(n_games):
            gdate = (base_date + pd.Timedelta(days=g)).strftime("%Y-%m-%d")
            for i in range(20):
                rows.append(_make_raw_row(
                    game_pk=g + 1,
                    game_date=gdate,
                    batter=100 + i,
                    pitcher=200,
                    at_bat_number=i + 1,
                    inning=1 + i // 3,
                    inning_topbot="Top",
                    events="strikeout" if i < 5 else "field_out",
                    description="swinging_strike" if i < 3 else "called_strike",
                    zone=5 if i < 10 else 12,
                    home_team="NYY",
                    away_team="BOS",
                ))
        return pd.DataFrame(rows)

    def test_produces_rolling_team_columns(self):
        raw_df = self._make_team_raw(n_games=8)

        result = compute_opposing_team_k_features(raw_df, windows=[5])

        assert "opp_team_k_rate_5g" in result.columns
        assert "opp_team_whiff_rate_5g" in result.columns
        assert "opp_team_chase_rate_5g" in result.columns

    def test_first_rows_are_nan_due_to_shift(self):
        raw_df = self._make_team_raw(n_games=8)

        result = compute_opposing_team_k_features(raw_df, windows=[5])

        assert pd.isna(result["opp_team_k_rate_5g"].iloc[0])


# ---------------------------------------------------------------------------
# compute_starter_interactions
# ---------------------------------------------------------------------------

class TestComputeStarterInteractions:

    def test_creates_interactions_when_columns_present(self):
        df = pd.DataFrame({
            "pitcher_k_rate_50g": [0.25],
            "opp_team_k_rate_50g": [0.22],
            "pitcher_swstr_rate_50g": [0.12],
            "opp_team_whiff_rate_50g": [0.28],
            "park_k_factor": [1.05],
            "pitcher_avg_bf_50g": [24.0],
        })

        result = compute_starter_interactions(df)

        assert result["ix_pitcher_k_x_opp_team_k"].iloc[0] == pytest.approx(0.25 * 0.22)
        assert result["ix_pitcher_swstr_x_opp_whiff"].iloc[0] == pytest.approx(0.12 * 0.28)
        assert result["ix_pitcher_k_x_park_k"].iloc[0] == pytest.approx(0.25 * 1.05)
        assert result["ix_pitcher_bf_x_pitcher_k"].iloc[0] == pytest.approx(24.0 * 0.25)

    def test_skips_missing_columns(self):
        df = pd.DataFrame({
            "pitcher_k_rate_50g": [0.25],
            "park_k_factor": [1.05],
        })

        result = compute_starter_interactions(df)

        assert "ix_pitcher_k_x_opp_team_k" not in result.columns
        assert "ix_pitcher_k_x_park_k" in result.columns

    def test_does_not_modify_original(self):
        df = pd.DataFrame({
            "pitcher_k_rate_50g": [0.25],
            "opp_team_k_rate_50g": [0.22],
        })
        original_cols = list(df.columns)

        compute_starter_interactions(df)

        assert list(df.columns) == original_cols


# ---------------------------------------------------------------------------
# compute_opp_lineup_handedness
# ---------------------------------------------------------------------------

class TestComputeOppLineupHandedness:

    def test_computes_lhb_pct(self):
        raw_rows = []
        for i in range(6):
            stand = "L" if i < 2 else "R"
            raw_rows.append(_make_raw_row(
                batter=100 + i,
                game_pk=1,
                game_date="2024-06-01",
                inning_topbot="Top",
                events="field_out",
                stand=stand,
                home_team="NYY",
                away_team="BOS",
            ))
        raw_df = pd.DataFrame(raw_rows)

        starters_df = pd.DataFrame([{
            "pitcher": 200, "game_pk": 1,
            "game_date": pd.Timestamp("2024-06-01"),
            "opposing_team": "BOS",
        }])

        result = compute_opp_lineup_handedness(raw_df, starters_df)

        assert result["opp_lineup_pct_lhb"].iloc[0] == pytest.approx(2.0 / 6.0)
