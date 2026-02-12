import numpy as np
import pandas as pd
import pytest

from src.features.opportunity_features import (
    _compute_expected_pas,
    _resolve_lineup_position,
    add_opportunity_features,
    compute_team_rolling_runs,
    derive_batting_order,
)


def _make_raw_pitch_rows(game_pk, game_date, home_team, away_team, home_score,
                         away_score, side_batters, inning_topbot):
    rows = []
    for at_bat_num, batter_id in enumerate(side_batters, start=1):
        rows.append({
            "batter": batter_id,
            "game_pk": game_pk,
            "game_date": game_date,
            "at_bat_number": at_bat_num,
            "inning_topbot": inning_topbot,
            "events": "single",
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "inning": 1,
        })
    return rows


class TestDeriveBattingOrder:

    def test_assigns_positions_1_through_9_for_standard_lineup(self):
        batters = list(range(101, 110))
        rows = _make_raw_pitch_rows(
            game_pk=1, game_date="2024-06-01", home_team="NYY", away_team="BOS",
            home_score=3, away_score=2, side_batters=batters, inning_topbot="Bot",
        )
        raw_df = pd.DataFrame(rows)

        result = derive_batting_order(raw_df)

        home_result = result.sort_values("batting_order_pos")
        assert list(home_result["batting_order_pos"]) == list(range(1, 10))

    def test_caps_pinch_hitters_at_position_9(self):
        batters = list(range(101, 112))
        rows = _make_raw_pitch_rows(
            game_pk=1, game_date="2024-06-01", home_team="NYY", away_team="BOS",
            home_score=3, away_score=2, side_batters=batters, inning_topbot="Bot",
        )
        raw_df = pd.DataFrame(rows)

        result = derive_batting_order(raw_df)

        assert result["batting_order_pos"].max() == 9
        assert len(result[result["batting_order_pos"] == 9]) >= 3

    def test_separates_home_and_away_batters(self):
        home_batters = list(range(101, 110))
        away_batters = list(range(201, 210))
        home_rows = _make_raw_pitch_rows(
            game_pk=1, game_date="2024-06-01", home_team="NYY", away_team="BOS",
            home_score=3, away_score=2, side_batters=home_batters,
            inning_topbot="Bot",
        )
        away_rows = _make_raw_pitch_rows(
            game_pk=1, game_date="2024-06-01", home_team="NYY", away_team="BOS",
            home_score=3, away_score=2, side_batters=away_batters,
            inning_topbot="Top",
        )
        raw_df = pd.DataFrame(home_rows + away_rows)

        result = derive_batting_order(raw_df)

        home_result = result[result["batter"].isin(home_batters)]
        away_result = result[result["batter"].isin(away_batters)]
        assert sorted(home_result["batting_order_pos"].tolist()) == list(range(1, 10))
        assert sorted(away_result["batting_order_pos"].tolist()) == list(range(1, 10))

    def test_returns_game_date_column(self):
        batters = list(range(101, 110))
        rows = _make_raw_pitch_rows(
            game_pk=1, game_date="2024-06-01", home_team="NYY", away_team="BOS",
            home_score=3, away_score=2, side_batters=batters, inning_topbot="Bot",
        )
        raw_df = pd.DataFrame(rows)

        result = derive_batting_order(raw_df)

        assert "game_date" in result.columns


class TestComputeTeamRollingRuns:

    def _make_team_raw_df(self, team, opponent, n_games, base_date="2024-06-01",
                          is_home=True, runs=5, opp_runs=3):
        rows = []
        for i in range(n_games):
            date = pd.Timestamp(base_date) + pd.Timedelta(days=i)
            ht = team if is_home else opponent
            at = opponent if is_home else team
            hs = runs if is_home else opp_runs
            aws = opp_runs if is_home else runs
            rows.append({
                "game_pk": 1000 + i,
                "game_date": str(date.date()),
                "home_team": ht,
                "away_team": at,
                "home_score": hs,
                "away_score": aws,
                "batter": 999,
                "events": "single",
                "inning_topbot": "Bot",
                "inning": 1,
                "at_bat_number": 1,
            })
        return pd.DataFrame(rows)

    def test_first_games_have_nan_runs(self):
        raw_df = self._make_team_raw_df("NYY", "BOS", n_games=3)

        result = compute_team_rolling_runs(raw_df, window=30)

        nyy_runs = result[result["team"] == "NYY"]["team_runs_per_game"]
        assert nyy_runs.isna().all()

    def test_rolling_uses_shift_to_prevent_leakage(self):
        raw_df = self._make_team_raw_df("NYY", "BOS", n_games=7, runs=4, opp_runs=2)
        last_game_pk = 1006
        raw_df.loc[raw_df["game_pk"] == last_game_pk, "home_score"] = 20

        result = compute_team_rolling_runs(raw_df, window=30)

        last_game_val = result[
            (result["team"] == "NYY") & (result["game_pk"] == last_game_pk)
        ]["team_runs_per_game"].iloc[0]
        assert last_game_val == pytest.approx(4.0, abs=0.01)

    def test_both_home_and_away_games_counted(self):
        home_df = self._make_team_raw_df(
            "NYY", "BOS", n_games=4, is_home=True, runs=6, opp_runs=2,
        )
        away_df = self._make_team_raw_df(
            "NYY", "LAD", n_games=4, is_home=False, runs=6, opp_runs=2,
            base_date="2024-06-05",
        )
        away_df["game_pk"] = away_df["game_pk"] + 100
        raw_df = pd.concat([home_df, away_df], ignore_index=True)

        result = compute_team_rolling_runs(raw_df, window=30)

        nyy_rows = result[result["team"] == "NYY"]
        assert len(nyy_rows) == 8
        assert nyy_rows["team_runs_per_game"].notna().any()


class TestAddOpportunityFeatures:

    def _make_pa_df(self, rows):
        return pd.DataFrame(rows)

    def _make_batting_order_df(self, entries):
        return pd.DataFrame(entries, columns=[
            "batter", "game_pk", "game_date", "batting_order_pos",
        ])

    def _make_team_runs_df(self, entries):
        return pd.DataFrame(entries, columns=[
            "team", "game_pk", "game_date", "team_runs_per_game",
        ])

    def test_all_four_columns_present(self):
        pa_df = self._make_pa_df([{
            "batter": 101, "game_pk": 1, "game_date": "2024-06-01",
            "home_team": "NYY", "away_team": "BOS", "is_home": 0,
        }])
        bo_df = self._make_batting_order_df([
            (101, 1, "2024-06-01", 3),
        ])
        tr_df = self._make_team_runs_df([
            ("BOS", 1, "2024-06-01", 4.5),
        ])

        result = add_opportunity_features(pa_df, bo_df, tr_df)

        for col in ["batting_order_pos", "batting_order_avg", "team_runs_per_game",
                     "expected_pas"]:
            assert col in result.columns

    def test_row_count_preserved(self):
        pa_df = self._make_pa_df([
            {"batter": 101, "game_pk": 1, "game_date": "2024-06-01",
             "home_team": "NYY", "away_team": "BOS", "is_home": 0},
            {"batter": 102, "game_pk": 1, "game_date": "2024-06-01",
             "home_team": "NYY", "away_team": "BOS", "is_home": 0},
        ])
        bo_df = self._make_batting_order_df([
            (101, 1, "2024-06-01", 1),
            (102, 1, "2024-06-01", 2),
        ])
        tr_df = self._make_team_runs_df([
            ("BOS", 1, "2024-06-01", 4.5),
        ])

        result = add_opportunity_features(pa_df, bo_df, tr_df)

        assert len(result) == 2

    def test_leadoff_gets_more_expected_pas_than_nine_hole(self):
        pa_df = self._make_pa_df([
            {"batter": 101, "game_pk": 1, "game_date": "2024-06-01",
             "home_team": "NYY", "away_team": "BOS", "is_home": 0},
            {"batter": 102, "game_pk": 1, "game_date": "2024-06-01",
             "home_team": "NYY", "away_team": "BOS", "is_home": 0},
        ])
        bo_df = self._make_batting_order_df([
            (101, 1, "2024-06-01", 1),
            (102, 1, "2024-06-01", 9),
        ])
        tr_df = self._make_team_runs_df([
            ("BOS", 1, "2024-06-01", 4.5),
        ])

        result = add_opportunity_features(pa_df, bo_df, tr_df)

        leadoff = result[result["batter"] == 101]["expected_pas"].iloc[0]
        nine_hole = result[result["batter"] == 102]["expected_pas"].iloc[0]
        assert leadoff > nine_hole

    def test_home_batter_gets_fewer_expected_pas(self):
        pa_df = self._make_pa_df([
            {"batter": 101, "game_pk": 1, "game_date": "2024-06-01",
             "home_team": "NYY", "away_team": "BOS", "is_home": 1},
            {"batter": 102, "game_pk": 2, "game_date": "2024-06-01",
             "home_team": "LAD", "away_team": "CHC", "is_home": 0},
        ])
        bo_df = self._make_batting_order_df([
            (101, 1, "2024-06-01", 5),
            (102, 2, "2024-06-01", 5),
        ])
        tr_df = self._make_team_runs_df([
            ("NYY", 1, "2024-06-01", 4.5),
            ("CHC", 2, "2024-06-01", 4.5),
        ])

        result = add_opportunity_features(pa_df, bo_df, tr_df)

        home_pas = result[result["batter"] == 101]["expected_pas"].iloc[0]
        away_pas = result[result["batter"] == 102]["expected_pas"].iloc[0]
        assert home_pas < away_pas

    def test_missing_team_runs_defaults_in_expected_pas(self):
        pa_df = self._make_pa_df([{
            "batter": 101, "game_pk": 1, "game_date": "2024-06-01",
            "home_team": "NYY", "away_team": "BOS", "is_home": 0,
        }])
        bo_df = self._make_batting_order_df([
            (101, 1, "2024-06-01", 5),
        ])
        tr_df = self._make_team_runs_df([
            ("XXX", 999, "2024-06-01", 4.5),
        ])

        result = add_opportunity_features(pa_df, bo_df, tr_df)

        assert result["expected_pas"].notna().all()


class TestExpectedPasHeuristic:

    def _build_df(self, batting_order_pos, batting_order_avg, team_runs_per_game,
                  is_home):
        return pd.DataFrame([{
            "batting_order_pos": batting_order_pos,
            "batting_order_avg": batting_order_avg,
            "team_runs_per_game": team_runs_per_game,
            "is_home": is_home,
        }])

    def test_leadoff_base_pas_value(self):
        df = self._build_df(
            batting_order_pos=1, batting_order_avg=1.0,
            team_runs_per_game=4.5, is_home=0,
        )

        result = _compute_expected_pas(df)

        assert result["expected_pas"].iloc[0] == pytest.approx(4.72)

    def test_nine_hole_base_pas_value(self):
        df = self._build_df(
            batting_order_pos=9, batting_order_avg=9.0,
            team_runs_per_game=4.5, is_home=0,
        )

        result = _compute_expected_pas(df)

        assert result["expected_pas"].iloc[0] == pytest.approx(4.20)

    def test_high_scoring_team_increases_expected_pas(self):
        df_high = self._build_df(
            batting_order_pos=5, batting_order_avg=5.0,
            team_runs_per_game=6.75, is_home=0,
        )
        df_default = self._build_df(
            batting_order_pos=5, batting_order_avg=5.0,
            team_runs_per_game=4.5, is_home=0,
        )

        result_high = _compute_expected_pas(df_high)
        result_default = _compute_expected_pas(df_default)

        assert result_high["expected_pas"].iloc[0] > result_default["expected_pas"].iloc[0]
