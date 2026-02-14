import numpy as np
import pandas as pd
import pytest

from src.models.train_starter_k_model import (
    ENGINEERED_PREFIXES,
    STATIC_FEATURES,
    prepare_features,
    select_feature_columns,
    time_based_split,
)


# ---------------------------------------------------------------------------
# select_feature_columns
# ---------------------------------------------------------------------------

class TestSelectFeatureColumns:

    def test_selects_static_and_prefixed_columns(self):
        df = pd.DataFrame({
            "pitcher": [1],
            "game_pk": [1],
            "game_date": ["2023-01-01"],
            "n_k_against": [5],
            "pitcher_k_rate_50g": [0.25],
            "opp_team_k_rate_50g": [0.22],
            "park_k_factor": [1.05],
            "is_home": [1],
            "month": [6],
            "ix_pitcher_k_x_opp_team_k": [0.05],
        })

        cols = select_feature_columns(df)

        assert "pitcher_k_rate_50g" in cols
        assert "opp_team_k_rate_50g" in cols
        assert "park_k_factor" in cols
        assert "ix_pitcher_k_x_opp_team_k" in cols
        assert "is_home" in cols
        assert "month" in cols

    def test_excludes_metadata_and_target(self):
        df = pd.DataFrame({
            "pitcher": [1],
            "game_pk": [1],
            "game_date": ["2023-01-01"],
            "n_k_against": [5],
            "home_team": ["NYY"],
            "away_team": ["BOS"],
            "opposing_team": ["BOS"],
            "park_k_factor": [1.05],
        })

        cols = select_feature_columns(df)

        assert "pitcher" not in cols
        assert "game_pk" not in cols
        assert "n_k_against" not in cols
        assert "home_team" not in cols
        assert "park_k_factor" in cols

    def test_matches_all_engineered_prefixes(self):
        columns = {"pitcher": [1], "game_pk": [1]}
        for prefix in ENGINEERED_PREFIXES:
            col_name = f"{prefix}test"
            columns[col_name] = [0.5]
        df = pd.DataFrame(columns)

        cols = select_feature_columns(df)

        for prefix in ENGINEERED_PREFIXES:
            assert f"{prefix}test" in cols

    def test_returns_empty_list_when_no_features(self):
        df = pd.DataFrame({
            "pitcher": [1],
            "game_pk": [1],
            "n_k_against": [5],
            "random_column": [0.5],
        })

        cols = select_feature_columns(df)

        assert cols == []


# ---------------------------------------------------------------------------
# prepare_features
# ---------------------------------------------------------------------------

class TestPrepareFeatures:

    def test_returns_correct_x_and_y(self):
        df = pd.DataFrame({
            "pitcher_k_rate_50g": [0.25, 0.30],
            "park_k_factor": [1.0, 1.1],
            "p_throws": ["R", "L"],
            "n_k_against": [5, 7],
        })
        feature_cols = ["pitcher_k_rate_50g", "park_k_factor", "p_throws"]

        X, y = prepare_features(df, feature_cols)

        assert list(X.columns) == feature_cols
        assert list(y) == [5, 7]

    def test_converts_categorical_columns(self):
        df = pd.DataFrame({
            "pitcher_k_rate_50g": [0.25, 0.30],
            "p_throws": ["R", "L"],
            "roof_type": ["dome", "open"],
            "n_k_against": [5, 7],
        })
        feature_cols = ["pitcher_k_rate_50g", "p_throws", "roof_type"]

        X, y = prepare_features(df, feature_cols)

        assert X["p_throws"].dtype.name == "category"
        assert X["roof_type"].dtype.name == "category"

    def test_non_categorical_columns_unchanged(self):
        df = pd.DataFrame({
            "pitcher_k_rate_50g": [0.25, 0.30],
            "n_k_against": [5, 7],
        })
        feature_cols = ["pitcher_k_rate_50g"]

        X, y = prepare_features(df, feature_cols)

        assert X["pitcher_k_rate_50g"].dtype == np.float64


# ---------------------------------------------------------------------------
# time_based_split
# ---------------------------------------------------------------------------

class TestTimeBasedSplit:

    @pytest.fixture
    def date_range_df(self):
        dates = pd.date_range("2023-01-01", "2025-01-01", freq="D")
        return pd.DataFrame({
            "game_date": dates,
            "n_k_against": range(len(dates)),
        })

    def test_splits_at_holdout_date(self, date_range_df):
        train, val, test = time_based_split(date_range_df)

        holdout = pd.Timestamp("2024-07-01")
        assert train["game_date"].max() < holdout
        assert val["game_date"].max() < holdout
        assert test["game_date"].min() >= holdout

    def test_no_row_overlap_and_total_preserved(self, date_range_df):
        train, val, test = time_based_split(date_range_df)

        assert len(train) + len(val) + len(test) == len(date_range_df)

    def test_val_is_tail_of_train_portion(self, date_range_df):
        train, val, test = time_based_split(date_range_df)

        assert train["game_date"].max() <= val["game_date"].min()
