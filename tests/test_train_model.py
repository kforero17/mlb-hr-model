import numpy as np
import pandas as pd
import pytest
import lightgbm as lgb

from src.models.train_model import (
    METADATA_COLUMNS,
    TARGET_COLUMN,
    compute_scale_pos_weight,
    prepare_features,
    time_based_split,
    train_lightgbm,
)


# ---------------------------------------------------------------------------
# time_based_split
# ---------------------------------------------------------------------------

class TestTimeBasedSplit:

    @pytest.fixture
    def date_range_df(self):
        dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
        return pd.DataFrame({
            "game_date": dates,
            "batter": range(len(dates)),
            "hit_hr": np.random.default_rng(42).integers(0, 2, len(dates)),
        })

    def test_train_dates_before_split_and_test_dates_after(self, date_range_df):
        train, val, test = time_based_split(date_range_df)

        split = pd.Timestamp("2024-07-01")
        assert train["game_date"].max() < split
        assert val["game_date"].max() < split
        assert test["game_date"].min() >= split

    def test_no_row_overlap_and_total_preserved(self, date_range_df):
        train, val, test = time_based_split(date_range_df)

        assert len(train) + len(val) + len(test) == len(date_range_df)

        all_indices = set(train.index) | set(val.index) | set(test.index)
        assert len(all_indices) == len(date_range_df)

    def test_val_is_tail_of_train_portion(self, date_range_df):
        train, val, test = time_based_split(date_range_df)

        assert train["game_date"].max() <= val["game_date"].min()


# ---------------------------------------------------------------------------
# prepare_features
# ---------------------------------------------------------------------------

class TestPrepareFeatures:

    def test_excludes_metadata_and_target_columns(self):
        df = pd.DataFrame({
            "batter": [1, 2],
            "game_pk": [10, 20],
            "game_date": pd.to_datetime(["2023-06-01", "2023-06-02"]),
            "pitcher": [100, 200],
            "home_team": ["NYY", "BOS"],
            "away_team": ["BOS", "NYY"],
            "hit_hr": [0, 1],
            "feature_a": [1.5, 2.5],
            "feature_b": [3.0, 4.0],
            "stand": ["R", "L"],
        })

        X, y = prepare_features(df)

        for col in METADATA_COLUMNS + [TARGET_COLUMN]:
            assert col not in X.columns

        assert list(y) == [0, 1]

    def test_categorical_columns_converted_to_category_dtype(self):
        df = pd.DataFrame({
            "batter": [1], "game_pk": [10],
            "game_date": pd.to_datetime(["2023-06-01"]),
            "pitcher": [100], "home_team": ["NYY"], "away_team": ["BOS"],
            "hit_hr": [0],
            "stand": ["R"], "p_throws": ["L"], "platoon": ["R_vs_L"],
            "feature_a": [1.0],
        })

        X, _ = prepare_features(df)

        assert X["stand"].dtype.name == "category"
        assert X["p_throws"].dtype.name == "category"
        assert X["platoon"].dtype.name == "category"


# ---------------------------------------------------------------------------
# compute_scale_pos_weight
# ---------------------------------------------------------------------------

class TestComputeScalePosWeight:

    def test_known_class_distribution(self):
        y = pd.Series([0] * 97 + [1] * 3)

        weight = compute_scale_pos_weight(y)

        assert weight == pytest.approx(97 / 3, rel=1e-6)


# ---------------------------------------------------------------------------
# train_lightgbm
# ---------------------------------------------------------------------------

class TestTrainLightGBM:

    def test_returns_booster_with_valid_predictions(self):
        rng = np.random.default_rng(42)
        n_train, n_val, n_features = 200, 50, 5

        X_train = pd.DataFrame(rng.standard_normal((n_train, n_features)),
                               columns=[f"f{i}" for i in range(n_features)])
        y_train = pd.Series(rng.integers(0, 2, n_train))
        X_val = pd.DataFrame(rng.standard_normal((n_val, n_features)),
                             columns=[f"f{i}" for i in range(n_features)])
        y_val = pd.Series(rng.integers(0, 2, n_val))

        model = train_lightgbm(X_train, y_train, X_val, y_val, categorical_features=[])

        assert isinstance(model, lgb.Booster)

        preds = model.predict(X_val)
        assert preds.shape == (n_val,)
        assert np.all((preds >= 0) & (preds <= 1))
