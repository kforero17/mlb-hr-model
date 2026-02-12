import numpy as np
import optuna
import pandas as pd
import pytest

from src.models.tuning import build_search_space, run_tuning

optuna.logging.set_verbosity(optuna.logging.WARNING)

EXPECTED_KEYS = {
    "num_leaves",
    "max_depth",
    "learning_rate",
    "feature_fraction",
    "feature_fraction_bynode",
    "bagging_fraction",
    "min_child_samples",
    "min_sum_hessian_in_leaf",
    "lambda_l1",
    "lambda_l2",
}


# ---------------------------------------------------------------------------
# build_search_space
# ---------------------------------------------------------------------------

class TestBuildSearchSpace:

    def test_returns_all_expected_keys(self):
        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        params = build_search_space(trial)

        assert set(params.keys()) == EXPECTED_KEYS

    def test_values_within_search_bounds(self):
        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        params = build_search_space(trial)

        assert 15 <= params["num_leaves"] <= 127
        assert 4 <= params["max_depth"] <= 12
        assert 0.01 <= params["learning_rate"] <= 0.1
        assert 0.5 <= params["feature_fraction"] <= 1.0
        assert 0.5 <= params["feature_fraction_bynode"] <= 1.0
        assert 0.5 <= params["bagging_fraction"] <= 1.0
        assert 50 <= params["min_child_samples"] <= 500
        assert 1e-4 <= params["min_sum_hessian_in_leaf"] <= 10.0
        assert 1e-3 <= params["lambda_l1"] <= 5.0
        assert 1e-3 <= params["lambda_l2"] <= 5.0

    def test_integer_params_are_int(self):
        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        params = build_search_space(trial)

        assert isinstance(params["num_leaves"], int)
        assert isinstance(params["max_depth"], int)
        assert isinstance(params["min_child_samples"], int)


# ---------------------------------------------------------------------------
# run_tuning
# ---------------------------------------------------------------------------

class TestRunTuning:

    @pytest.mark.slow
    def test_returns_dict_with_param_keys(self):
        rng = np.random.default_rng(42)
        all_dates = pd.date_range("2022-04-01", "2024-06-30", freq="D")
        game_dates = np.tile(all_dates, 4)[:len(all_dates) * 3]
        n = len(game_dates)
        df = pd.DataFrame({
            "game_date": game_dates,
            "batter": rng.integers(100, 110, n),
            "game_pk": rng.integers(1, 500, n),
            "pitcher": rng.integers(900, 910, n),
            "home_team": rng.choice(["NYY", "BOS", "LAD"], n),
            "away_team": rng.choice(["NYY", "BOS", "LAD"], n),
            "is_hr": rng.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], n),
            "feature_a": rng.standard_normal(n),
            "feature_b": rng.standard_normal(n),
            "feature_c": rng.standard_normal(n),
            "stand": rng.choice(["R", "L"], n),
            "p_throws": rng.choice(["R", "L"], n),
            "platoon": rng.choice(["R_vs_R", "R_vs_L", "L_vs_R", "L_vs_L"], n),
            "inning": rng.integers(1, 10, n),
        })

        result = run_tuning(df, n_trials=2, n_cv_folds=1)

        assert isinstance(result, dict)
        assert "num_leaves" in result
