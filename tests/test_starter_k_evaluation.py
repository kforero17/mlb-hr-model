import numpy as np
import pandas as pd
import pytest

from src.models.starter_k_evaluation import (
    evaluate_k_count,
    evaluate_over_under,
    fit_dispersion,
    predict_over_under,
    save_starter_k_report,
)


# ---------------------------------------------------------------------------
# evaluate_k_count
# ---------------------------------------------------------------------------

class TestEvaluateKCount:

    def test_perfect_prediction_gives_zero_mae(self):
        y_true = np.array([5, 6, 7, 4, 8])

        result = evaluate_k_count(y_true, y_true.astype(float))

        assert result["mae"] == pytest.approx(0.0)
        assert result["rmse"] == pytest.approx(0.0)

    def test_returns_expected_keys(self):
        y_true = np.array([5, 6, 7, 4, 8])
        lambda_pred = np.array([5.5, 5.5, 6.0, 4.5, 7.0])

        result = evaluate_k_count(y_true, lambda_pred)

        assert set(result.keys()) == {"mae", "rmse", "poisson_deviance", "mean_predicted", "mean_actual"}

    def test_mean_predicted_and_actual(self):
        y_true = np.array([4, 6, 8])
        lambda_pred = np.array([3.0, 5.0, 7.0])

        result = evaluate_k_count(y_true, lambda_pred)

        assert result["mean_actual"] == pytest.approx(6.0)
        assert result["mean_predicted"] == pytest.approx(5.0)

    def test_known_mae_and_rmse(self):
        y_true = np.array([5, 5, 5, 5])
        lambda_pred = np.array([4, 6, 4, 6])

        result = evaluate_k_count(y_true, lambda_pred)

        assert result["mae"] == pytest.approx(1.0)
        assert result["rmse"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# fit_dispersion
# ---------------------------------------------------------------------------

class TestFitDispersion:

    def test_returns_clipped_to_minimum(self):
        y_true = np.array([5.0, 5.0, 5.0])
        lambda_pred = np.array([5.0, 5.0, 5.0])

        result = fit_dispersion(y_true, lambda_pred)

        assert result == pytest.approx(1.01)

    def test_returns_float_in_range(self):
        y_true = np.array([3, 5, 7, 4, 8, 6, 9, 2, 5, 7])
        lambda_pred = np.array([5.0] * 10)

        d = fit_dispersion(y_true, lambda_pred)

        assert 1.01 <= d <= 3.0

    def test_clipped_to_maximum(self):
        y_true = np.array([0, 15, 0, 15, 0, 15])
        lambda_pred = np.array([1.0] * 6)

        d = fit_dispersion(y_true, lambda_pred)

        assert d == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# predict_over_under
# ---------------------------------------------------------------------------

class TestPredictOverUnder:

    def test_high_lambda_gives_high_p_over(self):
        result = predict_over_under(np.array([10.0]), 5.5, 1.14)

        assert result[0] > 0.8

    def test_low_lambda_gives_low_p_over(self):
        result = predict_over_under(np.array([2.0]), 7.5, 1.14)

        assert result[0] < 0.05

    def test_zero_lambda_gives_zero(self):
        result = predict_over_under(np.array([0.0]), 5.5, 1.14)

        assert result[0] == 0.0

    def test_monotonic_in_lambda(self):
        lambdas = np.array([3.0, 5.0, 7.0, 9.0])

        result = predict_over_under(lambdas, 6.5, 1.14)

        assert all(result[i] < result[i + 1] for i in range(len(result) - 1))

    def test_returns_array_matching_input_length(self):
        lambdas = np.array([4.0, 5.0, 6.0])

        result = predict_over_under(lambdas, 5.5, 1.2)

        assert len(result) == 3

    def test_probabilities_between_zero_and_one(self):
        lambdas = np.array([1.0, 5.0, 10.0, 15.0])

        result = predict_over_under(lambdas, 6.5, 1.5)

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


# ---------------------------------------------------------------------------
# evaluate_over_under
# ---------------------------------------------------------------------------

class TestEvaluateOverUnder:

    def test_returns_dataframe_with_expected_columns(self):
        y_true = np.array([5, 6, 7, 4, 8, 9, 3, 7, 6, 5])
        lambda_pred = np.array([5.5, 5.5, 6.0, 4.5, 7.0, 8.0, 3.5, 6.5, 5.0, 5.5])

        result = evaluate_over_under(y_true, lambda_pred, [5.5, 6.5], 1.14)

        assert set(result.columns) == {
            "line", "brier_score", "accuracy", "mean_p_over",
            "actual_over_rate", "n_over", "n_total",
        }
        assert len(result) == 2

    def test_n_total_equals_input_size(self):
        y_true = np.array([5, 6, 7])
        lambda_pred = np.array([5.0, 6.0, 7.0])

        result = evaluate_over_under(y_true, lambda_pred, [5.5], 1.14)

        assert result["n_total"].iloc[0] == 3

    def test_actual_over_rate_matches_data(self):
        y_true = np.array([3, 4, 7, 8])
        lambda_pred = np.array([5.0, 5.0, 5.0, 5.0])

        result = evaluate_over_under(y_true, lambda_pred, [5.5], 1.14)

        assert result["n_over"].iloc[0] == 2
        assert result["actual_over_rate"].iloc[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# save_starter_k_report
# ---------------------------------------------------------------------------

class TestSaveStarterKReport:

    def test_creates_report_and_plots(self, tmp_path):
        count_metrics = {
            "mae": 1.5, "rmse": 2.0, "poisson_deviance": 0.5,
            "mean_predicted": 5.5, "mean_actual": 5.3,
        }
        ou_metrics_df = pd.DataFrame({
            "line": [6.5],
            "brier_score": [0.2],
            "accuracy": [0.6],
            "mean_p_over": [0.4],
            "actual_over_rate": [0.35],
            "n_over": [35],
            "n_total": [100],
        })
        baselines = {"rolling_50g_mean": 1.8, "league_avg": 2.0}
        feature_names = ["feat_a", "feat_b", "feat_c"]
        importances = np.array([50.0, 30.0, 20.0])
        y_true = np.array([5, 6, 7, 4, 8, 3, 9, 5, 6, 7])
        lambda_pred = np.array([5.5, 5.5, 6.0, 4.5, 7.0, 3.5, 8.0, 5.0, 5.5, 6.5])

        save_starter_k_report(
            count_metrics, ou_metrics_df, baselines, feature_names,
            importances, y_true, lambda_pred, 1.14, tmp_path,
        )

        assert (tmp_path / "starter_k_report.txt").exists()
        assert (tmp_path / "k_distribution.png").exists()
        assert (tmp_path / "feature_importance_starter_k.png").exists()

    def test_report_contains_metric_values(self, tmp_path):
        count_metrics = {
            "mae": 1.234, "rmse": 2.0, "poisson_deviance": 0.5,
            "mean_predicted": 5.5, "mean_actual": 5.3,
        }
        ou_metrics_df = pd.DataFrame({
            "line": [6.5],
            "brier_score": [0.2],
            "accuracy": [0.6],
            "mean_p_over": [0.4],
            "actual_over_rate": [0.35],
            "n_over": [35],
            "n_total": [100],
        })
        feature_names = ["feat_a", "feat_b"]
        importances = np.array([50.0, 30.0])
        y_true = np.array([5, 6, 7, 4])
        lambda_pred = np.array([5.0, 6.0, 7.0, 4.0])

        save_starter_k_report(
            count_metrics, ou_metrics_df, {}, feature_names,
            importances, y_true, lambda_pred, 1.14, tmp_path,
        )

        report_text = (tmp_path / "starter_k_report.txt").read_text()
        assert "1.234" in report_text
        assert "STARTER K MODEL" in report_text
