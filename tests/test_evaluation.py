import numpy as np
import pandas as pd
import pytest

from src.models.evaluation import (
    evaluate_game_level_composition,
    evaluate_model,
    plot_feature_importance,
    plot_precision_recall_curve,
    save_evaluation_report,
)


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------

class TestEvaluateModel:

    def test_returns_all_expected_keys(self):
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        y_pred_proba = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9, 0.15, 0.25, 0.6, 0.05])

        metrics = evaluate_model(y_true, y_pred_proba)

        expected_keys = {"pr_auc", "roc_auc", "precision", "recall", "f1",
                         "n_positive", "n_total", "positive_rate",
                         "brier_score", "log_loss"}
        assert set(metrics.keys()) == expected_keys

    def test_metric_values_in_valid_ranges(self):
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        y_pred_proba = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9, 0.15, 0.25, 0.6, 0.05])

        metrics = evaluate_model(y_true, y_pred_proba)

        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert 0.0 <= metrics["pr_auc"] <= 1.0
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
        assert metrics["n_positive"] == 4
        assert metrics["n_total"] == 10

    def test_perfect_predictions_yield_perfect_scores(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred_proba = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

        metrics = evaluate_model(y_true, y_pred_proba)

        assert metrics["roc_auc"] == pytest.approx(1.0)
        assert metrics["pr_auc"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# plot_precision_recall_curve
# ---------------------------------------------------------------------------

class TestPlotPrecisionRecallCurve:

    def test_saves_file_to_disk(self, tmp_path):
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.1, 0.4, 0.6, 0.9])
        save_path = tmp_path / "pr_curve.png"

        plot_precision_recall_curve(y_true, y_pred_proba, save_path=save_path)

        assert save_path.exists()
        assert save_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# plot_feature_importance
# ---------------------------------------------------------------------------

class TestPlotFeatureImportance:

    def test_saves_file_to_disk(self, tmp_path):
        names = [f"feat_{i}" for i in range(5)]
        importances = np.array([100.0, 80.0, 60.0, 40.0, 20.0])
        save_path = tmp_path / "importance.png"

        plot_feature_importance(names, importances, top_n=5, save_path=save_path)

        assert save_path.exists()
        assert save_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# evaluate_game_level_composition
# ---------------------------------------------------------------------------

class TestEvaluateGameLevelComposition:

    def test_composes_pa_probabilities_to_game_level(self):
        test_df = pd.DataFrame({
            "batter": [100, 100, 200, 200],
            "game_pk": [1, 1, 1, 1],
            "game_date": pd.to_datetime(["2023-06-01"] * 4),
            "is_hr": [1, 0, 0, 0],
        })
        pa_pred_proba = np.array([0.4, 0.3, 0.1, 0.1])

        result = evaluate_game_level_composition(test_df, pa_pred_proba)

        expected_keys = {"pr_auc", "roc_auc", "precision", "recall", "f1",
                         "n_positive", "n_total", "positive_rate",
                         "brier_score", "log_loss"}
        assert set(result.keys()) == expected_keys

        df_check = test_df.copy()
        df_check["pred_proba"] = pa_pred_proba
        game_df = df_check.groupby(["batter", "game_pk", "game_date"]).agg(
            actual_hr=("is_hr", "max"),
        ).reset_index()
        assert game_df.loc[game_df["batter"] == 100, "actual_hr"].iloc[0] == 1
        assert game_df.loc[game_df["batter"] == 200, "actual_hr"].iloc[0] == 0


# ---------------------------------------------------------------------------
# save_evaluation_report
# ---------------------------------------------------------------------------

class TestSaveEvaluationReport:

    def test_creates_report_and_plots(self, tmp_path):
        metrics = {
            "pr_auc": 0.45, "roc_auc": 0.82,
            "precision": 0.30, "recall": 0.60, "f1": 0.40,
            "n_positive": 120, "n_total": 4000, "positive_rate": 0.03,
        }
        game_metrics = {
            "pr_auc": 0.50, "roc_auc": 0.85,
            "precision": 0.35, "recall": 0.65, "f1": 0.45,
            "n_positive": 80, "n_total": 2000, "positive_rate": 0.04,
        }
        names = ["feat_a", "feat_b", "feat_c"]
        importances = np.array([50.0, 30.0, 20.0])
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.1, 0.4, 0.6, 0.9])

        save_evaluation_report(metrics, names, importances, y_true, y_pred_proba,
                               game_metrics=game_metrics, output_dir=tmp_path)

        assert (tmp_path / "evaluation_report.txt").exists()
        assert (tmp_path / "precision_recall_curve.png").exists()
        assert (tmp_path / "feature_importance.png").exists()

        report_text = (tmp_path / "evaluation_report.txt").read_text()
        assert "Game-Level" in report_text

    def test_report_contains_metric_values(self, tmp_path):
        metrics = {
            "pr_auc": 0.45, "roc_auc": 0.82,
            "precision": 0.30, "recall": 0.60, "f1": 0.40,
            "n_positive": 120, "n_total": 4000, "positive_rate": 0.03,
        }
        game_metrics = {
            "pr_auc": 0.50, "roc_auc": 0.85,
            "precision": 0.35, "recall": 0.65, "f1": 0.45,
            "n_positive": 80, "n_total": 2000, "positive_rate": 0.04,
        }
        names = ["feat_a", "feat_b"]
        importances = np.array([50.0, 30.0])
        y_true = np.array([0, 1])
        y_pred_proba = np.array([0.2, 0.8])

        save_evaluation_report(metrics, names, importances, y_true, y_pred_proba,
                               game_metrics=game_metrics, output_dir=tmp_path)

        report_text = (tmp_path / "evaluation_report.txt").read_text()
        assert "0.4500" in report_text  # PR-AUC
        assert "0.8200" in report_text  # ROC-AUC
        assert "Game-Level" in report_text
