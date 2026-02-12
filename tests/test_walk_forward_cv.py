import numpy as np
import pandas as pd
import pytest

from src.models.walk_forward_cv import (
    CVFold,
    aggregate_cv_metrics,
    generate_cv_folds,
)


# ---------------------------------------------------------------------------
# generate_cv_folds
# ---------------------------------------------------------------------------

class TestGenerateCvFolds:

    @pytest.fixture
    def daily_game_dates(self):
        return pd.Series(pd.date_range("2022-04-01", "2024-06-30", freq="D"))

    def test_generates_expected_number_of_folds(self, daily_game_dates):
        folds = generate_cv_folds(
            daily_game_dates,
            holdout_date="2024-07-01",
            val_window_days=45,
            step_days=30,
            min_train_days=365,
        )

        assert len(folds) >= 10
        assert len(folds) <= 16

    def test_folds_are_temporally_ordered(self, daily_game_dates):
        folds = generate_cv_folds(
            daily_game_dates,
            holdout_date="2024-07-01",
            val_window_days=45,
            step_days=30,
            min_train_days=365,
        )

        for i in range(1, len(folds)):
            assert folds[i].val_start > folds[i - 1].val_start
            assert folds[i].val_end > folds[i - 1].val_end

    def test_train_end_equals_val_start(self, daily_game_dates):
        folds = generate_cv_folds(
            daily_game_dates,
            holdout_date="2024-07-01",
            val_window_days=45,
            step_days=30,
            min_train_days=365,
        )

        for fold in folds:
            assert fold.train_end == fold.val_start

    def test_val_windows_do_not_exceed_holdout(self, daily_game_dates):
        folds = generate_cv_folds(
            daily_game_dates,
            holdout_date="2024-07-01",
            val_window_days=45,
            step_days=30,
            min_train_days=365,
        )

        holdout_ts = pd.Timestamp("2024-07-01")
        for fold in folds:
            assert fold.val_end <= holdout_ts

    def test_first_fold_has_minimum_training_data(self, daily_game_dates):
        min_train_days = 365

        folds = generate_cv_folds(
            daily_game_dates,
            holdout_date="2024-07-01",
            val_window_days=45,
            step_days=30,
            min_train_days=min_train_days,
        )

        training_span = (folds[0].train_end - daily_game_dates.min()).days
        assert training_span >= min_train_days

    def test_returns_empty_for_insufficient_data(self):
        short_dates = pd.Series(pd.date_range("2024-01-01", periods=100, freq="D"))

        folds = generate_cv_folds(
            short_dates,
            holdout_date="2024-07-01",
            val_window_days=45,
            step_days=30,
            min_train_days=365,
        )

        assert folds == []

    def test_fold_indices_are_sequential(self, daily_game_dates):
        folds = generate_cv_folds(
            daily_game_dates,
            holdout_date="2024-07-01",
            val_window_days=45,
            step_days=30,
            min_train_days=365,
        )

        assert [f.fold_idx for f in folds] == list(range(len(folds)))


# ---------------------------------------------------------------------------
# aggregate_cv_metrics
# ---------------------------------------------------------------------------

class TestAggregateCvMetrics:

    def test_computes_mean_std_min_max(self):
        fold_metrics = [
            {"pr_auc": 0.4, "roc_auc": 0.7},
            {"pr_auc": 0.5, "roc_auc": 0.8},
            {"pr_auc": 0.6, "roc_auc": 0.9},
        ]

        summary = aggregate_cv_metrics(fold_metrics)

        assert summary["pr_auc_mean"] == pytest.approx(0.5)
        assert summary["pr_auc_std"] == pytest.approx(0.0816, rel=0.01)
        assert summary["pr_auc_min"] == pytest.approx(0.4)
        assert summary["pr_auc_max"] == pytest.approx(0.6)

    def test_includes_n_folds(self):
        fold_metrics = [
            {"pr_auc": 0.4},
            {"pr_auc": 0.5},
            {"pr_auc": 0.6},
        ]

        summary = aggregate_cv_metrics(fold_metrics)

        assert summary["n_folds"] == 3

    def test_handles_single_fold(self):
        fold_metrics = [{"pr_auc": 0.55}]

        summary = aggregate_cv_metrics(fold_metrics)

        assert summary["pr_auc_mean"] == pytest.approx(0.55)
        assert summary["pr_auc_std"] == pytest.approx(0.0)
        assert summary["pr_auc_min"] == pytest.approx(0.55)
        assert summary["pr_auc_max"] == pytest.approx(0.55)
