import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config.model_config import EVALUATION_DIR, TOP_N_FEATURES

logger = logging.getLogger(__name__)


def find_f1_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    denom = precisions[:-1] + recalls[:-1]
    denom = np.where(denom == 0, 1.0, denom)
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / denom
    return float(thresholds[np.argmax(f1_scores)])


def evaluate_model(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float | None = None,
) -> dict:
    if threshold is None:
        threshold = find_f1_optimal_threshold(y_true, y_pred_proba)
    y_pred = (y_pred_proba >= threshold).astype(int)

    pr_auc = average_precision_score(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_pred_proba)
    logloss = log_loss(y_true, y_pred_proba)
    n_positive = int(y_true.sum())
    n_total = len(y_true)
    positive_rate = n_positive / n_total if n_total > 0 else 0.0

    metrics = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier_score": brier,
        "log_loss": logloss,
        "n_positive": n_positive,
        "n_total": n_total,
        "positive_rate": positive_rate,
        "baseline_brier": positive_rate * (1 - positive_rate),
        "threshold": threshold,
    }

    logger.info(
        "Model evaluation — PR-AUC: %.4f | ROC-AUC: %.4f | "
        "Precision: %.4f | Recall: %.4f | F1: %.4f | "
        "Brier: %.6f (baseline: %.6f) | Log Loss: %.4f | "
        "Threshold: %.4f | Positive rate: %d/%d (%.2f%%)",
        pr_auc, roc_auc, precision, recall, f1,
        brier, positive_rate * (1 - positive_rate), logloss,
        threshold, n_positive, n_total, positive_rate * 100,
    )

    return metrics


def compute_multi_threshold_report(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = [0.02, 0.03, 0.04, 0.05, 0.06]

    rows: list[dict] = []
    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        n_predicted = int(y_pred.sum())
        rows.append({
            "threshold": t,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "n_predicted": n_predicted,
        })

    report = pd.DataFrame(rows)
    logger.info("Multi-threshold report:\n%s", report.to_string(index=False))
    return report


def compute_probability_distribution(y_pred_proba: np.ndarray) -> dict:
    return {
        "min": float(np.min(y_pred_proba)),
        "p5": float(np.percentile(y_pred_proba, 5)),
        "p25": float(np.percentile(y_pred_proba, 25)),
        "median": float(np.median(y_pred_proba)),
        "p75": float(np.percentile(y_pred_proba, 75)),
        "p95": float(np.percentile(y_pred_proba, 95)),
        "p99": float(np.percentile(y_pred_proba, 99)),
        "max": float(np.max(y_pred_proba)),
    }


def correct_scale_pos_weight(
    y_pred_proba: np.ndarray,
    scale_pos_weight: float,
) -> np.ndarray:
    eps = 1e-15
    clipped = np.clip(y_pred_proba, eps, 1.0 - eps)
    logits = np.log(clipped / (1.0 - clipped))
    corrected_logits = logits - np.log(scale_pos_weight)
    return 1.0 / (1.0 + np.exp(-corrected_logits))


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Path | None = None,
) -> None:
    precisions, recalls, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    baseline = y_true.sum() / len(y_true) if len(y_true) > 0 else 0.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(recalls, precisions, label=f"PR curve (AUC = {pr_auc:.4f})")
    ax.axhline(y=baseline, color="r", linestyle="--", label=f"Baseline ({baseline:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="best")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Precision-recall curve saved to %s", save_path)
    else:
        plt.show()

    plt.close(fig)


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    save_path: Path | None = None,
    label: str = "Model",
    calibrated_proba: np.ndarray | None = None,
) -> None:
    fraction_pos, mean_pred = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy="uniform",
    )
    brier = brier_score_loss(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        mean_pred, fraction_pos, "s-",
        label=f"{label} (Brier={brier:.4f})",
    )

    if calibrated_proba is not None:
        frac_cal, mean_cal = calibration_curve(
            y_true, calibrated_proba, n_bins=n_bins, strategy="uniform",
        )
        brier_cal = brier_score_loss(y_true, calibrated_proba)
        ax.plot(
            mean_cal, frac_cal, "o-",
            label=f"Calibrated (Brier={brier_cal:.4f})",
        )

    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.legend(loc="best")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Calibration curve saved to %s", save_path)
    else:
        plt.show()

    plt.close(fig)


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    top_n: int = TOP_N_FEATURES,
    save_path: Path | None = None,
) -> None:
    sorted_indices = np.argsort(importances)[::-1][:top_n]
    sorted_names = [feature_names[i] for i in sorted_indices]
    sorted_importances = importances[sorted_indices]

    sorted_names = sorted_names[::-1]
    sorted_importances = sorted_importances[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(sorted_names)), sorted_importances)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {len(sorted_names)} Feature Importances")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Feature importance plot saved to %s", save_path)
    else:
        plt.show()

    plt.close(fig)


def evaluate_game_level_composition(
    test_df: pd.DataFrame,
    pa_pred_proba: np.ndarray,
) -> dict:
    df = test_df[["batter", "game_pk", "game_date", "is_hr"]].copy()
    df["pred_proba"] = pa_pred_proba

    game_df = df.groupby(["batter", "game_pk", "game_date"]).agg(
        game_pred_proba=("pred_proba", lambda x: 1 - np.prod(1 - x)),
        actual_hr=("is_hr", "max"),
    ).reset_index()

    metrics = evaluate_model(
        y_true=game_df["actual_hr"].values,
        y_pred_proba=game_df["game_pred_proba"].values,
    )

    logger.info(
        "Game-level composition — PR-AUC: %.4f | ROC-AUC: %.4f",
        metrics["pr_auc"],
        metrics["roc_auc"],
    )

    return metrics


def fit_calibrator(
    y_val: np.ndarray,
    y_val_pred_proba: np.ndarray,
    method: str = "platt",
) -> IsotonicRegression | LogisticRegression:
    if method == "isotonic":
        calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        calibrator.fit(y_val_pred_proba, y_val)
    elif method == "platt":
        calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
        calibrator.fit(y_val_pred_proba.reshape(-1, 1), y_val)
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    logger.info("Fitted %s calibrator on %d validation samples", method, len(y_val))
    return calibrator


def apply_calibrator(
    calibrator: IsotonicRegression | LogisticRegression,
    y_pred_proba: np.ndarray,
) -> np.ndarray:
    if isinstance(calibrator, IsotonicRegression):
        calibrated = calibrator.predict(y_pred_proba)
    elif isinstance(calibrator, LogisticRegression):
        calibrated = calibrator.predict_proba(y_pred_proba.reshape(-1, 1))[:, 1]
    else:
        raise TypeError(f"Unsupported calibrator type: {type(calibrator)}")

    return np.clip(calibrated, 0.0, 1.0)


def _daily_precision_recall_for_k(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    game_dates: np.ndarray,
    k: int,
) -> tuple[float, float]:
    unique_dates = np.unique(game_dates)
    precisions: list[float] = []
    recalls: list[float] = []

    for date in unique_dates:
        mask = game_dates == date
        day_true = y_true[mask]
        day_pred = y_pred_proba[mask]

        top_indices = np.argsort(day_pred)[::-1][:k]
        hits = day_true[top_indices].sum()
        total_hrs = day_true.sum()

        precisions.append(hits / k)
        recalls.append(hits / total_hrs if total_hrs > 0 else 0.0)

    return float(np.mean(precisions)), float(np.mean(recalls))


def compute_daily_precision_recall_at_k(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    game_dates: np.ndarray,
    k_values: list[int] | None = None,
) -> dict:
    if k_values is None:
        k_values = [5, 10, 20]

    result: dict = {}
    for k in k_values:
        prec, rec = _daily_precision_recall_for_k(y_true, y_pred_proba, game_dates, k)
        result[f"precision_at_{k}"] = round(prec, 4)
        result[f"recall_at_{k}"] = round(rec, 4)

    result["n_days"] = len(np.unique(game_dates))

    logger.info(
        "Daily Precision/Recall@K — %s",
        " | ".join(f"{key}: {val}" for key, val in result.items()),
    )
    return result


def compute_bucket_hit_rates(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    bucket_edges: list[float] | None = None,
) -> pd.DataFrame:
    if bucket_edges is None:
        bucket_edges = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 1.0]

    overall_hr_rate = y_true.mean() if len(y_true) > 0 else 0.0
    labels = [
        f"{bucket_edges[i]:.2f}-{bucket_edges[i + 1]:.2f}"
        for i in range(len(bucket_edges) - 1)
    ]

    bucket_indices = np.digitize(y_pred_proba, bucket_edges) - 1
    bucket_indices = np.clip(bucket_indices, 0, len(labels) - 1)

    rows: list[dict] = []
    for i, label in enumerate(labels):
        mask = bucket_indices == i
        n_pred = int(mask.sum())
        n_hrs = int(y_true[mask].sum()) if n_pred > 0 else 0
        hit_rate = n_hrs / n_pred if n_pred > 0 else 0.0
        avg_predicted = float(y_pred_proba[mask].mean()) if n_pred > 0 else 0.0
        lift = hit_rate / overall_hr_rate if overall_hr_rate > 0 else 0.0

        rows.append({
            "bucket": label,
            "n_predictions": n_pred,
            "n_hrs": n_hrs,
            "hit_rate": round(hit_rate, 4),
            "avg_predicted": round(avg_predicted, 4),
            "lift": round(lift, 2),
        })

    bucket_df = pd.DataFrame(rows)
    logger.info("Bucket hit rates:\n%s", bucket_df.to_string(index=False))
    return bucket_df


def plot_bucket_hit_rates(
    bucket_df: pd.DataFrame,
    save_path: Path | None = None,
) -> None:
    overall_hr_rate = (
        bucket_df["n_hrs"].sum() / bucket_df["n_predictions"].sum()
        if bucket_df["n_predictions"].sum() > 0
        else 0.0
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(bucket_df))
    ax.bar(x, bucket_df["hit_rate"], color="steelblue", alpha=0.8, label="Actual hit rate")
    ax.plot(
        x, bucket_df["avg_predicted"], "ro-",
        markersize=8, label="Avg predicted probability",
    )
    ax.axhline(
        y=overall_hr_rate, color="gray", linestyle="--",
        label=f"Base rate ({overall_hr_rate:.4f})",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(bucket_df["bucket"], rotation=45, ha="right")
    ax.set_xlabel("Predicted probability bucket")
    ax.set_ylabel("HR rate")
    ax.set_title("Actual Hit Rate vs Predicted Probability Bucket")
    ax.legend(loc="upper left")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Bucket hit rates plot saved to %s", save_path)
    else:
        plt.show()

    plt.close(fig)


def _compute_bet_profits(
    y_true: np.ndarray,
    market_probs: np.ndarray,
    bet_mask: np.ndarray,
    bet_fraction: float,
) -> np.ndarray:
    bet_true = y_true[bet_mask]
    bet_market = market_probs[bet_mask]
    decimal_odds = 1.0 / bet_market

    profits = np.where(
        bet_true == 1,
        (decimal_odds - 1.0) * bet_fraction,
        -bet_fraction,
    )
    return profits


def backtest_betting_strategy(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    market_probs: np.ndarray | None = None,
    edge_threshold: float = 0.02,
    bet_fraction: float = 0.01,
) -> dict:
    if market_probs is None:
        base_rate = y_true.mean() if len(y_true) > 0 else 0.03
        market_probs = np.full_like(y_pred_proba, base_rate)

    edges = y_pred_proba - market_probs
    bet_mask = edges > edge_threshold

    n_bets = int(bet_mask.sum())
    if n_bets == 0:
        logger.info("Backtest: no bets placed (edge_threshold=%.4f)", edge_threshold)
        return {
            "n_bets": 0,
            "n_wins": 0,
            "win_rate": float("nan"),
            "total_profit": 0.0,
            "roi": float("nan"),
            "max_drawdown": 0.0,
            "avg_edge": float("nan"),
            "kelly_fraction": float("nan"),
        }

    profits = _compute_bet_profits(y_true, market_probs, bet_mask, bet_fraction)
    cumulative_pnl = np.cumsum(profits)
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdowns = running_max - cumulative_pnl
    max_drawdown = float(drawdowns.max())

    n_wins = int(y_true[bet_mask].sum())
    total_profit = float(profits.sum())
    roi = total_profit / (n_bets * bet_fraction)
    avg_edge = float(edges[bet_mask].mean())

    bet_market = market_probs[bet_mask]
    decimal_odds = 1.0 / bet_market
    kelly_values = edges[bet_mask] / (decimal_odds - 1.0)
    kelly_fraction = float(np.median(kelly_values))

    result = {
        "n_bets": n_bets,
        "n_wins": n_wins,
        "win_rate": round(n_wins / n_bets, 4),
        "total_profit": round(total_profit, 4),
        "roi": round(roi, 4),
        "max_drawdown": round(max_drawdown, 4),
        "avg_edge": round(avg_edge, 4),
        "kelly_fraction": round(kelly_fraction, 4),
    }

    logger.info(
        "Backtest — Bets: %d | Wins: %d (%.1f%%) | ROI: %.2f%% | "
        "Max DD: %.4f | Avg Edge: %.4f | Kelly: %.4f",
        n_bets, n_wins, result["win_rate"] * 100, result["roi"] * 100,
        max_drawdown, avg_edge, kelly_fraction,
    )
    return result


def compute_edge_distribution(
    y_pred_proba: np.ndarray,
    market_probs: np.ndarray,
    bins: list[float] | None = None,
) -> pd.DataFrame:
    if bins is None:
        bins = [-1.0, -0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04, 0.06, 0.10, 1.0]

    edges = y_pred_proba - market_probs
    labels = [
        f"{bins[i]:+.2f} to {bins[i + 1]:+.2f}"
        for i in range(len(bins) - 1)
    ]
    bucket_indices = np.digitize(edges, bins) - 1
    bucket_indices = np.clip(bucket_indices, 0, len(labels) - 1)

    rows: list[dict] = []
    for i, label in enumerate(labels):
        mask = bucket_indices == i
        rows.append({
            "edge_bin": label,
            "count": int(mask.sum()),
            "pct": round(float(mask.sum()) / len(edges) * 100, 2) if len(edges) > 0 else 0.0,
        })

    edge_df = pd.DataFrame(rows)
    logger.info("Edge distribution:\n%s", edge_df.to_string(index=False))
    return edge_df


def _format_metrics_block(m: dict, title: str) -> list[str]:
    lines: list[str] = []
    lines.append(title)
    lines.append("-" * 40)
    lines.append(f"  PR-AUC:        {m.get('pr_auc', 0):.4f}")
    lines.append(f"  ROC-AUC:       {m.get('roc_auc', 0):.4f}")
    lines.append(f"  Precision:     {m.get('precision', 0):.4f}")
    lines.append(f"  Recall:        {m.get('recall', 0):.4f}")
    lines.append(f"  F1 Score:      {m.get('f1', 0):.4f}")
    lines.append(f"  Brier Score:   {m.get('brier_score', 0):.6f} "
                 f"(baseline: {m.get('baseline_brier', 0):.6f})")
    lines.append(f"  Log Loss:      {m.get('log_loss', 0):.4f}")
    lines.append(f"  Threshold:     {m.get('threshold', 0):.4f}")
    lines.append(f"  Positive rate: {m.get('n_positive', 0)}/{m.get('n_total', 0)} "
                 f"({m.get('positive_rate', 0) * 100:.2f}%)")
    lines.append("")
    return lines


def _format_betting_block(m: dict) -> list[str]:
    lines: list[str] = []
    lines.append("Betting Evaluation")
    lines.append("=" * 60)
    lines.append("")

    k_values = sorted(
        {int(k.split("_")[-1]) for k in m if k.startswith("precision_at_")}
    )
    if k_values:
        lines.append("Daily Precision/Recall @ K")
        lines.append("-" * 40)
        for k in k_values:
            p = m.get(f"precision_at_{k}", 0)
            r = m.get(f"recall_at_{k}", 0)
            lines.append(f"  K={k:<4d}  Precision: {p:.4f}  Recall: {r:.4f}")
        lines.append(f"  Days evaluated: {m.get('n_days', 0)}")
        lines.append("")

    if "n_bets" in m:
        lines.append("Backtest Results")
        lines.append("-" * 40)
        lines.append(f"  Bets placed:   {m.get('n_bets', 0)}")
        lines.append(f"  Wins:          {m.get('n_wins', 0)}")
        lines.append(f"  Win rate:      {m.get('win_rate', 0):.4f}")
        lines.append(f"  Total profit:  {m.get('total_profit', 0):.4f}")
        lines.append(f"  ROI:           {m.get('roi', 0):.4f}")
        lines.append(f"  Max drawdown:  {m.get('max_drawdown', 0):.4f}")
        lines.append(f"  Avg edge:      {m.get('avg_edge', 0):.4f}")
        lines.append(f"  Kelly frac:    {m.get('kelly_fraction', 0):.4f}")
        lines.append("")

    return lines


def _format_cv_summary_block(
    summary: dict,
    fold_metrics: list[dict] | None = None,
) -> list[str]:
    lines: list[str] = []
    lines.append("Walk-Forward Cross-Validation")
    lines.append("=" * 60)
    lines.append("")

    n_folds = summary.get("n_folds", 0)
    lines.append(f"  Folds: {n_folds}")

    for metric in ["pr_auc", "roc_auc", "brier_score", "log_loss"]:
        mean_key = f"{metric}_mean"
        std_key = f"{metric}_std"
        min_key = f"{metric}_min"
        max_key = f"{metric}_max"
        if mean_key in summary:
            lines.append(
                f"  {metric:<14s}  "
                f"{summary[mean_key]:.4f} ± {summary[std_key]:.4f}  "
                f"[{summary[min_key]:.4f}, {summary[max_key]:.4f}]"
            )

    if fold_metrics:
        lines.append("")
        lines.append("  Per-Fold PR-AUC:")
        for i, fm in enumerate(fold_metrics):
            lines.append(f"    Fold {i:>2d}: {fm.get('pr_auc', 0):.4f}")

    lines.append("")
    return lines


def save_evaluation_report(
    metrics: dict,
    feature_names: list[str],
    importances: np.ndarray,
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    game_metrics: dict | None = None,
    calibrated_proba: np.ndarray | None = None,
    calibrated_metrics: dict | None = None,
    game_metrics_calibrated: dict | None = None,
    betting_metrics: dict | None = None,
    bucket_df: pd.DataFrame | None = None,
    game_dates: np.ndarray | None = None,
    multi_threshold_df: pd.DataFrame | None = None,
    prob_dist_raw: dict | None = None,
    prob_dist_calibrated: dict | None = None,
    edge_dist_df: pd.DataFrame | None = None,
    cv_summary: dict | None = None,
    cv_fold_metrics: list[dict] | None = None,
    output_dir: Path | None = None,
) -> None:
    output_dir = Path(output_dir) if output_dir is not None else EVALUATION_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "evaluation_report.txt"
    sorted_indices = np.argsort(importances)[::-1][:TOP_N_FEATURES]

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("MODEL EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append("")

    lines.extend(_format_metrics_block(metrics, "PA-Level Performance Metrics (Raw)"))

    if game_metrics is not None:
        lines.extend(_format_metrics_block(game_metrics, "Game-Level Composition Metrics (Raw)"))

    if calibrated_metrics is not None:
        lines.extend(_format_metrics_block(calibrated_metrics, "PA-Level Performance Metrics (Calibrated)"))

    if game_metrics_calibrated is not None:
        lines.extend(_format_metrics_block(game_metrics_calibrated, "Game-Level Composition Metrics (Calibrated)"))

    if cv_summary is not None and cv_summary.get("n_folds", 0) > 0:
        lines.extend(_format_cv_summary_block(cv_summary, cv_fold_metrics))

    if prob_dist_raw is not None:
        lines.append("Probability Distribution (Raw)")
        lines.append("-" * 40)
        for k, v in prob_dist_raw.items():
            lines.append(f"  {k:<8s} {v:.6f}")
        lines.append("")

    if prob_dist_calibrated is not None:
        lines.append("Probability Distribution (Calibrated)")
        lines.append("-" * 40)
        for k, v in prob_dist_calibrated.items():
            lines.append(f"  {k:<8s} {v:.6f}")
        lines.append("")

    if edge_dist_df is not None:
        lines.append("Edge Distribution (model - market)")
        lines.append("-" * 40)
        lines.append(edge_dist_df.to_string(index=False))
        lines.append("")

    if multi_threshold_df is not None:
        lines.append("Multi-Threshold Precision/Recall")
        lines.append("-" * 40)
        lines.append(multi_threshold_df.to_string(index=False))
        lines.append("")

    lines.append(f"Top {TOP_N_FEATURES} Features by Importance")
    lines.append("-" * 40)

    for rank, idx in enumerate(sorted_indices, start=1):
        lines.append(f"  {rank:>3}. {feature_names[idx]:<40s} {importances[idx]:.6f}")

    lines.append("")

    if betting_metrics is not None:
        lines.extend(_format_betting_block(betting_metrics))

    if bucket_df is not None:
        lines.append("Bucket Hit Rates")
        lines.append("-" * 40)
        lines.append(bucket_df.to_string(index=False))
        lines.append("")

    lines.append("=" * 60)

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Evaluation report saved to %s", report_path)

    plot_precision_recall_curve(
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        save_path=output_dir / "precision_recall_curve.png",
    )

    plot_feature_importance(
        feature_names=feature_names,
        importances=importances,
        save_path=output_dir / "feature_importance.png",
    )

    plot_calibration_curve(
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        label="Raw",
        calibrated_proba=calibrated_proba,
        save_path=output_dir / "calibration_curve.png",
    )

    if bucket_df is not None:
        plot_bucket_hit_rates(
            bucket_df=bucket_df,
            save_path=output_dir / "bucket_hit_rates.png",
        )

    logger.info("All evaluation artifacts saved to %s", output_dir)
