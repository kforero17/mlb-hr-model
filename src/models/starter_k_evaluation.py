import logging
from math import floor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.calibration import calibration_curve

from config.model_config import (
    STARTER_K_EVALUATION_DIR,
    STARTER_K_OVER_UNDER_LINES,
    TOP_N_FEATURES,
)

logger = logging.getLogger(__name__)


def evaluate_k_count(
    y_true: np.ndarray,
    lambda_pred: np.ndarray,
) -> dict:
    mae = float(np.mean(np.abs(y_true - lambda_pred)))
    rmse = float(np.sqrt(np.mean((y_true - lambda_pred) ** 2)))

    ratio = np.maximum(y_true, 1e-10) / np.maximum(lambda_pred, 1e-10)
    poisson_deviance = float(
        2 * np.mean(y_true * np.log(ratio) - (y_true - lambda_pred))
    )

    mean_predicted = float(np.mean(lambda_pred))
    mean_actual = float(np.mean(y_true))

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "poisson_deviance": poisson_deviance,
        "mean_predicted": mean_predicted,
        "mean_actual": mean_actual,
    }

    logger.info(
        "K count evaluation — MAE: %.3f | RMSE: %.3f | "
        "Poisson deviance: %.4f | Mean pred: %.2f | Mean actual: %.2f",
        mae, rmse, poisson_deviance, mean_predicted, mean_actual,
    )

    return metrics


def fit_dispersion(
    y_true: np.ndarray,
    lambda_pred: np.ndarray,
) -> float:
    residual_var = np.mean((y_true - lambda_pred) ** 2)
    mean_lambda = np.mean(lambda_pred)
    dispersion = residual_var / mean_lambda
    dispersion = float(np.clip(dispersion, 1.01, 3.0))

    logger.info(
        "Fitted dispersion ratio: %.4f (residual_var=%.4f, mean_lambda=%.4f)",
        dispersion, residual_var, mean_lambda,
    )

    return dispersion


def predict_over_under(
    lambda_pred: np.ndarray,
    line: float,
    dispersion: float,
) -> np.ndarray:
    lambda_pred = np.asarray(lambda_pred, dtype=float)
    p_over = np.zeros_like(lambda_pred)
    threshold = floor(line)

    valid = lambda_pred > 0
    lam = lambda_pred[valid]
    r = lam / (dispersion - 1)
    p = r / (r + lam)
    p_over[valid] = 1.0 - nbinom.cdf(threshold, r, p)

    return p_over


def evaluate_over_under(
    y_true: np.ndarray,
    lambda_pred: np.ndarray,
    lines: list[float],
    dispersion: float,
) -> pd.DataFrame:
    rows: list[dict] = []

    for line in lines:
        p_over = predict_over_under(lambda_pred, line, dispersion)
        actual_over = (y_true > line).astype(int)
        n_total = len(y_true)
        n_over = int(actual_over.sum())

        brier_score = float(np.mean((p_over - actual_over) ** 2))
        predicted_over = (p_over > 0.5).astype(int)
        accuracy = float(np.mean(predicted_over == actual_over))
        mean_p_over = float(np.mean(p_over))
        actual_over_rate = n_over / n_total if n_total > 0 else 0.0

        rows.append({
            "line": line,
            "brier_score": brier_score,
            "accuracy": accuracy,
            "mean_p_over": mean_p_over,
            "actual_over_rate": actual_over_rate,
            "n_over": n_over,
            "n_total": n_total,
        })

    ou_df = pd.DataFrame(rows)
    logger.info("Over/under evaluation:\n%s", ou_df.to_string(index=False))
    return ou_df


def compute_baselines(starter_df: pd.DataFrame) -> dict:
    baselines: dict = {}

    if "n_k_against" not in starter_df.columns:
        logger.warning("n_k_against column not found — skipping baselines")
        return baselines

    if "pitcher" not in starter_df.columns:
        logger.warning("pitcher column not found — skipping baselines")
        return baselines

    df = starter_df.sort_values(["pitcher", "game_date"]).copy()
    actual = df["n_k_against"].values

    df["rolling_50g_k"] = (
        df.groupby("pitcher")["n_k_against"]
        .transform(lambda x: x.shift(1).rolling(50, min_periods=5).mean())
    )
    mask_rolling = df["rolling_50g_k"].notna()
    if mask_rolling.sum() > 0:
        mae_rolling = float(
            np.mean(np.abs(actual[mask_rolling] - df.loc[mask_rolling, "rolling_50g_k"].values))
        )
        baselines["rolling_50g_mean"] = mae_rolling
        logger.info("Baseline rolling 50g MAE: %.3f (n=%d)", mae_rolling, mask_rolling.sum())

    df["season_avg_k"] = (
        df.groupby("pitcher")["n_k_against"]
        .transform(lambda x: x.shift(1).expanding(min_periods=3).mean())
    )
    mask_season = df["season_avg_k"].notna()
    if mask_season.sum() > 0:
        mae_season = float(
            np.mean(np.abs(actual[mask_season] - df.loc[mask_season, "season_avg_k"].values))
        )
        baselines["season_avg"] = mae_season
        logger.info("Baseline season avg MAE: %.3f (n=%d)", mae_season, mask_season.sum())

    overall_mean = float(actual.mean())
    mae_league = float(np.mean(np.abs(actual - overall_mean)))
    baselines["league_avg"] = mae_league
    logger.info("Baseline league avg MAE: %.3f (mean=%.2f)", mae_league, overall_mean)

    return baselines


def _plot_k_distribution(
    y_true: np.ndarray,
    lambda_pred: np.ndarray,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    max_k = int(max(y_true.max(), np.ceil(lambda_pred.max()))) + 2
    bins = np.arange(-0.5, max_k + 0.5, 1)

    ax.hist(y_true, bins=bins, alpha=0.6, label="Actual K", density=True)
    ax.hist(lambda_pred, bins=50, alpha=0.5, label="Predicted lambda", density=True)
    ax.set_xlabel("Strikeouts")
    ax.set_ylabel("Density")
    ax.set_title("Predicted vs Actual Strikeout Distribution")
    ax.legend(loc="best")
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    logger.info("K distribution plot saved to %s", save_path)
    plt.close(fig)


def _plot_ou_calibration(
    y_true: np.ndarray,
    lambda_pred: np.ndarray,
    line: float,
    dispersion: float,
    save_path: Path,
) -> None:
    p_over = predict_over_under(lambda_pred, line, dispersion)
    actual_over = (y_true > line).astype(int)

    try:
        fraction_pos, mean_pred = calibration_curve(
            actual_over, p_over, n_bins=10, strategy="uniform",
        )
    except ValueError:
        logger.warning("Could not compute calibration curve for line %.1f", line)
        return

    brier = float(np.mean((p_over - actual_over) ** 2))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        mean_pred, fraction_pos, "s-",
        label=f"O/U {line} (Brier={brier:.4f})",
    )
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted P(over)")
    ax.set_ylabel("Fraction actually over")
    ax.set_title(f"O/U {line} Calibration Curve")
    ax.legend(loc="best")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    logger.info("O/U calibration curve saved to %s", save_path)
    plt.close(fig)


def _plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    top_n: int,
    save_path: Path,
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
    ax.set_title(f"Top {len(sorted_names)} Feature Importances (Starter K)")
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    logger.info("Feature importance plot saved to %s", save_path)
    plt.close(fig)


def save_starter_k_report(
    count_metrics: dict,
    ou_metrics_df: pd.DataFrame,
    baselines: dict,
    feature_names: list[str],
    importances: np.ndarray,
    y_true: np.ndarray,
    lambda_pred: np.ndarray,
    dispersion: float,
    output_dir: Path | None = None,
) -> None:
    output_dir = Path(output_dir) if output_dir is not None else STARTER_K_EVALUATION_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "starter_k_report.txt"
    sorted_indices = np.argsort(importances)[::-1][:TOP_N_FEATURES]

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("STARTER K MODEL EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append("")

    lines.append("Count Accuracy")
    lines.append("-" * 40)
    lines.append(f"  MAE:              {count_metrics.get('mae', 0):.3f}")
    lines.append(f"  RMSE:             {count_metrics.get('rmse', 0):.3f}")
    lines.append(f"  Poisson Deviance: {count_metrics.get('poisson_deviance', 0):.4f}")
    lines.append(f"  Mean Predicted:   {count_metrics.get('mean_predicted', 0):.2f}")
    lines.append(f"  Mean Actual:      {count_metrics.get('mean_actual', 0):.2f}")
    lines.append(f"  Dispersion:       {dispersion:.4f}")
    lines.append("")

    lines.append("Over/Under Accuracy")
    lines.append("-" * 40)
    lines.append(ou_metrics_df.to_string(index=False))
    lines.append("")

    if baselines:
        lines.append("Baseline Comparison (MAE)")
        lines.append("-" * 40)
        lines.append(f"  Model MAE:        {count_metrics.get('mae', 0):.3f}")
        for name, mae_val in baselines.items():
            lines.append(f"  {name:<20s} {mae_val:.3f}")
        lines.append("")

    lines.append(f"Top {TOP_N_FEATURES} Features by Importance")
    lines.append("-" * 40)
    for rank, idx in enumerate(sorted_indices, start=1):
        lines.append(f"  {rank:>3}. {feature_names[idx]:<40s} {importances[idx]:.6f}")
    lines.append("")

    lines.append("=" * 60)

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Starter K report saved to %s", report_path)

    _plot_k_distribution(
        y_true=y_true,
        lambda_pred=lambda_pred,
        save_path=output_dir / "k_distribution.png",
    )

    _plot_ou_calibration(
        y_true=y_true,
        lambda_pred=lambda_pred,
        line=6.5,
        dispersion=dispersion,
        save_path=output_dir / "ou_calibration_6_5.png",
    )

    _plot_feature_importance(
        feature_names=feature_names,
        importances=importances,
        top_n=TOP_N_FEATURES,
        save_path=output_dir / "feature_importance_starter_k.png",
    )

    logger.info("All starter K evaluation artifacts saved to %s", output_dir)
