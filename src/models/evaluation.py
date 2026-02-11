import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config.model_config import EVALUATION_DIR, TOP_N_FEATURES

logger = logging.getLogger(__name__)


def evaluate_model(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    y_pred = (y_pred_proba >= threshold).astype(int)

    pr_auc = average_precision_score(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    n_positive = int(y_true.sum())
    n_total = len(y_true)
    positive_rate = n_positive / n_total if n_total > 0 else 0.0

    metrics = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_positive": n_positive,
        "n_total": n_total,
        "positive_rate": positive_rate,
    }

    logger.info(
        "Model evaluation — PR-AUC: %.4f | ROC-AUC: %.4f | "
        "Precision: %.4f | Recall: %.4f | F1: %.4f | "
        "Positive rate: %d/%d (%.2f%%)",
        pr_auc,
        roc_auc,
        precision,
        recall,
        f1,
        n_positive,
        n_total,
        positive_rate * 100,
    )

    return metrics


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


def save_evaluation_report(
    metrics: dict,
    feature_names: list[str],
    importances: np.ndarray,
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
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
    lines.append("Performance Metrics")
    lines.append("-" * 40)
    lines.append(f"  PR-AUC:        {metrics.get('pr_auc', 0):.4f}")
    lines.append(f"  ROC-AUC:       {metrics.get('roc_auc', 0):.4f}")
    lines.append(f"  Precision:     {metrics.get('precision', 0):.4f}")
    lines.append(f"  Recall:        {metrics.get('recall', 0):.4f}")
    lines.append(f"  F1 Score:      {metrics.get('f1', 0):.4f}")
    lines.append(f"  Positive rate: {metrics.get('n_positive', 0)}/{metrics.get('n_total', 0)} "
                 f"({metrics.get('positive_rate', 0) * 100:.2f}%)")
    lines.append("")
    lines.append(f"Top {TOP_N_FEATURES} Features by Importance")
    lines.append("-" * 40)

    for rank, idx in enumerate(sorted_indices, start=1):
        lines.append(f"  {rank:>3}. {feature_names[idx]:<40s} {importances[idx]:.6f}")

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

    logger.info("All evaluation artifacts saved to %s", output_dir)
