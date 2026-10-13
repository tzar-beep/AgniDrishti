"""Evaluation metrics and plots (PR curve, calibration curve)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

Curves = dict[str, tuple[np.ndarray, np.ndarray]]


def compute_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """PR-AUC (average precision), ROC-AUC, Brier score, base rate and row count."""
    return {
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "positive_rate": round(float(np.mean(y)), 4),
        "n": int(len(y)),
    }


def plot_calibration(curves: Curves, path: str) -> None:
    """Save a reliability diagram for each named (y, p) pair."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, (y, p) in curves.items():
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, "o-", label=name)
    ax.set(xlabel="Predicted probability", ylabel="Observed frequency", title="Calibration (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_pr(curves: Curves, path: str) -> None:
    """Save precision-recall curves for each named (y, p) pair."""
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (y, p) in curves.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=f"{name} (AP={average_precision_score(y, p):.3f})")
    base = np.mean(next(iter(curves.values()))[0])
    ax.axhline(base, color="k", ls="--", lw=1, label=f"base rate {base:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
