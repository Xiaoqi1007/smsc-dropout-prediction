"""Common discrimination and calibration metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    auc,
)

METRICS = [
    "brier",
    "log_loss",
    "auroc",
    "auprc",
    "ap",
    "ece_10",
    "mean_probability",
    "prevalence",
    "calibration_gap",
]


def calibration_bins(y, p, n_bins=10):
    y, p = np.asarray(y), np.asarray(p)
    index = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for i in range(n_bins):
        mask = index == i
        rows.append(
            dict(
                bin=i,
                lower=i / n_bins,
                upper=(i + 1) / n_bins,
                n_rows=int(mask.sum()),
                mean_probability=float(p[mask].mean()) if mask.any() else np.nan,
                observed=float(y[mask].mean()) if mask.any() else np.nan,
            )
        )
    return pd.DataFrame(rows)


def evaluate(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if len(y) == 0:
        return {
            **{k: np.nan for k in METRICS},
            "n_rows": 0,
            "n_positive": 0,
            "valid_two_classes": False,
        }
    bins = calibration_bins(y, p)
    ece = float(
        (bins.n_rows / len(y) * (bins.observed - bins.mean_probability).abs()).sum()
    )
    row = dict(
        n_rows=len(y),
        n_positive=int(y.sum()),
        prevalence=float(y.mean()),
        mean_probability=float(p.mean()),
        calibration_gap=float(p.mean() - y.mean()),
        brier=brier_score_loss(y, p),
        log_loss=log_loss(y, p, labels=[0, 1]),
        ece_10=ece,
        valid_two_classes=len(np.unique(y)) == 2,
        auroc=np.nan,
        auprc=np.nan,
        ap=np.nan,
    )
    if row["valid_two_classes"]:
        precision, recall, _ = precision_recall_curve(y, p)
        row.update(
            auroc=roc_auc_score(y, p),
            auprc=auc(recall, precision),
            ap=average_precision_score(y, p),
        )
    return row


def interpolate_roc_pr(y, probability, grid):
    """Retain vertical curve segments when sampling onto a shared plotting grid.

    Collapsing repeated FPR/recall values to their maximum creates an optimistic
    envelope and changes the line between successive operating points.
    """
    fpr, tpr, _ = roc_curve(y, probability)
    precision, recall, _ = precision_recall_curve(y, probability)
    return [np.interp(grid, fpr, tpr), np.interp(grid, recall[::-1], precision[::-1])]
