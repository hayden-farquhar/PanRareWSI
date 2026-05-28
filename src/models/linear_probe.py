"""
PanRareWSI — L2-regularised logistic regression linear probe.

Shared between Phase 3 (baseline replication) and Phase 4 (rare-cohort benchmark).
Uses mean-pooled UNI2-h slide embeddings (1536-dim) as input features.

Pre-registration reference: §8 (linear probe as primary model).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    auroc: float
    auprc: float
    brier: float
    n_train: int
    n_test: int
    n_pos_train: int
    n_pos_test: int
    best_C: float
    probas: np.ndarray
    labels: np.ndarray


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    Cs: int = 10,
    cv_inner: int = 3,
    max_iter: int = 5000,
    random_state: int = 42,
) -> ProbeResult:
    """Train L2-logistic regression with inner CV for C selection, evaluate on held-out test."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegressionCV(
        Cs=Cs,
        cv=min(cv_inner, int(y_train.sum()), int((1 - y_train).sum())),
        penalty="l2",
        scoring="roc_auc",
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state,
        class_weight="balanced",
    )
    clf.fit(X_train_s, y_train)

    probas = clf.predict_proba(X_test_s)[:, 1]

    if len(np.unique(y_test)) < 2:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = roc_auc_score(y_test, probas)
        auprc = average_precision_score(y_test, probas)

    brier = brier_score_loss(y_test, probas)

    return ProbeResult(
        auroc=auroc,
        auprc=auprc,
        brier=brier,
        n_train=len(y_train),
        n_test=len(y_test),
        n_pos_train=int(y_train.sum()),
        n_pos_test=int(y_test.sum()),
        best_C=float(clf.C_[0]),
        probas=probas,
        labels=y_test,
    )


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn=roc_auc_score,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a metric. Returns (point, lower, upper)."""
    rng = np.random.RandomState(seed)
    point = metric_fn(y_true, y_score)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        boots.append(metric_fn(y_true[idx], y_score[idx]))
    alpha = (1 - ci) / 2
    lower = np.percentile(boots, 100 * alpha)
    upper = np.percentile(boots, 100 * (1 - alpha))
    return point, lower, upper
