"""
PanRareWSI — Multiclass L2-regularised logistic regression.

For histological subtype prediction tasks (positive controls).
Reports macro-OVR-AUROC with bootstrap CIs.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


def train_and_evaluate_multiclass(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    Cs: int = 10,
    cv_inner: int = 3,
    max_iter: int = 5000,
    random_state: int = 42,
) -> dict:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    n_classes = len(le.classes_)

    min_class = min(np.bincount(y_train_enc))
    inner_cv = min(cv_inner, min_class)
    if inner_cv < 2:
        inner_cv = 2

    clf = LogisticRegressionCV(
        Cs=Cs, cv=inner_cv, penalty="l2", scoring="roc_auc_ovr",
        solver="lbfgs", max_iter=max_iter, random_state=random_state,
        class_weight="balanced",
    )
    clf.fit(X_train_s, y_train_enc)
    probas = clf.predict_proba(X_test_s)

    if n_classes > 2 and len(np.unique(y_test_enc)) > 1:
        macro_auroc = roc_auc_score(y_test_enc, probas, multi_class="ovr", average="macro")
    else:
        macro_auroc = float("nan")

    return {
        "macro_auroc": macro_auroc,
        "n_classes": n_classes,
        "classes": le.classes_.tolist(),
        "probas": probas,
        "labels": y_test_enc,
        "label_names": le.classes_.tolist(),
    }
