"""
PanRareWSI — Permutation tests with BH-FDR correction.

Runs 10,000 label permutations per cell to compute p-values for AUROC > 0.50.
Full 5-fold CV repeated each permutation. BH-FDR correction across primary cells.

Pre-registration §10.6, §12.

Usage:
    python3 -m src.permutation_tests [--n-perms 10000]
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.phase4_benchmark import (
    BIOMARKER_CELLS, COHORTS, load_cohort_data, parse_patient_id, _get_binary_labels,
)

logger = logging.getLogger(__name__)


def _run_cv_auroc(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> float:
    """Run 5-fold CV and return pooled AUROC. Lightweight version for permutations."""
    all_probas = np.full(len(y), np.nan)

    for fold_idx in range(5):
        test_mask = folds == fold_idx
        train_mask = ~test_mask

        y_train, y_test = y[train_mask], y[test_mask]
        if len(np.unique(y_test)) < 2 or len(np.unique(y_train)) < 2:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X[train_mask])
        X_test_s = scaler.transform(X[test_mask])

        n_pos = int(y_train.sum())
        inner_cv = min(3, n_pos, int(len(y_train) - n_pos))
        if inner_cv < 2:
            inner_cv = 2

        clf = LogisticRegressionCV(
            Cs=5, cv=inner_cv, penalty="l2", scoring="roc_auc",
            solver="lbfgs", max_iter=2000, random_state=42,
            class_weight="balanced",
        )
        clf.fit(X_train_s, y_train)
        all_probas[test_mask] = clf.predict_proba(X_test_s)[:, 1]

    valid = ~np.isnan(all_probas)
    if valid.sum() < 10 or len(np.unique(y[valid])) < 2:
        return 0.5
    return roc_auc_score(y[valid], all_probas[valid])


def permutation_test_cell(
    X: np.ndarray, y: np.ndarray, folds: np.ndarray,
    observed_auroc: float, n_perms: int = 10000, seed: int = 42,
) -> dict:
    """Permutation test for one cell."""
    rng = np.random.RandomState(seed)
    null_aurocs = []

    for _ in tqdm(range(n_perms), desc="Perms", unit="perm", leave=False):
        y_perm = rng.permutation(y)
        null_auroc = _run_cv_auroc(X, y_perm, folds)
        null_aurocs.append(null_auroc)

    null_aurocs = np.array(null_aurocs)
    p_value = (np.sum(null_aurocs >= observed_auroc) + 1) / (n_perms + 1)

    return {
        "observed_auroc": observed_auroc,
        "p_value": p_value,
        "null_mean": float(np.mean(null_aurocs)),
        "null_std": float(np.std(null_aurocs)),
        "n_perms": n_perms,
    }


def bh_fdr(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR correction. Returns list of significant flags."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    significant = [False] * n
    for rank, (orig_idx, p) in enumerate(indexed, 1):
        threshold = (rank / n) * alpha
        if p <= threshold:
            significant[orig_idx] = True
        else:
            break
    return significant


def run_permutation_tests(
    project_root: Path | None = None, n_perms: int = 10000,
) -> list[dict]:
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    # Load Phase 4 results for observed AUROCs
    phase4_path = project_root / "results" / "phase4_benchmark.json"
    phase4_results = json.loads(phase4_path.read_text())
    ok_cells = {r["cell"]: r for r in phase4_results if r["status"] == "ok"}

    perm_results = []

    for cohort in COHORTS:
        try:
            features, slide_ids, labels, splits = load_cohort_data(cohort, project_root)
        except Exception as e:
            logger.error(f"{cohort}: {e}")
            continue

        slide_patient = {sid: parse_patient_id(sid) for sid in slide_ids}
        patient_features = {}
        for i, sid in enumerate(slide_ids):
            pid = slide_patient[sid]
            if pid not in patient_features:
                patient_features[pid] = []
            patient_features[pid].append(features[i])
        patient_mean = {pid: np.mean(f, axis=0) for pid, f in patient_features.items()}

        split_map = dict(zip(splits["patient_id"], splits["fold"]))

        for cell in BIOMARKER_CELLS.get(cohort, []):
            cell_name = f"{cohort}/{cell['name']}"
            if cell_name not in ok_cells:
                continue

            observed = ok_cells[cell_name]["pooled_auroc"]
            y_series = _get_binary_labels(labels, cell)
            label_map = dict(zip(labels["patient_id"], y_series))

            pids, X, y, folds = [], [], [], []
            for pid, feat in patient_mean.items():
                if pid in label_map and pid in split_map and pd.notna(label_map[pid]):
                    pids.append(pid)
                    X.append(feat)
                    y.append(label_map[pid])
                    folds.append(split_map[pid])

            X, y, folds = np.array(X), np.array(y), np.array(folds)

            logger.info(f"{cell_name}: n={len(y)}, observed AUROC={observed:.3f}")
            result = permutation_test_cell(X, y, folds, observed, n_perms=n_perms)
            result["cell"] = cell_name
            result["tier"] = cell["tier"]
            perm_results.append(result)
            logger.info(f"  p={result['p_value']:.4f}, null={result['null_mean']:.3f}±{result['null_std']:.3f}")

    # BH-FDR on primary cells
    primary_results = [r for r in perm_results if r["tier"] == "primary"]
    primary_pvals = [r["p_value"] for r in primary_results]
    significant = bh_fdr(primary_pvals, alpha=0.05)
    for r, sig in zip(primary_results, significant):
        r["bh_significant"] = sig

    for r in perm_results:
        if r["tier"] != "primary":
            r["bh_significant"] = None

    # Save
    out_path = project_root / "results" / "permutation_tests.json"
    with open(out_path, "w") as f:
        json.dump(perm_results, f, indent=2)
    logger.info(f"\nSaved: {out_path}")

    # Summary
    n_sig = sum(1 for r in primary_results if r["bh_significant"])
    logger.info(f"Primary cells: {n_sig}/{len(primary_results)} significant after BH-FDR (α=0.05)")
    for r in sorted(perm_results, key=lambda x: x["p_value"]):
        sig = "***" if r.get("bh_significant") else "   "
        logger.info(f"  {sig} {r['cell']:30s} p={r['p_value']:.4f} AUROC={r['observed_auroc']:.3f}")

    return perm_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    n_perms = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    run_permutation_tests(n_perms=n_perms)
