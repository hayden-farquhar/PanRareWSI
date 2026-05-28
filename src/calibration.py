"""
PanRareWSI — Calibration analysis (pre-registration §10.7, RQ2).

For each cell: re-run 5-fold CV to get pooled out-of-fold predictions, then
report pre-calibration ECE (15-bin) and post-calibration ECE after Platt
scaling (primary) and isotonic regression (sensitivity). Recalibration is
fit via nested CV on the pooled OOF predictions to avoid test leakage.

RQ2: Are post-calibration ECE below 0.10?

Usage:
    python3 -m src.calibration
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.models.linear_probe import train_and_evaluate
from src.phase4_benchmark import (
    BIOMARKER_CELLS, COHORTS, load_cohort_data, parse_patient_id, _get_binary_labels,
)

logger = logging.getLogger(__name__)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """ECE with equal-width bins (Naeini et al. 2015)."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob > lo) & (y_prob <= hi) if i > 0 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return ece


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """Return (bin_centers, bin_accuracy, bin_counts) for a reliability diagram."""
    bins = np.linspace(0, 1, n_bins + 1)
    centers, accs, counts = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob > lo) & (y_prob <= hi) if i > 0 else (y_prob >= lo) & (y_prob <= hi)
        centers.append((lo + hi) / 2)
        if mask.sum() == 0:
            accs.append(np.nan)
            counts.append(0)
        else:
            accs.append(y_true[mask].mean())
            counts.append(int(mask.sum()))
    return np.array(centers), np.array(accs), np.array(counts)


def recalibrate_cv(y_true: np.ndarray, y_prob: np.ndarray, method: str = "platt",
                   n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Recalibrate OOF predictions via nested CV (no test leakage)."""
    recal = np.full(len(y_prob), np.nan)
    n_pos = int(y_true.sum())
    splits = min(n_splits, n_pos, int(len(y_true) - n_pos))
    if splits < 2:
        return y_prob  # too few to recalibrate
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(y_prob.reshape(-1, 1), y_true):
        if len(np.unique(y_true[tr])) < 2:
            recal[te] = y_prob[te]
            continue
        if method == "platt":
            lr = LogisticRegression(max_iter=1000)
            lr.fit(y_prob[tr].reshape(-1, 1), y_true[tr])
            recal[te] = lr.predict_proba(y_prob[te].reshape(-1, 1))[:, 1]
        else:  # isotonic
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(y_prob[tr], y_true[tr])
            recal[te] = ir.predict(y_prob[te])
    recal[np.isnan(recal)] = y_prob[np.isnan(recal)]
    return recal


def get_oof_predictions(cohort, cell, features, slide_ids, labels, splits):
    """Re-run 5-fold CV and return pooled out-of-fold (y_true, y_prob)."""
    slide_patient = {sid: parse_patient_id(sid) for sid in slide_ids}
    patient_features = {}
    for i, sid in enumerate(slide_ids):
        pid = slide_patient[sid]
        patient_features.setdefault(pid, []).append(features[i])
    patient_mean = {pid: np.mean(f, axis=0) for pid, f in patient_features.items()}

    y_series = _get_binary_labels(labels, cell)
    label_map = dict(zip(labels["patient_id"], y_series))
    split_map = dict(zip(splits["patient_id"], splits["fold"]))

    X, y, folds = [], [], []
    for pid, feat in patient_mean.items():
        if pid in label_map and pid in split_map and pd.notna(label_map[pid]):
            X.append(feat)
            y.append(label_map[pid])
            folds.append(split_map[pid])
    X, y, folds = np.array(X), np.array(y), np.array(folds)

    all_probas = np.full(len(y), np.nan)
    for fi in range(5):
        te = folds == fi
        tr = ~te
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        res = train_and_evaluate(X[tr], y[tr], X[te], y[te])
        all_probas[te] = res.probas
    valid = ~np.isnan(all_probas)
    return y[valid], all_probas[valid]


def run_calibration(project_root: Path | None = None) -> list[dict]:
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    phase4 = json.loads((project_root / "results" / "phase4_benchmark.json").read_text())
    ok_cells = {r["cell"] for r in phase4 if r["status"] == "ok"}

    results = []
    for cohort in COHORTS:
        try:
            features, slide_ids, labels, splits = load_cohort_data(cohort, project_root)
        except Exception:
            continue
        for cell in BIOMARKER_CELLS.get(cohort, []):
            cell_name = f"{cohort}/{cell['name']}"
            if cell_name not in ok_cells:
                continue

            y_true, y_prob = get_oof_predictions(cohort, cell, features, slide_ids, labels, splits)
            if len(y_true) < 10 or len(np.unique(y_true)) < 2:
                continue

            ece_pre = expected_calibration_error(y_true, y_prob, n_bins=15)
            platt = recalibrate_cv(y_true, y_prob, "platt")
            iso = recalibrate_cv(y_true, y_prob, "isotonic")
            ece_platt = expected_calibration_error(y_true, platt, n_bins=15)
            ece_iso = expected_calibration_error(y_true, iso, n_bins=15)

            results.append({
                "cell": cell_name,
                "cohort": cohort,
                "biomarker": cell["name"],
                "tier": cell["tier"],
                "n": len(y_true),
                "ece_pre": round(ece_pre, 4),
                "ece_platt": round(ece_platt, 4),
                "ece_isotonic": round(ece_iso, 4),
                "platt_below_0.10": bool(ece_platt < 0.10),
            })
            logger.info(f"  {cell_name:28s} ECE: pre={ece_pre:.3f} platt={ece_platt:.3f} iso={ece_iso:.3f} (n={len(y_true)})")

    # Summary
    n_below = sum(1 for r in results if r["platt_below_0.10"])
    logger.info(f"\n{'='*60}")
    logger.info(f"CALIBRATION SUMMARY (RQ2)")
    logger.info(f"{'='*60}")
    logger.info(f"Cells: {len(results)}")
    logger.info(f"Post-Platt ECE < 0.10: {n_below}/{len(results)}")
    logger.info(f"Mean pre-cal ECE: {np.mean([r['ece_pre'] for r in results]):.3f}")
    logger.info(f"Mean post-Platt ECE: {np.mean([r['ece_platt'] for r in results]):.3f}")

    out_path = project_root / "results" / "calibration.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {out_path}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_calibration()
