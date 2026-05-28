"""
PanRareWSI — export per-patient out-of-fold (OOF) predictions.

Regenerates the linear-probe out-of-fold predicted probabilities used for the
pooled AUROC/AUPRC, bootstrap CIs, and calibration analyses, and writes them as
a tidy CSV so the discrimination and calibration results can be re-checked
without access to the gated UNI2-h embeddings.

Reuses the exact Phase-4 pipeline (same mean-pooled features, frozen splits,
deterministic L2 logistic probe, seed 42), so the regenerated OOF probabilities
reproduce the published `results/phase4_benchmark.json` AUROCs. Each cell's
recomputed pooled AUROC is verified against the stored value before writing.

Usage:
    python3 -m src.export_oof_predictions
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.phase4_benchmark import (
    BIOMARKER_CELLS,
    COHORTS,
    _get_binary_labels,
    load_cohort_data,
    parse_patient_id,
)
from src.models.linear_probe import train_and_evaluate


def oof_for_cell(cohort, cell, features, slide_ids, labels, splits):
    """Reproduce the Phase-4 fold loop and return per-patient OOF records.

    Returns (records, pooled_auroc) or (None, None) if the cell is not a
    powered binary cell (skip / multiclass / insufficient n / single-class).
    """
    if cell.get("skip") or cell.get("multiclass"):
        return None, None

    slide_patient = {sid: parse_patient_id(sid) for sid in slide_ids}
    patient_features: dict[str, list] = {}
    for i, sid in enumerate(slide_ids):
        patient_features.setdefault(slide_patient[sid], []).append(features[i])
    patient_mean_features = {
        pid: np.mean(feats, axis=0) for pid, feats in patient_features.items()
    }

    y_series = _get_binary_labels(labels, cell)
    label_map = dict(zip(labels["patient_id"], y_series))
    split_map = dict(zip(splits["patient_id"], splits["fold"]))

    pids, X_all, y_all, fold_all = [], [], [], []
    for pid, feat in patient_mean_features.items():
        if pid in label_map and pid in split_map and pd.notna(label_map[pid]):
            pids.append(pid)
            X_all.append(feat)
            y_all.append(label_map[pid])
            fold_all.append(split_map[pid])

    if not pids:
        return None, None

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    fold_all = np.array(fold_all)

    n_pos, n_neg = int(y_all.sum()), int((1 - y_all).sum())
    if n_pos < 5 or n_neg < 5:
        return None, None

    all_probas = np.full(len(y_all), np.nan)
    for fold_idx in range(5):
        test_mask = fold_all == fold_idx
        train_mask = ~test_mask
        if len(np.unique(y_all[test_mask])) < 2 or len(np.unique(y_all[train_mask])) < 2:
            continue
        result = train_and_evaluate(
            X_all[train_mask], y_all[train_mask], X_all[test_mask], y_all[test_mask]
        )
        all_probas[test_mask] = result.probas

    valid = ~np.isnan(all_probas)
    if valid.sum() < 10 or len(np.unique(y_all[valid])) < 2:
        return None, None

    pooled_auroc = float(roc_auc_score(y_all[valid], all_probas[valid]))
    records = [
        {
            "cohort": cohort,
            "biomarker": cell["name"],
            "tier": cell["tier"],
            "patient_id": pids[i],
            "y_true": int(y_all[i]),
            "oof_proba": round(float(all_probas[i]), 6),
            "fold": int(fold_all[i]),
        }
        for i in range(len(pids))
        if valid[i]
    ]
    return records, pooled_auroc


def main():
    root = Path(__file__).resolve().parent.parent
    stored = {
        r["cell"]: r
        for r in json.loads((root / "results" / "phase4_benchmark.json").read_text())
        if isinstance(r, dict)
    }

    all_records, checks = [], []
    for cohort in COHORTS:
        features, slide_ids, labels, splits = load_cohort_data(cohort, root)
        for cell in BIOMARKER_CELLS.get(cohort, []):
            records, pooled = oof_for_cell(cohort, cell, features, slide_ids, labels, splits)
            if records is None:
                continue
            all_records.extend(records)
            cell_name = f"{cohort}/{cell['name']}"
            ref = stored.get(cell_name, {}).get("pooled_auroc")
            match = ref is not None and abs(round(pooled, 4) - ref) <= 0.0005
            checks.append((cell_name, len(records), round(pooled, 4), ref, match))

    out_csv = root / "results" / "oof_predictions.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["cohort", "biomarker", "tier", "patient_id", "y_true", "oof_proba", "fold"]
        )
        w.writeheader()
        w.writerows(all_records)

    n_ok = sum(1 for *_, m in checks if m)
    print(f"{'cell':28} {'n':>4} {'recomputed':>10} {'published':>10}  match")
    for name, n, rec, ref, m in checks:
        print(f"{name:28} {n:>4} {rec:>10.4f} {str(ref):>10}  {'OK' if m else 'MISMATCH'}")
    print(f"\nCells: {len(checks)} | AUROC verified: {n_ok}/{len(checks)}")
    print(f"Rows written: {len(all_records)} -> {out_csv}")
    if n_ok != len(checks):
        raise SystemExit("Verification failed: regenerated AUROC does not match published value.")


if __name__ == "__main__":
    main()
