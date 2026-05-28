"""
PanRareWSI — S2: Mean+variance pooling sensitivity analysis (§5.1).

Re-pools patch embeddings as concat(mean, variance) → 3072-d, re-runs the
linear probe, and compares to mean-only (1536-d). Promotion criterion:
mean+var beats mean-only by >0.02 AUROC in ≥50% of cells.

Local cohorts (6) use local .h5 patches. THYM uses mean+var features computed
on Modal (saved to data/embeddings/THYM/meanvar_features.npy if available;
otherwise THYM is skipped with a note).

Usage:
    python3 -m src.s2_meanvar_pool
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.phase4_benchmark import (
    BIOMARKER_CELLS, COHORTS, load_cohort_data, parse_patient_id, _get_binary_labels,
)

logger = logging.getLogger(__name__)

LOCAL_COHORTS = ["ACC", "UVM", "MESO", "CHOL", "KICH", "DLBC"]


def compute_meanvar(cohort, project_root):
    """Compute concat(mean, var) features per slide from local .h5 patches."""
    embed_dir = project_root / "data" / "embeddings" / cohort
    out_npy = embed_dir / "meanvar_features.npy"
    out_manifest = embed_dir / "meanvar_manifest.json"
    if out_npy.exists():
        return np.load(out_npy), json.loads(out_manifest.read_text())["slide_ids"]

    h5_files = sorted(embed_dir.glob("*.h5"))
    if not h5_files:
        return None, None
    feats, sids = [], []
    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as f:
            arr = f["features"][:][0]  # (n_patches, 1536)
        mv = np.concatenate([arr.mean(axis=0), arr.var(axis=0)])  # (3072,)
        feats.append(mv); sids.append(h5_path.stem)
    feats = np.array(feats, dtype=np.float32)
    np.save(out_npy, feats)
    out_manifest.write_text(json.dumps({"slide_ids": sids}, indent=2))
    return feats, sids


def cv_auroc(X, y, folds):
    probas = np.full(len(y), np.nan)
    for fi in range(5):
        te = folds == fi; tr = ~te
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        n_pos = int(y[tr].sum()); icv = max(2, min(3, n_pos, int(len(y[tr]) - n_pos)))
        clf = LogisticRegressionCV(Cs=10, cv=icv, penalty="l2", scoring="roc_auc",
                                   solver="lbfgs", max_iter=5000, random_state=42,
                                   class_weight="balanced")
        clf.fit(Xtr, y[tr]); probas[te] = clf.predict_proba(Xte)[:, 1]
    v = ~np.isnan(probas)
    return roc_auc_score(y[v], probas[v]) if v.sum() > 10 and len(np.unique(y[v])) > 1 else np.nan


def run(project_root: Path | None = None):
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    phase4 = json.loads((project_root / "results" / "phase4_benchmark.json").read_text())
    ok = {r["cell"]: r for r in phase4 if r["status"] == "ok"}

    results = []
    for cohort in LOCAL_COHORTS:
        feats_mv, sids_mv = compute_meanvar(cohort, project_root)
        if feats_mv is None:
            logger.warning(f"{cohort}: no patches, skipping")
            continue
        labels = pd.read_parquet(project_root / "data" / "labels" / f"{cohort.lower()}_labels.parquet")
        splits = pd.read_csv(project_root / "data" / "splits" / f"{cohort.lower()}_splits.csv")

        # patient-level aggregation of mean+var features
        slide_patient = {sid: parse_patient_id(sid) for sid in sids_mv}
        pf = {}
        for i, sid in enumerate(sids_mv):
            pf.setdefault(slide_patient[sid], []).append(feats_mv[i])
        pmean = {pid: np.mean(f, axis=0) for pid, f in pf.items()}
        smap = dict(zip(splits["patient_id"], splits["fold"]))

        for cell in BIOMARKER_CELLS.get(cohort, []):
            cell_name = f"{cohort}/{cell['name']}"
            if cell_name not in ok:
                continue
            y_series = _get_binary_labels(labels, cell)
            lmap = dict(zip(labels["patient_id"], y_series))
            X, y, folds = [], [], []
            for pid, feat in pmean.items():
                if pid in lmap and pid in smap and pd.notna(lmap[pid]):
                    X.append(feat); y.append(lmap[pid]); folds.append(smap[pid])
            X, y, folds = np.array(X), np.array(y), np.array(folds)
            if len(y) < 10 or len(np.unique(y)) < 2:
                continue
            mv_auroc = cv_auroc(X, y, folds)
            base = ok[cell_name]["pooled_auroc"]
            results.append({
                "cell": cell_name, "base_auroc": base,
                "meanvar_auroc": round(mv_auroc, 4) if not np.isnan(mv_auroc) else None,
                "delta": round(mv_auroc - base, 4) if not np.isnan(mv_auroc) else None,
            })
            logger.info(f"  {cell_name:28s} mean={base:.3f} mean+var={results[-1]['meanvar_auroc']} delta={results[-1]['delta']}")

    improved = sum(1 for r in results if r["delta"] and r["delta"] > 0.02)
    out = {"cells": results, "n_cells": len(results),
           "n_improved_gt_0.02": improved,
           "fraction_improved": round(improved / len(results), 3) if results else 0,
           "promotion_criterion_met": (improved / len(results) >= 0.50) if results else False,
           "note": "THYM excluded (patches on Modal Volume; mean+var would require Modal re-pool)"}
    out_path = project_root / "results" / "s2_meanvar.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"\nS2: {improved}/{len(results)} cells improved >0.02 ({out['fraction_improved']:.0%}); "
                f"promotion {'MET' if out['promotion_criterion_met'] else 'not met'}")
    logger.info(f"Saved: {out_path}")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
