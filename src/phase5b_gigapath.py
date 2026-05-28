"""
PanRareWSI — Phase 5b: Prov-GigaPath comparison + UNI2/GigaPath ensemble (§5.2, §6.3).

Loads pre-extracted Prov-GigaPath slide embeddings (seandavis/tcga_provgigapath_embeddings,
layer 13, 768-d), runs the same linear-probe benchmark on the SAME frozen splits and
cells, and builds a UNI2+GigaPath ensemble (mean of out-of-fold predicted probabilities).
Reports UNI2 vs GigaPath vs ensemble AUROC per cell.

Usage:
    python3 -m src.phase5b_gigapath
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.metrics import roc_auc_score

from src.models.linear_probe import train_and_evaluate
from src.phase4_benchmark import (
    BIOMARKER_CELLS, COHORTS, load_cohort_data, parse_patient_id, _get_binary_labels,
)

logger = logging.getLogger(__name__)

GIGAPATH_LAYER = 13  # final slide representation


def parse_sample_type(filename: str) -> str:
    """Sample type from TCGA filename: TCGA-XX-XXXX-01Z-... → '01'."""
    bc = filename.split(".")[0]
    parts = bc.split("-")
    return parts[3][:2] if len(parts) > 3 else "??"


def load_gigapath_features(project_root: Path) -> dict[str, dict[str, np.ndarray]]:
    """Return {cohort: {patient_id: mean layer-13 embedding over primary slides}}."""
    p = hf_hub_download("seandavis/tcga_provgigapath_embeddings",
                        "provgigapath_embeddings_with_metadata.parquet", repo_type="dataset")
    df = pd.read_parquet(p)

    by_cohort = {}
    for cohort in COHORTS:
        sub = df[df["cancer type abbreviation"] == cohort]
        patient_feats = {}
        for _, row in sub.iterrows():
            if parse_sample_type(row["filename"]) != "01":
                continue  # primary tumour only
            pid = row["_PATIENT"]
            layer13 = np.array(row["embedding"][GIGAPATH_LAYER], dtype=np.float32)
            patient_feats.setdefault(pid, []).append(layer13)
        by_cohort[cohort] = {pid: np.mean(v, axis=0) for pid, v in patient_feats.items()}
        logger.info(f"  GigaPath {cohort}: {len(by_cohort[cohort])} patients")
    return by_cohort


def run(project_root: Path | None = None):
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    phase4 = json.loads((project_root / "results" / "phase4_benchmark.json").read_text())
    ok_cells = {r["cell"]: r for r in phase4 if r["status"] == "ok"}

    logger.info("Loading Prov-GigaPath features...")
    giga = load_gigapath_features(project_root)

    results = []
    for cohort in COHORTS:
        try:
            u_features, u_slide_ids, labels, splits = load_cohort_data(cohort, project_root)
        except Exception:
            continue

        # UNI2 patient-level features
        slide_patient = {sid: parse_patient_id(sid) for sid in u_slide_ids}
        u_pf = {}
        for i, sid in enumerate(u_slide_ids):
            u_pf.setdefault(slide_patient[sid], []).append(u_features[i])
        u_pmean = {pid: np.mean(f, axis=0) for pid, f in u_pf.items()}

        g_pmean = giga.get(cohort, {})
        smap = dict(zip(splits["patient_id"], splits["fold"]))

        for cell in BIOMARKER_CELLS.get(cohort, []):
            cell_name = f"{cohort}/{cell['name']}"
            if cell_name not in ok_cells:
                continue
            y_series = _get_binary_labels(labels, cell)
            lmap = dict(zip(labels["patient_id"], y_series))

            # Patients with BOTH UNI2 and GigaPath features + label + split
            pids, Xu, Xg, y, folds = [], [], [], [], []
            for pid in u_pmean:
                if (pid in g_pmean and pid in lmap and pid in smap and pd.notna(lmap[pid])):
                    pids.append(pid); Xu.append(u_pmean[pid]); Xg.append(g_pmean[pid])
                    y.append(lmap[pid]); folds.append(smap[pid])
            Xu, Xg, y, folds = np.array(Xu), np.array(Xg), np.array(y), np.array(folds)
            n_pos = int(y.sum()) if len(y) else 0
            if len(y) < 10 or len(np.unique(y)) < 2 or n_pos < 5 or (len(y) - n_pos) < 5:
                continue

            # 5-fold OOF predictions for UNI2, GigaPath
            u_oof = np.full(len(y), np.nan)
            g_oof = np.full(len(y), np.nan)
            for fi in range(5):
                te = folds == fi; tr = ~te
                if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
                    continue
                u_oof[te] = train_and_evaluate(Xu[tr], y[tr], Xu[te], y[te]).probas
                g_oof[te] = train_and_evaluate(Xg[tr], y[tr], Xg[te], y[te]).probas

            valid = ~np.isnan(u_oof) & ~np.isnan(g_oof)
            if valid.sum() < 10 or len(np.unique(y[valid])) < 2:
                continue
            ens_oof = (u_oof + g_oof) / 2.0

            u_auroc = roc_auc_score(y[valid], u_oof[valid])
            g_auroc = roc_auc_score(y[valid], g_oof[valid])
            e_auroc = roc_auc_score(y[valid], ens_oof[valid])

            results.append({
                "cell": cell_name, "tier": cell["tier"], "n": int(valid.sum()), "n_pos": n_pos,
                "uni2_auroc": round(u_auroc, 4),
                "gigapath_auroc": round(g_auroc, 4),
                "ensemble_auroc": round(e_auroc, 4),
                "ensemble_vs_best_single": round(e_auroc - max(u_auroc, g_auroc), 4),
            })
            logger.info(f"  {cell_name:26s} UNI2={u_auroc:.3f} GigaPath={g_auroc:.3f} "
                        f"Ens={e_auroc:.3f} (Δvs-best {results[-1]['ensemble_vs_best_single']:+.3f})")

    # Summary
    u = [r["uni2_auroc"] for r in results]
    g = [r["gigapath_auroc"] for r in results]
    e = [r["ensemble_auroc"] for r in results]
    ens_helps = sum(1 for r in results if r["ensemble_vs_best_single"] > 0.02)
    summary = {
        "n_cells": len(results),
        "mean_uni2": round(float(np.mean(u)), 4),
        "mean_gigapath": round(float(np.mean(g)), 4),
        "mean_ensemble": round(float(np.mean(e)), 4),
        "uni2_beats_gigapath": sum(1 for r in results if r["uni2_auroc"] > r["gigapath_auroc"]),
        "ensemble_improves_gt0.02": ens_helps,
        "cells": results,
    }
    out_path = project_root / "results" / "phase5b_gigapath.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("PHASE 5b: UNI2 vs Prov-GigaPath vs Ensemble")
    logger.info(f"{'='*60}")
    logger.info(f"Cells compared: {len(results)}")
    logger.info(f"Mean AUROC — UNI2: {summary['mean_uni2']:.3f}, GigaPath: {summary['mean_gigapath']:.3f}, Ensemble: {summary['mean_ensemble']:.3f}")
    logger.info(f"UNI2 > GigaPath in {summary['uni2_beats_gigapath']}/{len(results)} cells")
    logger.info(f"Ensemble improves >0.02 over best single in {ens_helps}/{len(results)} cells")
    logger.info(f"Saved: {out_path}")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
