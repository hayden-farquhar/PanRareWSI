"""
PanRareWSI — Pre-registered sensitivity analyses (§11 registry).

Covers the analytical sensitivity analyses that reuse existing infrastructure:
  S1  PCA dimensionality reduction (95% variance) before linear probe
  S3  LOOCV instead of 5-fold CV (3 smallest cohorts: CHOL, DLBC, KICH)
  S5  BY-adjusted p-values instead of BH-FDR (22 primary cells)
  S6  Threshold sensitivity at 0.60 and 0.70
  S9  Continuous regression (Spearman r) for Q75-binarised cells

Plus the GTF2I-WHO confound test (§4, §10.5) and tripartite outcome classification (§7.5).

S2 (mean+variance pooling) and S7/S8 (site-aware) are in separate scripts due to
re-pooling / conditional-trigger requirements. S4 (isotonic) is in calibration.py.

Usage:
    python3 -m src.sensitivity_analyses
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.models.linear_probe import train_and_evaluate
from src.phase4_benchmark import (
    BIOMARKER_CELLS, COHORTS, load_cohort_data, parse_patient_id, _get_binary_labels,
)

logger = logging.getLogger(__name__)


def _patient_matrix(cohort, cell, features, slide_ids, labels, splits):
    slide_patient = {sid: parse_patient_id(sid) for sid in slide_ids}
    pf = {}
    for i, sid in enumerate(slide_ids):
        pf.setdefault(slide_patient[sid], []).append(features[i])
    pmean = {pid: np.mean(f, axis=0) for pid, f in pf.items()}
    y_series = _get_binary_labels(labels, cell)
    lmap = dict(zip(labels["patient_id"], y_series))
    smap = dict(zip(splits["patient_id"], splits["fold"]))
    X, y, folds = [], [], []
    for pid, feat in pmean.items():
        if pid in lmap and pid in smap and pd.notna(lmap[pid]):
            X.append(feat); y.append(lmap[pid]); folds.append(smap[pid])
    return np.array(X), np.array(y), np.array(folds)


def _cv_auroc(X, y, folds, pca=False, loocv=False):
    n = len(y)
    probas = np.full(n, np.nan)
    if loocv:
        fold_assign = np.arange(n)  # each patient its own fold
        n_folds = n
    else:
        fold_assign = folds
        n_folds = 5
    unique_folds = range(n_folds) if not loocv else range(n)
    for fi in unique_folds:
        te = fold_assign == fi
        tr = ~te
        if te.sum() == 0:
            continue
        y_tr, y_te = y[tr], y[te]
        if len(np.unique(y_tr)) < 2:
            continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        if pca:
            p = PCA(n_components=0.95, svd_solver="full")
            Xtr = p.fit_transform(Xtr)
            Xte = p.transform(Xte)
        n_pos = int(y_tr.sum())
        icv = min(3, n_pos, int(len(y_tr) - n_pos))
        if icv < 2:
            icv = 2
        clf = LogisticRegressionCV(Cs=10, cv=icv, penalty="l2", scoring="roc_auc",
                                   solver="lbfgs", max_iter=5000, random_state=42,
                                   class_weight="balanced")
        clf.fit(Xtr, y_tr)
        probas[te] = clf.predict_proba(Xte)[:, 1]
    valid = ~np.isnan(probas)
    if valid.sum() < 10 or len(np.unique(y[valid])) < 2:
        return np.nan
    return roc_auc_score(y[valid], probas[valid])


def by_fdr(pvals, alpha=0.05):
    """Benjamini-Yekutieli FDR (valid under arbitrary dependence)."""
    n = len(pvals)
    c_n = np.sum(1.0 / np.arange(1, n + 1))  # harmonic number penalty
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    sig = [False] * n
    for rank, (idx, p) in enumerate(indexed, 1):
        if p <= (rank / (n * c_n)) * alpha:
            sig[idx] = True
        else:
            break
    return sig


def cramers_v(x, y):
    """Cramér's V for two categorical series."""
    from scipy.stats import chi2_contingency
    ct = pd.crosstab(x, y)
    chi2 = chi2_contingency(ct)[0]
    n = ct.sum().sum()
    r, k = ct.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


def classify_tripartite(auroc, ci_low, fdr_sig, rec_thr=0.65, notrec_thr=0.55):
    if auroc >= rec_thr and ci_low > 0.50 and fdr_sig:
        return "recoverable"
    if auroc < notrec_thr or ci_low <= 0.50:
        return "not_recoverable"
    return "inconclusive"


def run_all(project_root: Path | None = None):
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    phase4 = json.loads((project_root / "results" / "phase4_benchmark.json").read_text())
    perm = json.loads((project_root / "results" / "permutation_tests.json").read_text())
    perm_map = {r["cell"]: r for r in perm}
    ok = {r["cell"]: r for r in phase4 if r["status"] == "ok"}

    out = {}

    # ---------- S6: threshold sensitivity + tripartite classification ----------
    logger.info("=== S6: Threshold sensitivity + tripartite classification ===")
    tripartite = {}
    for thr in [0.60, 0.65, 0.70]:
        counts = {"recoverable": 0, "inconclusive": 0, "not_recoverable": 0}
        for cell, r in ok.items():
            if r["tier"] != "primary":
                continue
            pm = perm_map.get(cell, {})
            cat = classify_tripartite(r["pooled_auroc"], r["auroc_ci"][0],
                                      bool(pm.get("bh_significant")), rec_thr=thr)
            counts[cat] += 1
            if thr == 0.65:
                tripartite[cell] = cat
        out[f"threshold_{thr}"] = counts
        logger.info(f"  thr={thr}: {counts}")
    # category change between 0.60 and 0.70
    changed = 0
    for cell, r in ok.items():
        if r["tier"] != "primary":
            continue
        pm = perm_map.get(cell, {})
        c60 = classify_tripartite(r["pooled_auroc"], r["auroc_ci"][0], bool(pm.get("bh_significant")), 0.60)
        c70 = classify_tripartite(r["pooled_auroc"], r["auroc_ci"][0], bool(pm.get("bh_significant")), 0.70)
        if c60 != c70:
            changed += 1
    n_primary = sum(1 for r in ok.values() if r["tier"] == "primary")
    out["threshold_change_0.60_to_0.70"] = {"changed": changed, "total": n_primary,
                                            "fraction": round(changed / n_primary, 3),
                                            "threshold_sensitive": changed / n_primary >= 0.30}
    out["tripartite_classification"] = tripartite
    logger.info(f"  Cells changing category 0.60→0.70: {changed}/{n_primary} ({changed/n_primary:.0%})")

    # ---------- S5: BY-FDR vs BH-FDR ----------
    logger.info("\n=== S5: Benjamini-Yekutieli FDR (vs BH) ===")
    primary_perm = [(c, perm_map[c]) for c in ok if ok[c]["tier"] == "primary" and c in perm_map]
    pvals = [pm["p_value"] for _, pm in primary_perm]
    by_sig = by_fdr(pvals)
    bh_sig = [bool(pm.get("bh_significant")) for _, pm in primary_perm]
    by_results = []
    for (cell, pm), bys, bhs in zip(primary_perm, by_sig, bh_sig):
        by_results.append({"cell": cell, "p_value": pm["p_value"], "bh_sig": bhs, "by_sig": bys})
    n_bh = sum(bh_sig); n_by = sum(by_sig)
    out["S5_by_fdr"] = {"n_bh_significant": n_bh, "n_by_significant": n_by, "cells": by_results}
    logger.info(f"  BH-significant: {n_bh}, BY-significant: {n_by}")
    for r in by_results:
        if r["bh_sig"] or r["by_sig"]:
            logger.info(f"    {r['cell']:28s} p={r['p_value']:.5f} BH={r['bh_sig']} BY={r['by_sig']}")

    # ---------- S1: PCA, S3: LOOCV, S9: continuous ----------
    logger.info("\n=== S1 (PCA), S3 (LOOCV), S9 (continuous regression) ===")
    s1, s3, s9 = [], [], []
    q75_cols = {"label_tmb_high": "tmb", "label_fga_high": "fga", "label_aneuploidy_high": "aneuploidy"}

    for cohort in COHORTS:
        try:
            features, slide_ids, labels, splits = load_cohort_data(cohort, project_root)
        except Exception:
            continue
        for cell in BIOMARKER_CELLS.get(cohort, []):
            cell_name = f"{cohort}/{cell['name']}"
            if cell_name not in ok:
                continue
            X, y, folds = _patient_matrix(cohort, cell, features, slide_ids, labels, splits)
            if len(y) < 10 or len(np.unique(y)) < 2:
                continue
            base = ok[cell_name]["pooled_auroc"]

            # S1 PCA
            pca_auroc = _cv_auroc(X, y, folds, pca=True)
            s1.append({"cell": cell_name, "base_auroc": base,
                       "pca_auroc": round(pca_auroc, 4) if not np.isnan(pca_auroc) else None,
                       "delta": round(pca_auroc - base, 4) if not np.isnan(pca_auroc) else None})

            # S3 LOOCV (only 3 smallest cohorts)
            if cohort in ("CHOL", "DLBC", "KICH"):
                loo_auroc = _cv_auroc(X, y, folds, loocv=True)
                s3.append({"cell": cell_name, "base_auroc": base,
                           "loocv_auroc": round(loo_auroc, 4) if not np.isnan(loo_auroc) else None,
                           "delta": round(loo_auroc - base, 4) if not np.isnan(loo_auroc) else None})

            # S9 continuous regression (Spearman) for Q75 cells
            col = cell.get("col")
            if col in q75_cols:
                cont_col = q75_cols[col]
                # patient-level continuous values
                slide_patient = {sid: parse_patient_id(sid) for sid in slide_ids}
                pf = {}
                for i, sid in enumerate(slide_ids):
                    pf.setdefault(slide_patient[sid], []).append(features[i])
                pmean = {pid: np.mean(f, axis=0) for pid, f in pf.items()}
                cont_map = dict(zip(labels["patient_id"], labels[cont_col]))
                smap = dict(zip(splits["patient_id"], splits["fold"]))
                Xc, yc, foldc = [], [], []
                for pid, feat in pmean.items():
                    if pid in cont_map and pid in smap and pd.notna(cont_map[pid]):
                        Xc.append(feat); yc.append(float(cont_map[pid])); foldc.append(smap[pid])
                Xc, yc, foldc = np.array(Xc), np.array(yc), np.array(foldc)
                # CV ridge regression → pooled OOF predictions → Spearman
                from sklearn.linear_model import RidgeCV
                preds = np.full(len(yc), np.nan)
                for fi in range(5):
                    te = foldc == fi; tr = ~te
                    if te.sum() == 0 or tr.sum() < 5:
                        continue
                    sc = StandardScaler(); Xtr = sc.fit_transform(Xc[tr]); Xte = sc.transform(Xc[te])
                    rr = RidgeCV(alphas=[0.1, 1, 10, 100, 1000])
                    rr.fit(Xtr, yc[tr])
                    preds[te] = rr.predict(Xte)
                valid = ~np.isnan(preds)
                if valid.sum() > 10:
                    rho, pval = spearmanr(yc[valid], preds[valid])
                    s9.append({"cell": cell_name, "continuous_var": cont_col,
                               "spearman_r": round(rho, 4), "spearman_p": round(pval, 5),
                               "binary_auroc": base})

    out["S1_pca"] = s1
    out["S3_loocv"] = s3
    out["S9_continuous"] = s9

    s1_improved = sum(1 for r in s1 if r["delta"] and r["delta"] > 0.02)
    logger.info(f"  S1 PCA: {s1_improved}/{len(s1)} cells improved >0.02 (promotion needs ≥75%)")
    logger.info(f"  S3 LOOCV: {len(s3)} cells (3 smallest cohorts)")
    for r in s3:
        logger.info(f"    {r['cell']:24s} 5fold={r['base_auroc']:.3f} LOOCV={r['loocv_auroc']} delta={r['delta']}")
    logger.info(f"  S9 continuous: {len(s9)} Q75 cells")
    for r in s9:
        logger.info(f"    {r['cell']:24s} Spearman r={r['spearman_r']} (p={r['spearman_p']}) vs binary AUROC {r['binary_auroc']:.3f}")

    # ---------- GTF2I-WHO confound test (§4, §10.5) ----------
    logger.info("\n=== GTF2I-WHO confound test (Cramér's V) ===")
    thym_labels = pd.read_parquet(project_root / "data" / "labels" / "thym_labels.parquet")
    sub = thym_labels.dropna(subset=["mut_GTF2I", "label_who_subtype"])
    v = cramers_v(sub["mut_GTF2I"], sub["label_who_subtype"])
    out["gtf2i_who_confound"] = {"cramers_v": round(v, 4), "n": len(sub),
                                 "interpretation": "confound test triggered (V>0.5)" if v > 0.5 else "low association"}
    logger.info(f"  Cramér's V(GTF2I, WHO subtype) = {v:.3f} (n={len(sub)})")
    if v > 0.5:
        logger.info("  V>0.5 → within-stratum residualised analysis required (see §10.5)")
        # Stratified: A+AB vs rest
        feats_thym = np.load(project_root / "data" / "embeddings" / "THYM" / "mean_pooled_features.npy")
        man = json.loads((project_root / "data" / "embeddings" / "THYM" / "mean_pool_manifest.json").read_text())
        sids = man["slide_ids"]
        cell = next(c for c in BIOMARKER_CELLS["THYM"] if c["name"] == "GTF2I")
        splits = pd.read_csv(project_root / "data" / "splits" / "thym_splits.csv")
        # within A+AB stratum
        sub_aab = sub[sub["label_who_subtype"].isin(["A", "AB"])]["patient_id"].tolist()
        sub_other = sub[~sub["label_who_subtype"].isin(["A", "AB"])]["patient_id"].tolist()
        X, y, folds = _patient_matrix("THYM", cell, feats_thym, sids, thym_labels, splits)
        # map patient ids
        slide_patient = {sid: parse_patient_id(sid) for sid in sids}
        pf = {}
        for i, sid in enumerate(sids):
            pf.setdefault(slide_patient[sid], []).append(feats_thym[i])
        pmean = {pid: np.mean(f, axis=0) for pid, f in pf.items()}
        lmap = dict(zip(thym_labels["patient_id"], thym_labels["mut_GTF2I"]))
        smap = dict(zip(splits["patient_id"], splits["fold"]))
        for stratum_name, pid_list in [("A+AB", sub_aab), ("B1-B3+TC", sub_other)]:
            Xs, ys, fs = [], [], []
            for pid in pid_list:
                if pid in pmean and pid in lmap and pid in smap and pd.notna(lmap[pid]):
                    Xs.append(pmean[pid]); ys.append(float(lmap[pid])); fs.append(smap[pid])
            Xs, ys, fs = np.array(Xs), np.array(ys), np.array(fs)
            if len(ys) >= 10 and len(np.unique(ys)) >= 2:
                a = _cv_auroc(Xs, ys, fs)
                out["gtf2i_who_confound"][f"within_{stratum_name}_auroc"] = round(a, 4) if not np.isnan(a) else None
                out["gtf2i_who_confound"][f"within_{stratum_name}_n"] = len(ys)
                out["gtf2i_who_confound"][f"within_{stratum_name}_prevalence"] = round(float(ys.mean()), 3)
                logger.info(f"    Within {stratum_name} (n={len(ys)}, prev={ys.mean():.2f}): AUROC={a:.3f}")

    # Save
    out_path = project_root / "results" / "sensitivity_analyses.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info(f"\nSaved: {out_path}")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_all()
