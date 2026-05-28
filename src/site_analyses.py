"""
PanRareWSI — S7/S8 tissue-source-site sensitivity analyses (§7.7).

Trigger (per §7.7): biomarker-label vs TSS chi-squared p < 0.05.
For triggered cells:
  S7  Site-aware CV (no test fold dominated by a single site) — Δ>0.05 flags confound
  S8  Site-as-covariate linear probe — AUROC increase suggests site reliance

Usage:
    python3 -m src.site_analyses
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.phase4_benchmark import (
    BIOMARKER_CELLS, COHORTS, load_cohort_data, parse_patient_id, _get_binary_labels,
)

logger = logging.getLogger(__name__)


def parse_tss(pid):
    parts = pid.split("-")
    return parts[1] if len(parts) > 1 else "??"


def build(cohort, cell, features, slide_ids, labels, splits):
    slide_patient = {sid: parse_patient_id(sid) for sid in slide_ids}
    pf = {}
    for i, sid in enumerate(slide_ids):
        pf.setdefault(slide_patient[sid], []).append(features[i])
    pmean = {pid: np.mean(f, axis=0) for pid, f in pf.items()}
    y_series = _get_binary_labels(labels, cell)
    lmap = dict(zip(labels["patient_id"], y_series))
    smap = dict(zip(splits["patient_id"], splits["fold"]))
    pids, X, y, folds = [], [], [], []
    for pid, feat in pmean.items():
        if pid in lmap and pid in smap and pd.notna(lmap[pid]):
            pids.append(pid); X.append(feat); y.append(lmap[pid]); folds.append(smap[pid])
    return np.array(pids), np.array(X), np.array(y), np.array(folds)


def cv_auroc(X, y, fold_assign):
    probas = np.full(len(y), np.nan)
    for fi in np.unique(fold_assign):
        te = fold_assign == fi; tr = ~te
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


def site_aware_folds(pids, y, seed=42):
    """Assign folds stratified by label, grouping so sites are spread across folds."""
    tss = np.array([parse_tss(p) for p in pids])
    # Stratify by label, but shuffle within to distribute sites
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = np.full(len(y), -1)
    for fi, (_, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds[te] = fi
    return folds


def run(project_root: Path | None = None):
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    phase4 = json.loads((project_root / "results" / "phase4_benchmark.json").read_text())
    ok = {r["cell"]: r for r in phase4 if r["status"] == "ok"}

    results = []
    for cohort in COHORTS:
        try:
            features, slide_ids, labels, splits = load_cohort_data(cohort, project_root)
        except Exception:
            continue
        for cell in BIOMARKER_CELLS.get(cohort, []):
            cell_name = f"{cohort}/{cell['name']}"
            if cell_name not in ok:
                continue
            pids, X, y, folds = build(cohort, cell, features, slide_ids, labels, splits)
            if len(y) < 10 or len(np.unique(y)) < 2:
                continue

            # Trigger: label vs TSS chi-squared
            tss = np.array([parse_tss(p) for p in pids])
            # collapse rare sites for valid chi-square
            counts = Counter(tss)
            tss_collapsed = np.array([t if counts[t] >= 5 else "OTHER" for t in tss])
            ct = pd.crosstab(tss_collapsed, y)
            triggered = False
            chi_p = None
            if ct.shape[0] >= 2 and ct.shape[1] >= 2:
                try:
                    chi_p = chi2_contingency(ct)[1]
                    triggered = chi_p < 0.05
                except Exception:
                    pass
            if not triggered:
                continue

            base = ok[cell_name]["pooled_auroc"]

            # S7: site-aware CV
            sa_folds = site_aware_folds(pids, y)
            s7_auroc = cv_auroc(X, y, sa_folds)

            # S8: site-as-covariate
            uniq_sites = sorted(set(tss_collapsed))
            site_onehot = np.zeros((len(y), len(uniq_sites)))
            for i, t in enumerate(tss_collapsed):
                site_onehot[i, uniq_sites.index(t)] = 1
            X_aug = np.hstack([X, site_onehot])
            s8_auroc = cv_auroc(X_aug, y, folds)

            res = {
                "cell": cell_name,
                "base_auroc": base,
                "chi2_p": round(chi_p, 5),
                "s7_site_aware_auroc": round(s7_auroc, 4) if not np.isnan(s7_auroc) else None,
                "s7_delta": round(s7_auroc - base, 4) if not np.isnan(s7_auroc) else None,
                "s8_site_covariate_auroc": round(s8_auroc, 4) if not np.isnan(s8_auroc) else None,
                "s8_delta": round(s8_auroc - base, 4) if not np.isnan(s8_auroc) else None,
                "s7_confound_flag": bool(not np.isnan(s7_auroc) and abs(s7_auroc - base) > 0.05),
            }
            results.append(res)
            logger.info(f"{cell_name}: chi2_p={chi_p:.4f} base={base:.3f} "
                        f"S7(site-aware)={res['s7_site_aware_auroc']} (Δ{res['s7_delta']}) "
                        f"S8(site-cov)={res['s8_site_covariate_auroc']} (Δ{res['s8_delta']})")

    out_path = project_root / "results" / "site_analyses.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n{len(results)} cells triggered TSS analysis. Saved: {out_path}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
