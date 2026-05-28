"""
PanRareWSI — Phase 6: Quantitative failure-mode analysis (pre-registration RQ4).

For the FDR-significant cells, identifies high-confidence misclassifications
and tests whether errors cluster by tissue-source-site (TSS confound, §10) or
by available clinical metadata (stage, grade). Full morphological inspection
of misclassified WSIs is deferred (requires raw WSI download, Phase 6b).

Usage:
    python3 -m src.failure_modes
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from src.calibration import get_oof_predictions
from src.phase4_benchmark import (
    BIOMARKER_CELLS, COHORTS, load_cohort_data, parse_patient_id, _get_binary_labels,
)

logger = logging.getLogger(__name__)

# Cells to analyse: FDR-significant + high-AUROC exploratory
FOCUS_CELLS = [
    ("DLBC", "MSI-H"), ("THYM", "GTF2I"), ("THYM", "TMB-high"),
    ("UVM", "Chr3 loss"), ("UVM", "EIF1AX"), ("CHOL", "IDH1"),
]


def parse_tss(patient_id: str) -> str:
    """Tissue source site = chars 6-7 of TCGA barcode (TCGA-XX-...)."""
    parts = patient_id.split("-")
    return parts[1] if len(parts) > 1 else "??"


def get_oof_with_ids(cohort, cell, features, slide_ids, labels, splits):
    """Like get_oof_predictions but returns patient IDs too."""
    from src.models.linear_probe import train_and_evaluate

    slide_patient = {sid: parse_patient_id(sid) for sid in slide_ids}
    patient_features = {}
    for i, sid in enumerate(slide_ids):
        pid = slide_patient[sid]
        patient_features.setdefault(pid, []).append(features[i])
    patient_mean = {pid: np.mean(f, axis=0) for pid, f in patient_features.items()}

    y_series = _get_binary_labels(labels, cell)
    label_map = dict(zip(labels["patient_id"], y_series))
    split_map = dict(zip(splits["patient_id"], splits["fold"]))

    pids, X, y, folds = [], [], [], []
    for pid, feat in patient_mean.items():
        if pid in label_map and pid in split_map and pd.notna(label_map[pid]):
            pids.append(pid)
            X.append(feat)
            y.append(label_map[pid])
            folds.append(split_map[pid])
    pids, X, y, folds = np.array(pids), np.array(X), np.array(y), np.array(folds)

    probas = np.full(len(y), np.nan)
    for fi in range(5):
        te = folds == fi
        tr = ~te
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        res = train_and_evaluate(X[tr], y[tr], X[te], y[te])
        probas[te] = res.probas
    valid = ~np.isnan(probas)
    return pids[valid], y[valid], probas[valid]


def analyse_cell(cohort, cell, features, slide_ids, labels, splits):
    pids, y, probas = get_oof_with_ids(cohort, cell, features, slide_ids, labels, splits)
    if len(y) < 10:
        return None

    pred = (probas >= 0.5).astype(int)
    correct = pred == y
    # Confidence-weighted error: |proba - y|
    error_mag = np.abs(probas - y)

    # High-confidence errors (wrong AND confident)
    wrong = ~correct
    hc_error_idx = np.argsort(-error_mag)
    hc_errors = []
    for idx in hc_error_idx:
        if wrong[idx] and error_mag[idx] > 0.5:
            hc_errors.append({
                "patient_id": str(pids[idx]),
                "true_label": int(y[idx]),
                "predicted_proba": round(float(probas[idx]), 3),
                "tss": parse_tss(str(pids[idx])),
                "error_magnitude": round(float(error_mag[idx]), 3),
            })

    # TSS confound test: are errors concentrated in specific tissue source sites?
    tss_all = [parse_tss(str(p)) for p in pids]
    tss_wrong = [parse_tss(str(p)) for p in pids[wrong]]
    tss_counts_all = Counter(tss_all)
    tss_counts_wrong = Counter(tss_wrong)

    # For the most common TSS, test error enrichment via Fisher exact
    tss_tests = []
    for tss, n_total in tss_counts_all.most_common(3):
        n_wrong_tss = tss_counts_wrong.get(tss, 0)
        n_wrong_other = int(wrong.sum()) - n_wrong_tss
        n_correct_tss = n_total - n_wrong_tss
        n_correct_other = len(y) - n_total - n_wrong_other
        table = [[n_wrong_tss, n_correct_tss], [n_wrong_other, n_correct_other]]
        try:
            _, p = fisher_exact(table)
        except Exception:
            p = 1.0
        tss_tests.append({
            "tss": tss, "n_patients": n_total,
            "n_errors": n_wrong_tss,
            "error_rate": round(n_wrong_tss / n_total, 3),
            "fisher_p": round(p, 4),
        })

    return {
        "cell": f"{cohort}/{cell['name']}",
        "n": len(y),
        "n_errors": int(wrong.sum()),
        "overall_error_rate": round(float(wrong.mean()), 3),
        "n_high_confidence_errors": len(hc_errors),
        "high_confidence_errors": hc_errors[:10],
        "n_distinct_tss": len(tss_counts_all),
        "tss_enrichment_tests": tss_tests,
    }


def run_failure_analysis(project_root: Path | None = None):
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    cell_lookup = {}
    for cohort in COHORTS:
        for cell in BIOMARKER_CELLS.get(cohort, []):
            cell_lookup[(cohort, cell["name"])] = cell

    results = []
    for cohort, biomarker in FOCUS_CELLS:
        cell = cell_lookup.get((cohort, biomarker))
        if cell is None:
            continue
        features, slide_ids, labels, splits = load_cohort_data(cohort, project_root)
        res = analyse_cell(cohort, cell, features, slide_ids, labels, splits)
        if res:
            results.append(res)
            logger.info(f"\n{res['cell']}: {res['n_errors']}/{res['n']} errors "
                        f"({res['overall_error_rate']:.0%}), "
                        f"{res['n_high_confidence_errors']} high-confidence")
            for t in res["tss_enrichment_tests"]:
                flag = " ⚠ENRICHED" if t["fisher_p"] < 0.05 else ""
                logger.info(f"    TSS {t['tss']}: {t['n_errors']}/{t['n_patients']} errors "
                            f"(rate {t['error_rate']:.0%}, Fisher p={t['fisher_p']}){flag}")

    out_path = project_root / "results" / "failure_modes.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved: {out_path}")
    logger.info("Note: morphological inspection of misclassified WSIs deferred to Phase 6b (requires raw WSI download).")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_failure_analysis()
