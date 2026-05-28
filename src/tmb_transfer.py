"""
PanRareWSI — TMB-high cross-cohort transfer (RQ3, completing the pre-registered
transfer experiment alongside TP53).

Trains a TMB-high classifier on TCGA-COAD (common cohort; TMB binarised at COAD
Q75) and applies it zero-shot to the rare cohorts that have a TMB-high cell
(ACC, THYM, DLBC). Compares transfer AUROC to within-cohort AUROC.

Requires data/embeddings/COAD/mean_pooled_features.npy (from modal_coad_meanpool.py).

Usage:
    python3 -m src.tmb_transfer
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.phase4_benchmark import load_cohort_data, parse_patient_id

logger = logging.getLogger(__name__)

CBIO = "https://www.cbioportal.org/api"
TARGETS = ["ACC", "THYM", "DLBC"]  # rare cohorts with a TMB-high cell


def cbio_tmb_coad() -> dict[str, float]:
    """Pull TMB_NONSYNONYMOUS for TCGA-COAD PanCancer Atlas from cBioPortal."""
    study = "coadread_tcga_pan_can_atlas_2018"  # COAD+READ merged in PanCancer Atlas
    out = {}
    for dtype in ("SAMPLE", "PATIENT"):
        url = f"{CBIO}/studies/{study}/clinical-data?clinicalDataType={dtype}&attributeId=TMB_NONSYNONYMOUS"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
            for d in data:
                v = d.get("value")
                if v and v not in ("NA", "NaN"):
                    try:
                        out[d["patientId"]] = float(v)
                    except ValueError:
                        pass
            if out:
                break
        except Exception as e:
            logger.warning(f"cBioPortal {dtype} fetch failed: {e}")
    return out


def patient_features(features, slide_ids):
    pf = {}
    for i, sid in enumerate(slide_ids):
        pf.setdefault(parse_patient_id(sid), []).append(features[i])
    return {pid: np.mean(v, axis=0) for pid, v in pf.items()}


def run(project_root: Path | None = None):
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    # COAD source
    coad_dir = project_root / "data" / "embeddings" / "COAD"
    coad_feats = np.load(coad_dir / "mean_pooled_features.npy")
    coad_sids = json.loads((coad_dir / "mean_pool_manifest.json").read_text())["slide_ids"]
    coad_pf = patient_features(coad_feats, coad_sids)

    tmb = cbio_tmb_coad()
    logger.info(f"COAD TMB values: {len(tmb)} patients")
    if not tmb:
        logger.error("No COAD TMB labels — aborting")
        return

    q75 = np.quantile(list(tmb.values()), 0.75)
    logger.info(f"COAD TMB Q75 threshold: {q75:.2f}")

    Xc, yc = [], []
    for pid, feat in coad_pf.items():
        if pid in tmb:
            Xc.append(feat); yc.append(1.0 if tmb[pid] >= q75 else 0.0)
    Xc, yc = np.array(Xc), np.array(yc)
    logger.info(f"COAD training set: n={len(yc)}, TMB-high prevalence={yc.mean():.2f}")

    scaler = StandardScaler()
    Xc_s = scaler.fit_transform(Xc)
    clf = LogisticRegressionCV(Cs=10, cv=5, penalty="l2", scoring="roc_auc",
                               solver="lbfgs", max_iter=5000, random_state=42,
                               class_weight="balanced")
    clf.fit(Xc_s, yc)

    # Within-cohort AUROCs from phase4 for comparison
    p4 = {r["cell"]: r for r in json.loads((project_root / "results" / "phase4_benchmark.json").read_text()) if r["status"] == "ok"}

    results = []
    for cohort in TARGETS:
        features, slide_ids, labels, splits = load_cohort_data(cohort, project_root)
        pf = patient_features(features, slide_ids)
        lmap = dict(zip(labels["patient_id"], labels["label_tmb_high"]))
        X, y = [], []
        for pid, feat in pf.items():
            if pid in lmap and pd.notna(lmap[pid]):
                X.append(feat); y.append(float(lmap[pid]))
        X, y = np.array(X), np.array(y)
        if len(np.unique(y)) < 2:
            continue
        probas = clf.predict_proba(scaler.transform(X))[:, 1]
        transfer_auroc = roc_auc_score(y, probas)
        within = p4.get(f"{cohort}/TMB-high", {}).get("pooled_auroc")
        results.append({
            "target_cohort": cohort, "biomarker": "TMB-high",
            "transfer_auroc": round(transfer_auroc, 4),
            "within_auroc": within,
            "delta": round(transfer_auroc - within, 4) if within else None,
            "n": len(y), "prevalence": round(float(y.mean()), 3),
        })
        logger.info(f"  {cohort} TMB-high: transfer={transfer_auroc:.3f} within={within} "
                    f"delta={results[-1]['delta']} (n={len(y)})")

    out = {"source": "TCGA-COAD TMB-high", "source_n": len(yc),
           "source_q75": round(float(q75), 3), "source_prevalence": round(float(yc.mean()), 3),
           "transfer_results": results}
    out_path = project_root / "results" / "tmb_transfer.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved: {out_path}")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
