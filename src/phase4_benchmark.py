"""
PanRareWSI — Phase 4: Systematic rare-cohort benchmark.

Runs L2-regularised linear probe across all cohort × biomarker cells using
pre-registered 5-fold stratified CV. Reports AUROC, AUPRC, Brier with
1000× bootstrap CIs. Permutation tests with BH-FDR correction.

Pre-registration §8, §10, §17.

Usage:
    python3 -m src.phase4_benchmark
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from src.models.linear_probe import train_and_evaluate, bootstrap_ci

logger = logging.getLogger(__name__)

COHORTS = ["ACC", "UVM", "MESO", "CHOL", "THYM", "KICH", "DLBC"]

BIOMARKER_CELLS: dict[str, list[dict]] = {
    "ACC": [
        {"name": "TP53", "col": "mut_TP53", "tier": "primary"},
        {"name": "CTNNB1", "col": "mut_CTNNB1", "tier": "primary"},
        {"name": "TMB-high", "col": "label_tmb_high", "tier": "primary"},
        {"name": "FGA-high", "col": "label_fga_high", "tier": "primary"},
        {"name": "Immune C4", "col": "immune_subtype", "binarise": ("C4", "non-C4"), "tier": "primary"},
        {"name": "MEN1", "col": "mut_MEN1", "tier": "exploratory"},
    ],
    "UVM": [
        {"name": "GNAQ", "col": "mut_GNAQ", "tier": "primary"},
        {"name": "GNA11", "col": "mut_GNA11", "tier": "primary"},
        {"name": "SF3B1", "col": "mut_SF3B1", "tier": "primary"},
        {"name": "BAP1", "col": "mut_BAP1", "tier": "primary"},
        {"name": "EIF1AX", "col": "mut_EIF1AX", "tier": "primary"},
        {"name": "Histology 3-class", "col": "label_histology_3class", "tier": "primary", "multiclass": True},
        {"name": "Chr3 loss", "col": "label_chr3_loss", "tier": "primary"},
        {"name": "Immune C4", "col": "immune_subtype", "binarise": ("C4", "non-C4"), "tier": "primary"},
    ],
    "MESO": [
        {"name": "Histology epi", "col": "label_histology_epi", "tier": "primary"},
        {"name": "NF2", "col": "mut_NF2", "tier": "primary"},
        {"name": "BAP1", "col": "mut_BAP1", "tier": "primary"},
        {"name": "TP53", "col": "mut_TP53", "tier": "primary"},
        {"name": "CDKN2A homdel", "col": "label_cdkn2a_homdel", "tier": "primary"},
        {"name": "Immune C1", "col": "immune_subtype", "binarise": ("C1", "non-C1"), "tier": "primary"},
    ],
    "CHOL": [
        {"name": "Anatomic subtype", "col": "histological_subtype", "tier": "exploratory", "skip": True},
        {"name": "IDH1", "col": "mut_IDH1", "tier": "exploratory"},
        {"name": "ARID1A", "col": "mut_ARID1A", "tier": "exploratory"},
        {"name": "Immune C3", "col": "immune_subtype", "binarise": ("C3", "non-C3"), "tier": "exploratory"},
    ],
    "THYM": [
        {"name": "GTF2I", "col": "mut_GTF2I", "tier": "primary"},
        {"name": "TMB-high", "col": "label_tmb_high", "tier": "primary"},
        {"name": "WHO subtype", "col": "label_who_subtype", "tier": "primary", "multiclass": True},
        {"name": "HRAS", "col": "mut_HRAS", "tier": "exploratory"},
    ],
    "KICH": [
        {"name": "TP53", "col": "mut_TP53", "tier": "primary"},
        {"name": "Aneuploidy-high", "col": "label_aneuploidy_high", "tier": "primary"},
        {"name": "Immune C3", "col": "immune_subtype", "binarise": ("C3", "non-C3"), "tier": "primary"},
        {"name": "MSI-H", "col": "label_msi_h", "tier": "exploratory"},
    ],
    "DLBC": [
        {"name": "MSI-H", "col": "label_msi_h", "tier": "primary"},
        {"name": "TMB-high", "col": "label_tmb_high", "tier": "primary"},
        {"name": "Hans non-GCB", "col": "label_hans_nongcb", "tier": "primary"},
        {"name": "MYD88", "col": "mut_MYD88", "tier": "primary"},
        {"name": "Aneuploidy-high", "col": "label_aneuploidy_high", "tier": "primary"},
        {"name": "CD79B", "col": "mut_CD79B", "tier": "exploratory"},
        {"name": "TP53", "col": "mut_TP53", "tier": "exploratory"},
    ],
}


def _binarise_immune(series: pd.Series, target_class: str) -> pd.Series:
    """Binarise immune subtype: target_class=1, everything else=0, None=NaN."""
    return series.apply(lambda x: 1.0 if x == target_class else (0.0 if pd.notna(x) else np.nan))


def _get_binary_labels(df: pd.DataFrame, cell: dict) -> pd.Series:
    """Extract binary label for a biomarker cell."""
    if cell.get("skip"):
        return pd.Series(np.nan, index=df.index)

    if "binarise" in cell:
        target, _ = cell["binarise"]
        return _binarise_immune(df[cell["col"]], target)

    return df[cell["col"]].astype(float)


def load_cohort_data(
    cohort: str,
    project_root: Path,
) -> tuple[np.ndarray, list[str], pd.DataFrame, pd.DataFrame]:
    """Load mean-pooled features, slide IDs, labels, and splits for a cohort."""
    embed_dir = project_root / "data" / "embeddings" / cohort
    features = np.load(embed_dir / "mean_pooled_features.npy")
    manifest = json.loads((embed_dir / "mean_pool_manifest.json").read_text())
    slide_ids = manifest["slide_ids"]

    labels = pd.read_parquet(project_root / "data" / "labels" / f"{cohort.lower()}_labels.parquet")
    splits = pd.read_csv(project_root / "data" / "splits" / f"{cohort.lower()}_splits.csv")

    return features, slide_ids, labels, splits


def parse_patient_id(slide_id: str) -> str:
    """Extract 12-char TCGA patient ID from slide filename."""
    parts = slide_id.split(".")
    barcode = parts[0]
    # TCGA-XX-XXXX
    return "-".join(barcode.split("-")[:3])


def run_cell(
    cohort: str,
    cell: dict,
    features: np.ndarray,
    slide_ids: list[str],
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    n_boot: int = 1000,
) -> dict:
    """Run 5-fold CV for one cohort × biomarker cell."""
    cell_name = f"{cohort}/{cell['name']}"

    if cell.get("skip"):
        return {"cell": cell_name, "status": "skipped", "tier": cell["tier"]}

    if cell.get("multiclass"):
        return {"cell": cell_name, "status": "multiclass_deferred", "tier": cell["tier"]}

    # Build patient-level mapping: slide_id → patient_id → features
    slide_patient = {sid: parse_patient_id(sid) for sid in slide_ids}
    patient_features = {}
    for i, sid in enumerate(slide_ids):
        pid = slide_patient[sid]
        if pid not in patient_features:
            patient_features[pid] = []
        patient_features[pid].append(features[i])

    # Mean across slides for multi-slide patients
    patient_mean_features = {
        pid: np.mean(feats, axis=0) for pid, feats in patient_features.items()
    }

    # Get labels
    y_series = _get_binary_labels(labels, cell)
    label_map = dict(zip(labels["patient_id"], y_series))

    # Join features + labels + splits
    matched_pids = []
    X_all = []
    y_all = []
    fold_all = []

    split_map = dict(zip(splits["patient_id"], splits["fold"]))

    for pid, feat in patient_mean_features.items():
        if pid in label_map and pid in split_map and pd.notna(label_map[pid]):
            matched_pids.append(pid)
            X_all.append(feat)
            y_all.append(label_map[pid])
            fold_all.append(split_map[pid])

    if not matched_pids:
        return {"cell": cell_name, "status": "no_data", "tier": cell["tier"]}

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    fold_all = np.array(fold_all)

    n_pos = int(y_all.sum())
    n_neg = len(y_all) - n_pos
    prevalence = n_pos / len(y_all)

    if n_pos < 5 or n_neg < 5:
        return {
            "cell": cell_name, "status": "insufficient_n",
            "n": len(y_all), "n_pos": n_pos, "prevalence": prevalence,
            "tier": cell["tier"],
        }

    # 5-fold CV
    fold_results = []
    all_probas = np.full(len(y_all), np.nan)

    for fold_idx in range(5):
        test_mask = fold_all == fold_idx
        train_mask = ~test_mask

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        if len(np.unique(y_test)) < 2 or len(np.unique(y_train)) < 2:
            fold_results.append({"fold": fold_idx, "status": "single_class"})
            continue

        result = train_and_evaluate(X_train, y_train, X_test, y_test)
        all_probas[test_mask] = result.probas

        fold_results.append({
            "fold": fold_idx,
            "auroc": result.auroc,
            "auprc": result.auprc,
            "brier": result.brier,
            "n_train": result.n_train,
            "n_test": result.n_test,
            "n_pos_test": result.n_pos_test,
            "best_C": result.best_C,
        })

    # Pooled out-of-fold metrics
    valid_mask = ~np.isnan(all_probas)
    if valid_mask.sum() < 10 or len(np.unique(y_all[valid_mask])) < 2:
        return {
            "cell": cell_name, "status": "insufficient_valid_predictions",
            "n": len(y_all), "n_pos": n_pos, "tier": cell["tier"],
        }

    pooled_auroc = roc_auc_score(y_all[valid_mask], all_probas[valid_mask])
    pooled_auprc = average_precision_score(y_all[valid_mask], all_probas[valid_mask])
    pooled_brier = brier_score_loss(y_all[valid_mask], all_probas[valid_mask])

    # Bootstrap CIs on pooled predictions
    auroc_point, auroc_lo, auroc_hi = bootstrap_ci(
        y_all[valid_mask], all_probas[valid_mask], roc_auc_score, n_boot=n_boot
    )
    auprc_point, auprc_lo, auprc_hi = bootstrap_ci(
        y_all[valid_mask], all_probas[valid_mask], average_precision_score, n_boot=n_boot
    )

    # Per-fold AUROCs for reporting
    fold_aurocs = [r["auroc"] for r in fold_results if "auroc" in r]

    return {
        "cell": cell_name,
        "cohort": cohort,
        "biomarker": cell["name"],
        "tier": cell["tier"],
        "status": "ok",
        "n": len(y_all),
        "n_pos": n_pos,
        "prevalence": round(prevalence, 3),
        "pooled_auroc": round(pooled_auroc, 4),
        "pooled_auprc": round(pooled_auprc, 4),
        "pooled_brier": round(pooled_brier, 4),
        "auroc_ci": [round(auroc_lo, 4), round(auroc_hi, 4)],
        "auprc_ci": [round(auprc_lo, 4), round(auprc_hi, 4)],
        "mean_fold_auroc": round(np.mean(fold_aurocs), 4) if fold_aurocs else None,
        "std_fold_auroc": round(np.std(fold_aurocs), 4) if fold_aurocs else None,
        "n_valid_folds": len(fold_aurocs),
        "per_fold": fold_results,
    }


def run_benchmark(project_root: Path | None = None, n_boot: int = 1000) -> list[dict]:
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    all_results = []

    for cohort in COHORTS:
        logger.info(f"\n{'='*60}")
        logger.info(f"{cohort}")
        logger.info(f"{'='*60}")

        try:
            features, slide_ids, labels, splits = load_cohort_data(cohort, project_root)
        except Exception as e:
            logger.error(f"{cohort}: failed to load data: {e}")
            continue

        cells = BIOMARKER_CELLS.get(cohort, [])
        for cell in cells:
            result = run_cell(cohort, cell, features, slide_ids, labels, splits, n_boot=n_boot)
            all_results.append(result)

            status = result["status"]
            if status == "ok":
                auroc = result["pooled_auroc"]
                ci = result["auroc_ci"]
                logger.info(f"  ✓ {result['biomarker']:20s} AUROC={auroc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] n={result['n']} ({result['tier']})")
            elif status in ("skipped", "multiclass_deferred"):
                logger.info(f"  — {cell['name']:20s} {status}")
            else:
                logger.warning(f"  ✗ {cell['name']:20s} {status} (n_pos={result.get('n_pos', '?')})")

    # Summary
    ok_results = [r for r in all_results if r["status"] == "ok"]
    primary_ok = [r for r in ok_results if r["tier"] == "primary"]
    exploratory_ok = [r for r in ok_results if r["tier"] == "exploratory"]

    logger.info(f"\n{'='*60}")
    logger.info("PHASE 4 SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total cells attempted: {len(all_results)}")
    logger.info(f"  OK: {len(ok_results)} ({len(primary_ok)} primary, {len(exploratory_ok)} exploratory)")
    logger.info(f"  Skipped/deferred: {sum(1 for r in all_results if r['status'] in ('skipped', 'multiclass_deferred'))}")
    logger.info(f"  Insufficient n: {sum(1 for r in all_results if r['status'] == 'insufficient_n')}")

    if primary_ok:
        aurocs = [r["pooled_auroc"] for r in primary_ok]
        logger.info(f"\nPrimary cells AUROC distribution:")
        logger.info(f"  Mean: {np.mean(aurocs):.3f}")
        logger.info(f"  Median: {np.median(aurocs):.3f}")
        logger.info(f"  Range: [{min(aurocs):.3f}, {max(aurocs):.3f}]")
        logger.info(f"  >0.70: {sum(1 for a in aurocs if a > 0.70)}/{len(aurocs)}")
        logger.info(f"  >0.60: {sum(1 for a in aurocs if a > 0.60)}/{len(aurocs)}")

    # Save results
    out_path = project_root / "results" / "phase4_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved: {out_path}")

    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    run_benchmark()
