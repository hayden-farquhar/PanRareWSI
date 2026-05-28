"""
PanRareWSI — Data linkage and integrity checks per pre-registration §13.

Cross-references patient IDs between UNI2-h embedding files and cBioPortal
labels. Reports patient-flow diagram, unmatched patients in each direction,
sample-type filtering, and the 10% attrition guard.

Run after embeddings are downloaded and labels are built:
    python3 -m src.linkage_check
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

COHORTS = ["ACC", "UVM", "MESO", "CHOL", "THYM", "KICH", "DLBC"]

EXPECTED_N = {
    "ACC": 93, "UVM": 80, "MESO": 87, "CHOL": 51,
    "THYM": 124, "KICH": 113, "DLBC": 48,
}


def parse_tcga_barcode(filename: str) -> dict | None:
    """Extract patient ID and sample type from a TCGA slide filename.

    TCGA barcodes: TCGA-{TSS}-{participant}-{sample}{vial}-{portion}{analyte}-{plate}-{center}
    Slide files typically: TCGA-XX-XXXX-01Z-00-DX1.{uuid}.h5
    Patient ID = first 12 chars (TCGA-XX-XXXX)
    Sample type = chars 13-14 (01=primary, 02=recurrent, 06=metastatic, 10-14=normal)
    """
    stem = Path(filename).stem
    parts = stem.split(".")
    barcode = parts[0]

    match = re.match(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-(\d{2})", barcode)
    if not match:
        return None
    return {
        "patient_id": match.group(1),
        "sample_type": match.group(2),
        "full_barcode": barcode,
        "filename": filename,
    }


def check_cohort_linkage(
    cohort: str,
    embed_dir: Path,
    label_dir: Path,
) -> dict:
    """Run linkage and integrity checks for one cohort."""
    cohort_embed_dir = embed_dir / cohort
    label_path = label_dir / f"{cohort.lower()}_labels.parquet"

    result = {"cohort": cohort, "issues": []}

    if not cohort_embed_dir.exists():
        result["status"] = "missing_embeddings"
        result["issues"].append(f"Embedding directory not found: {cohort_embed_dir}")
        return result

    if not label_path.exists():
        result["status"] = "missing_labels"
        result["issues"].append(f"Label file not found: {label_path}")
        return result

    # Step 1: Parse embedding filenames (from .h5 files or _manifest.txt)
    h5_files = sorted(cohort_embed_dir.glob("*.h5"))
    manifest_path = cohort_embed_dir / "_manifest.txt"
    if not h5_files and manifest_path.exists():
        slide_ids = manifest_path.read_text().strip().split("\n")
        h5_names = [f"{sid}.h5" for sid in slide_ids]
    else:
        h5_names = [f.name for f in h5_files]
    result["n_slides_total"] = len(h5_names)

    parsed = []
    unparseable = []
    for name in h5_names:
        info = parse_tcga_barcode(name)
        if info:
            parsed.append(info)
        else:
            unparseable.append(name)

    if unparseable:
        result["issues"].append(f"{len(unparseable)} unparseable filenames: {unparseable[:3]}")

    slides_df = pd.DataFrame(parsed)

    # Step 1b: Filter to primary tumour slides (sample type 01)
    primary_mask = slides_df["sample_type"] == "01"
    n_non_primary = (~primary_mask).sum()
    if n_non_primary > 0:
        non_primary_types = slides_df.loc[~primary_mask, "sample_type"].value_counts().to_dict()
        result["non_primary_slides"] = non_primary_types
        logger.info(f"  {cohort}: {n_non_primary} non-primary slides excluded: {non_primary_types}")

    primary_slides = slides_df[primary_mask].copy()
    embed_patients = set(primary_slides["patient_id"].unique())
    result["n_primary_slides"] = len(primary_slides)
    result["n_embed_patients"] = len(embed_patients)

    # Multi-slide patients
    slides_per_patient = primary_slides.groupby("patient_id").size()
    multi_slide = slides_per_patient[slides_per_patient > 1]
    if len(multi_slide) > 0:
        result["multi_slide_patients"] = len(multi_slide)
        result["max_slides_per_patient"] = int(multi_slide.max())
        logger.info(f"  {cohort}: {len(multi_slide)} patients with multiple slides (max {multi_slide.max()})")

    # Step 2: Load labels
    labels_df = pd.read_parquet(label_path)
    label_patients = set(labels_df["patient_id"].unique())
    result["n_label_patients"] = len(label_patients)

    # Step 3: Compute set intersections
    matched = embed_patients & label_patients
    embed_only = embed_patients - label_patients
    label_only = label_patients - embed_patients

    result["n_matched"] = len(matched)
    result["n_embed_only"] = len(embed_only)
    result["n_label_only"] = len(label_only)

    if embed_only:
        result["embed_only_patients"] = sorted(embed_only)
        result["issues"].append(
            f"{len(embed_only)} patients with embeddings but no labels: {sorted(embed_only)[:5]}"
        )

    if label_only:
        result["label_only_patients"] = sorted(label_only)
        if len(label_only) > 5:
            logger.info(f"  {cohort}: {len(label_only)} patients with labels but no embeddings")

    # Step 4: 10% attrition guard
    expected = EXPECTED_N[cohort]
    attrition_pct = 100 * (1 - len(matched) / expected)
    result["expected_n"] = expected
    result["attrition_pct"] = round(attrition_pct, 1)

    if abs(attrition_pct) > 10:
        result["issues"].append(
            f"ATTRITION GUARD: {attrition_pct:+.1f}% from expected n={expected} "
            f"(matched={len(matched)}). Investigate before proceeding."
        )

    result["status"] = "ok" if not result["issues"] else "issues_found"
    return result


def run_all_checks(project_root: Path | None = None) -> list[dict]:
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    embed_dir = project_root / "data" / "embeddings"
    label_dir = project_root / "data" / "labels"

    # Step 5: Record HuggingFace commit hash
    hf_cache = Path.home() / ".cache/huggingface/hub/datasets--MahmoodLab--UNI2-h-features"
    refs_file = hf_cache / "refs" / "main"
    if refs_file.exists():
        commit_hash = refs_file.read_text().strip()
        logger.info(f"UNI2-h-features commit hash: {commit_hash}")
    else:
        commit_hash = "unknown"
        logger.warning("Could not determine UNI2-h-features commit hash")

    results = []
    for cohort in COHORTS:
        logger.info(f"\n{'='*50}")
        logger.info(f"{cohort}")
        logger.info(f"{'='*50}")
        result = check_cohort_linkage(cohort, embed_dir, label_dir)
        result["hf_commit"] = commit_hash
        results.append(result)

        logger.info(f"  Slides: {result.get('n_slides_total', '?')} total, "
                     f"{result.get('n_primary_slides', '?')} primary")
        logger.info(f"  Patients: {result.get('n_embed_patients', '?')} from embeddings, "
                     f"{result.get('n_label_patients', '?')} from labels, "
                     f"{result.get('n_matched', '?')} matched")
        logger.info(f"  Attrition from expected: {result.get('attrition_pct', '?')}%")
        if result["issues"]:
            for issue in result["issues"]:
                logger.warning(f"  ⚠ {issue}")
        else:
            logger.info(f"  ✓ All checks passed")

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("SUMMARY")
    logger.info(f"{'='*50}")
    total_matched = sum(r.get("n_matched", 0) for r in results)
    total_expected = sum(EXPECTED_N.values())
    all_ok = all(r["status"] == "ok" for r in results)

    for r in results:
        flag = "✓" if r["status"] == "ok" else "⚠" if r["status"] == "issues_found" else "✗"
        logger.info(
            f"  {flag} {r['cohort']:6s}: {r.get('n_matched', 0):3d}/{r.get('expected_n', 0)} matched "
            f"({r.get('attrition_pct', '?'):>5}% attrition)"
        )
    logger.info(f"  Total: {total_matched}/{total_expected} matched")
    logger.info(f"  HF commit: {commit_hash}")
    logger.info(f"  Overall: {'PASS' if all_ok else 'ISSUES FOUND'}")

    # Save report
    report_path = project_root / "data" / "linkage_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nReport saved: {report_path}")

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    run_all_checks()
