"""
PanRareWSI — Supplementary label sourcing.

Pulls labels that are NOT in the base cBioPortal clinical-data endpoint:
  1. CDKN2A homozygous deletion (MESO) from GISTIC CNA
  2. UVM chromosome 3 status from Firehose cytogenetic abnormality
  3. DLBC ABC/GCB subtype via Hans algorithm from Firehose IHC markers
  4. Thorsson et al. 2018 immune subtypes (C1–C6) from PanCancer immune TSV

Pre-registration: https://doi.org/10.17605/OSF.IO/Y6HVP
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CBIO_BASE = "https://www.cbioportal.org/api"

PANCAN_STUDIES = {
    "ACC": "acc_tcga_pan_can_atlas_2018",
    "UVM": "uvm_tcga_pan_can_atlas_2018",
    "MESO": "meso_tcga_pan_can_atlas_2018",
    "CHOL": "chol_tcga_pan_can_atlas_2018",
    "THYM": "thym_tcga_pan_can_atlas_2018",
    "KICH": "kich_tcga_pan_can_atlas_2018",
    "DLBC": "dlbc_tcga_pan_can_atlas_2018",
}

FIREHOSE_STUDIES = {
    "ACC": "acc_tcga",
    "UVM": "uvm_tcga",
    "MESO": "meso_tcga",
    "CHOL": "chol_tcga",
    "THYM": "thym_tcga",
    "KICH": "kich_tcga",
    "DLBC": "dlbc_tcga",
}


def _cbio_get(endpoint: str):
    url = f"{CBIO_BASE}{endpoint}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _get_clinical(study: str, attr: str) -> dict[str, str]:
    for dtype in ("PATIENT", "SAMPLE"):
        data = _cbio_get(
            f"/studies/{study}/clinical-data"
            f"?clinicalDataType={dtype}&attributeId={attr}"
        )
        if data:
            return {
                d["patientId"]: d["value"]
                for d in data
                if d.get("value")
                and d["value"] not in ("NA", "NaN", "[Not Available]", "[Not Applicable]")
            }
    return {}


# ── 1. CDKN2A homozygous deletion (MESO) ────────────────────────────────

def get_cdkn2a_homdel(cohort: str = "MESO") -> dict[str, int]:
    study = PANCAN_STUDIES[cohort]
    gene_info = _cbio_get("/genes/CDKN2A")
    entrez = gene_info["entrezGeneId"]
    gistic_profile = f"{study}_gistic"
    data = _cbio_get(
        f"/molecular-profiles/{gistic_profile}/molecular-data"
        f"?entrezGeneId={entrez}&sampleListId={study}_all"
    )
    result = {}
    for d in data:
        val = int(d["value"])
        result[d["patientId"]] = 1 if val == -2 else 0
    return result


# ── 2. UVM chromosome 3 status ──────────────────────────────────────────

def get_uvm_chr3_status() -> dict[str, int]:
    cyto = _get_clinical(FIREHOSE_STUDIES["UVM"], "CYTOGENETIC_ABNORMALITY_TYPE")
    result = {}
    for pid, val in cyto.items():
        has_chr3_loss = "chromosome 3 loss" in val.lower()
        result[pid] = 1 if has_chr3_loss else 0
    return result


# ── 3. DLBC ABC/GCB via Hans algorithm ──────────────────────────────────

def get_dlbc_hans_subtype() -> dict[str, str]:
    """
    Hans algorithm: CD10+ → GCB; CD10− & BCL6+ & MUM1− → GCB; else → non-GCB (ABC).
    Extracts marker status from paired IMMUNOPHENOTYPIC_ANALYSIS_TESTED / _RESULTS.
    """
    study = FIREHOSE_STUDIES["DLBC"]
    tested = _get_clinical(study, "IMMUNOPHENOTYPIC_ANALYSIS_TESTED")
    results = _get_clinical(study, "IMMUNOPHENOTYPIC_ANALYSIS_RESULTS")

    subtypes = {}
    for pid in tested:
        if pid not in results:
            continue
        markers = tested[pid].split("|")
        values = results[pid].split("|")
        if len(markers) != len(values):
            continue

        marker_status = {}
        for m, v in zip(markers, values):
            m_clean = m.strip().upper()
            v_clean = v.strip().lower()
            if v_clean == "positive":
                marker_status[m_clean] = True
            elif v_clean == "negative":
                marker_status[m_clean] = False

        cd10 = marker_status.get("CD10 > 30%")
        bcl6 = marker_status.get("BCL6 > 30%")
        mum1 = marker_status.get("MUM1 > 30%")

        if cd10 is True:
            subtypes[pid] = "GCB"
        elif cd10 is False and bcl6 is True and mum1 is False:
            subtypes[pid] = "GCB"
        elif cd10 is False:
            subtypes[pid] = "non-GCB"
        # If CD10 missing, skip

    return subtypes


# ── 4. Thorsson immune subtypes (C1–C6) ─────────────────────────────────

THORSSON_XENA_URL = (
    "https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com"
    "/download/Subtype_Immune_Model_Based.txt.gz"
)

TCGA_COHORT_TSS = {
    "ACC": {"OR"},
    "UVM": {"V4", "V3", "VD", "YZ", "WC"},
    "MESO": {"3U", "3H", "TS", "UD"},
    "CHOL": {"W5", "ZH"},
    "THYM": {"3G", "3Q", "4V", "X7", "ZB"},
    "KICH": {"KL", "KN", "KO"},
    "DLBC": {"FA", "FF", "FM", "GR", "GS", "HS"},
}


def get_thorsson_immune_subtypes() -> dict[str, str]:
    """
    Download Thorsson et al. 2018 immune subtype assignments from UCSC Xena.
    Returns {patient_id: immune_subtype_code}.
    """
    import gzip

    logger.info("Downloading Thorsson immune subtypes from UCSC Xena...")
    req = urllib.request.Request(THORSSON_XENA_URL)
    req.add_header("User-Agent", "PanRareWSI/1.0")
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = gzip.decompress(resp.read())

    result = {}
    for line in raw.decode().split("\n"):
        if not line or line.startswith("sample"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        barcode = parts[0]
        subtype_full = parts[1]

        pid = "-".join(barcode.split("-")[:3])

        match = re.search(r"Immune (C\d)", subtype_full)
        code = match.group(1) if match else subtype_full
        result[pid] = code

    return result


# ── Main: merge supplementary labels into existing parquets ─────────────

def merge_supplementary_labels(label_dir: str | Path) -> None:
    label_dir = Path(label_dir)

    # --- MESO: CDKN2A homdel ---
    logger.info("\n=== MESO: CDKN2A homozygous deletion ===")
    cdkn2a = get_cdkn2a_homdel("MESO")
    n_pos = sum(v for v in cdkn2a.values())
    logger.info(f"  CDKN2A homdel: {n_pos}/{len(cdkn2a)} ({100*n_pos/len(cdkn2a):.0f}%)")

    meso_path = label_dir / "meso_labels.parquet"
    meso = pd.read_parquet(meso_path)
    meso["label_cdkn2a_homdel"] = meso["patient_id"].map(cdkn2a)
    meso.to_parquet(meso_path, index=False)
    logger.info(f"  Updated {meso_path}")

    # --- UVM: chromosome 3 status ---
    logger.info("\n=== UVM: Chromosome 3 loss ===")
    chr3 = get_uvm_chr3_status()
    n_pos = sum(v for v in chr3.values())
    logger.info(f"  Chr3 loss: {n_pos}/{len(chr3)} ({100*n_pos/len(chr3):.0f}%)")

    uvm_path = label_dir / "uvm_labels.parquet"
    uvm = pd.read_parquet(uvm_path)
    uvm["label_chr3_loss"] = uvm["patient_id"].map(chr3)
    n_mapped = uvm["label_chr3_loss"].notna().sum()
    logger.info(f"  Mapped to {n_mapped}/{len(uvm)} UVM patients (Firehose subset)")
    uvm.to_parquet(uvm_path, index=False)
    logger.info(f"  Updated {uvm_path}")

    # --- DLBC: Hans algorithm ABC/GCB ---
    logger.info("\n=== DLBC: Hans algorithm subtype ===")
    hans = get_dlbc_hans_subtype()
    if hans:
        counts = Counter(hans.values())
        logger.info(f"  Hans subtypes: {dict(counts)}")

        dlbc_path = label_dir / "dlbc_labels.parquet"
        dlbc = pd.read_parquet(dlbc_path)
        dlbc["label_hans_subtype"] = dlbc["patient_id"].map(hans)
        n_mapped = dlbc["label_hans_subtype"].notna().sum()
        logger.info(f"  Mapped to {n_mapped}/{len(dlbc)} DLBC patients")
        dlbc["label_hans_nongcb"] = dlbc["label_hans_subtype"].map(
            {"GCB": 0, "non-GCB": 1}
        )
        dlbc.to_parquet(dlbc_path, index=False)
        logger.info(f"  Updated {dlbc_path}")
    else:
        logger.warning("  No Hans subtypes could be derived")

    # --- Thorsson immune subtypes ---
    logger.info("\n=== Thorsson immune subtypes (C1–C6) ===")
    try:
        all_subtypes = get_thorsson_immune_subtypes()
        logger.info(f"  Total TCGA samples with immune subtype: {len(all_subtypes)}")

        for cohort in PANCAN_STUDIES:
            cohort_path = label_dir / f"{cohort.lower()}_labels.parquet"
            if not cohort_path.exists():
                continue
            df = pd.read_parquet(cohort_path)
            df["immune_subtype"] = df["patient_id"].map(all_subtypes)
            n_mapped = df["immune_subtype"].notna().sum()
            logger.info(f"  {cohort}: {n_mapped}/{len(df)} mapped")
            if n_mapped > 0:
                counts = df["immune_subtype"].value_counts()
                logger.info(f"    Distribution: {dict(counts)}")
            df.to_parquet(cohort_path, index=False)
    except Exception as e:
        logger.warning(f"  Thorsson download failed: {e}")
        logger.info("  Immune subtypes can be added later from manual download")

    # --- Rebuild missingness report ---
    logger.info("\n=== Rebuilding missingness report ===")
    from data_loaders import build_missingness_report
    all_miss = []
    for cohort in PANCAN_STUDIES:
        cohort_path = label_dir / f"{cohort.lower()}_labels.parquet"
        if cohort_path.exists():
            df = pd.read_parquet(cohort_path)
            miss = build_missingness_report(df, cohort)
            all_miss.append(miss)
    miss_df = pd.concat(all_miss, ignore_index=True)
    miss_df.to_parquet(label_dir / "missingness_report.parquet", index=False)
    miss_df.to_csv(label_dir / "missingness_report.csv", index=False)
    logger.info("  Missingness report updated")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    project_root = Path(__file__).resolve().parent.parent
    merge_supplementary_labels(project_root / "data" / "labels")
