"""
PanRareWSI — cBioPortal label extraction and data pipeline.

Pulls mutation, clinical, and molecular profile data for the 7 rare TCGA cohorts
and constructs per-cohort label parquet files.

Pre-registration: https://doi.org/10.17605/OSF.IO/Y6HVP
"""

from __future__ import annotations

import json
import logging
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CBIO_BASE = "https://www.cbioportal.org/api"

COHORTS = {
    "ACC": "acc_tcga_pan_can_atlas_2018",
    "UVM": "uvm_tcga_pan_can_atlas_2018",
    "MESO": "meso_tcga_pan_can_atlas_2018",
    "CHOL": "chol_tcga_pan_can_atlas_2018",
    "THYM": "thym_tcga_pan_can_atlas_2018",
    "KICH": "kich_tcga_pan_can_atlas_2018",
    "DLBC": "dlbc_tcga_pan_can_atlas_2018",
}

FIREHOSE_COHORTS = {
    "ACC": "acc_tcga",
    "UVM": "uvm_tcga",
    "MESO": "meso_tcga",
    "CHOL": "chol_tcga",
    "THYM": "thym_tcga",
    "KICH": "kich_tcga",
    "DLBC": "dlbc_tcga",
}

DRIVER_GENES: dict[str, list[str]] = {
    "ACC": ["TP53", "CTNNB1", "MEN1", "ZNRF3", "PRKAR1A"],
    "UVM": ["BAP1", "SF3B1", "GNAQ", "GNA11", "EIF1AX"],
    "MESO": ["BAP1", "NF2", "TP53", "SETD2", "CDKN2A"],
    "CHOL": ["IDH1", "IDH2", "TP53", "KRAS", "ARID1A"],
    "THYM": ["GTF2I", "HRAS", "TP53", "NRAS"],
    "KICH": ["TP53", "PTEN", "TERT"],
    "DLBC": ["MYD88", "CD79B", "TP53", "BCL2", "EZH2"],
}


def _cbio_get(endpoint: str) -> Any:
    url = f"{CBIO_BASE}{endpoint}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _cbio_post(endpoint: str, body: dict) -> Any:
    url = f"{CBIO_BASE}{endpoint}"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def get_patients(study_id: str) -> list[str]:
    data = _cbio_get(f"/studies/{study_id}/patients")
    return [p["patientId"] for p in data]


def get_clinical_data(study_id: str, attr_id: str) -> dict[str, str]:
    for dtype in ("PATIENT", "SAMPLE"):
        data = _cbio_get(
            f"/studies/{study_id}/clinical-data"
            f"?clinicalDataType={dtype}&attributeId={attr_id}"
        )
        if data:
            return {
                d["patientId"]: d["value"]
                for d in data
                if d.get("value")
                and d["value"] not in ("NA", "NaN", "[Not Available]", "[Not Applicable]")
            }
    return {}


def get_mutation_status(study_id: str, gene_symbol: str) -> dict[str, int]:
    gene_info = _cbio_get(f"/genes/{gene_symbol}")
    entrez_id = gene_info["entrezGeneId"]
    mut_profile = f"{study_id}_mutations"
    mutations = _cbio_get(
        f"/molecular-profiles/{mut_profile}/mutations"
        f"?entrezGeneId={entrez_id}&sampleListId={study_id}_all&projection=SUMMARY"
    )
    mutated = set(m["patientId"] for m in mutations)
    all_patients = get_patients(study_id)
    return {pid: (1 if pid in mutated else 0) for pid in all_patients}


def get_histological_subtypes(cohort: str) -> dict[str, str]:
    study_id = FIREHOSE_COHORTS[cohort]
    return get_clinical_data(study_id, "HISTOLOGICAL_DIAGNOSIS")


def build_cohort_labels(cohort: str) -> pd.DataFrame:
    study_id = COHORTS[cohort]
    patients = get_patients(study_id)
    logger.info(f"{cohort}: {len(patients)} patients from {study_id}")

    df = pd.DataFrame({"patient_id": patients})
    df["cohort"] = cohort

    # --- Mutation labels ---
    for gene in DRIVER_GENES[cohort]:
        try:
            mut = get_mutation_status(study_id, gene)
            df[f"mut_{gene}"] = df["patient_id"].map(mut)
            n_pos = df[f"mut_{gene}"].sum()
            logger.info(f"  {gene}: {n_pos}/{len(patients)} mutated ({100*n_pos/len(patients):.0f}%)")
        except Exception as e:
            logger.warning(f"  {gene}: FAILED — {e}")
            df[f"mut_{gene}"] = np.nan

    # --- Continuous genomic features (for Q75 binarisation) ---
    for attr, col in [
        ("MSI_SCORE_MANTIS", "msi_mantis"),
        ("MSI_SENSOR_SCORE", "msi_sensor"),
        ("MUTATION_COUNT", "mutation_count"),
        ("TMB_NONSYNONYMOUS", "tmb"),
        ("ANEUPLOIDY_SCORE", "aneuploidy"),
        ("FRACTION_GENOME_ALTERED", "fga"),
    ]:
        vals = get_clinical_data(study_id, attr)
        df[col] = df["patient_id"].map(vals).apply(
            lambda x: float(x) if pd.notna(x) else np.nan
        )
        n_valid = df[col].notna().sum()
        logger.info(f"  {attr}: {n_valid}/{len(patients)} non-null")

    # --- Histological subtypes (from Firehose Legacy) ---
    try:
        subtypes = get_histological_subtypes(cohort)
        df["histological_subtype"] = df["patient_id"].map(subtypes)
        counts = df["histological_subtype"].value_counts()
        logger.info(f"  Histological subtypes: {dict(counts)}")
    except Exception as e:
        logger.warning(f"  Histological subtypes: FAILED — {e}")
        df["histological_subtype"] = np.nan

    # --- Clinical stage and grade ---
    for attr, col in [
        ("AJCC_PATHOLOGIC_TUMOR_STAGE", "stage"),
        ("GRADE", "grade"),
        ("SEX", "sex"),
        ("AGE", "age"),
    ]:
        vals = get_clinical_data(study_id, attr)
        df[col] = df["patient_id"].map(vals)

    return df


def binarise_q75(series: pd.Series, training_mask: pd.Series | None = None) -> pd.Series:
    """Binarise at Q75 threshold. If training_mask provided, compute Q75 on training only."""
    if training_mask is not None:
        threshold = series[training_mask].quantile(0.75)
    else:
        threshold = series.quantile(0.75)
    return (series >= threshold).astype(float).where(series.notna())


def add_derived_binary_labels(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Add the pre-registered binary biomarker labels."""
    df = df.copy()

    # MSI-H (cohort-specific definitions per pre-registration §10.5)
    if cohort == "DLBC":
        df["label_msi_h"] = (df["msi_mantis"] >= 0.4).astype(float).where(df["msi_mantis"].notna())
    elif cohort == "KICH":
        df["label_msi_h"] = (df["msi_sensor"] >= 3.5).astype(float).where(df["msi_sensor"].notna())

    # Q75-binarised labels (threshold computed on full cohort for now;
    # per-fold thresholds applied during training per pre-reg §17.2)
    if "tmb" in df.columns:
        df["label_tmb_high"] = binarise_q75(df["tmb"])
    if "fga" in df.columns:
        df["label_fga_high"] = binarise_q75(df["fga"])
    if "aneuploidy" in df.columns:
        df["label_aneuploidy_high"] = binarise_q75(df["aneuploidy"])

    # Histological subtypes — cohort-specific encoding
    if cohort == "UVM" and "histological_subtype" in df.columns:
        mapping = {}
        for val in df["histological_subtype"].dropna().unique():
            v = val.lower()
            if "epithelioid" in v and "spindle" in v:
                mapping[val] = "Mixed"
            elif "spindle" in v and "epithelioid" in v:
                mapping[val] = "Mixed"
            elif "spindle" in v:
                mapping[val] = "Spindle"
            elif "epithelioid" in v:
                mapping[val] = "Epithelioid"
            else:
                mapping[val] = "Mixed"
        df["label_histology_3class"] = df["histological_subtype"].map(mapping)

    if cohort == "MESO" and "histological_subtype" in df.columns:
        df["label_histology_epi"] = df["histological_subtype"].apply(
            lambda x: 0 if pd.isna(x) else (1 if "epithelioid" in str(x).lower() else 0)
        ).where(df["histological_subtype"].notna())

    if cohort == "THYM" and "histological_subtype" in df.columns:
        who_map = {}
        for val in df["histological_subtype"].dropna().unique():
            v = val.lower()
            if "type ab" in v:
                who_map[val] = "AB"
            elif "type a" in v:
                who_map[val] = "A"
            elif "type b1" in v:
                who_map[val] = "B1"
            elif "type b2" in v:
                who_map[val] = "B2"
            elif "type b3" in v:
                who_map[val] = "B3"
            elif "carcinoma" in v:
                who_map[val] = "TC"
            elif "micronodular" in v or "mn" in v:
                who_map[val] = "MN"
            else:
                who_map[val] = val
        df["label_who_subtype"] = df["histological_subtype"].map(who_map)

    return df


def build_missingness_report(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith(("mut_", "label_", "msi_", "tmb", "fga", "aneuploidy"))]
    rows = []
    for col in cols:
        n_total = len(df)
        n_valid = df[col].notna().sum()
        n_missing = n_total - n_valid
        if df[col].dtype in ("float64", "int64") and df[col].dropna().isin([0, 1]).all():
            n_pos = int(df[col].sum())
            prevalence = n_pos / n_valid if n_valid > 0 else 0
        else:
            n_pos = None
            prevalence = None
        rows.append({
            "cohort": cohort,
            "variable": col,
            "n_total": n_total,
            "n_valid": n_valid,
            "n_missing": n_missing,
            "pct_missing": 100 * n_missing / n_total,
            "n_positive": n_pos,
            "prevalence": prevalence,
        })
    return pd.DataFrame(rows)


def pull_all_labels(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_missingness = []

    for cohort in COHORTS:
        logger.info(f"\n{'='*60}\n{cohort}\n{'='*60}")

        df = build_cohort_labels(cohort)
        df = add_derived_binary_labels(df, cohort)

        out_path = output_dir / f"{cohort.lower()}_labels.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"  Saved {out_path} ({len(df)} rows, {len(df.columns)} cols)")

        miss = build_missingness_report(df, cohort)
        all_missingness.append(miss)

    miss_df = pd.concat(all_missingness, ignore_index=True)
    miss_path = output_dir / "missingness_report.parquet"
    miss_df.to_parquet(miss_path, index=False)
    logger.info(f"\nMissingness report: {miss_path}")

    miss_csv = output_dir / "missingness_report.csv"
    miss_df.to_csv(miss_csv, index=False)
    logger.info(f"Missingness CSV: {miss_csv}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    project_root = Path(__file__).resolve().parent.parent
    pull_all_labels(project_root / "data" / "labels")
