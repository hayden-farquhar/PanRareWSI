"""
PanRareWSI — Generate frozen patient-level stratified 5-fold CV splits.

Per pre-registration §17: patient-level stratified 5-fold CV with seed=42.
Stratification is on the most prevalent binary biomarker per cohort.
Splits are frozen at generation time and deposited to OSF.

Pre-registration: https://doi.org/10.17605/OSF.IO/Y6HVP
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

SEED = 42
N_FOLDS = 5

STRATIFICATION_TARGET: dict[str, str] = {
    "ACC": "mut_TP53",
    "UVM": "mut_GNAQ",
    "MESO": "label_cdkn2a_homdel",
    "CHOL": "mut_IDH1",
    "THYM": "mut_GTF2I",
    "KICH": "mut_TP53",
    "DLBC": "label_msi_h",
}


def build_splits(label_dir: str | Path, split_dir: str | Path) -> None:
    label_dir = Path(label_dir)
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "seed": SEED,
        "n_folds": N_FOLDS,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stratification_targets": STRATIFICATION_TARGET,
        "cohorts": {},
    }

    for cohort, target_col in STRATIFICATION_TARGET.items():
        cohort_path = label_dir / f"{cohort.lower()}_labels.parquet"
        if not cohort_path.exists():
            logger.warning(f"{cohort}: label file not found, skipping")
            continue

        df = pd.read_parquet(cohort_path)
        logger.info(f"\n{cohort}: {len(df)} patients, stratifying on {target_col}")

        if target_col not in df.columns:
            logger.warning(f"  {target_col} not found in columns, falling back to unstratified")
            strat_labels = np.zeros(len(df))
        else:
            strat_labels = df[target_col].fillna(-1).values

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        X_dummy = np.zeros((len(df), 1))

        fold_assignments = np.full(len(df), -1, dtype=int)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_dummy, strat_labels)):
            fold_assignments[val_idx] = fold_idx

        split_df = pd.DataFrame({
            "patient_id": df["patient_id"],
            "fold": fold_assignments,
        })

        for fold_idx in range(N_FOLDS):
            fold_mask = split_df["fold"] == fold_idx
            n_fold = fold_mask.sum()
            if target_col in df.columns:
                fold_pos = df.loc[fold_mask.values, target_col].sum()
                fold_total = df.loc[fold_mask.values, target_col].notna().sum()
                prev = fold_pos / fold_total if fold_total > 0 else 0
                logger.info(f"  Fold {fold_idx}: n={n_fold}, {target_col} prev={prev:.2f}")
            else:
                logger.info(f"  Fold {fold_idx}: n={n_fold}")

        out_path = split_dir / f"{cohort.lower()}_splits.csv"
        split_df.to_csv(out_path, index=False)
        logger.info(f"  Saved {out_path}")

        content_hash = hashlib.sha256(
            split_df.to_csv(index=False).encode()
        ).hexdigest()[:16]
        metadata["cohorts"][cohort] = {
            "n_patients": len(df),
            "stratification_target": target_col,
            "sha256_prefix": content_hash,
            "fold_sizes": [int((split_df["fold"] == i).sum()) for i in range(N_FOLDS)],
        }

    meta_path = split_dir / "split_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"\nMetadata: {meta_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    project_root = Path(__file__).resolve().parent.parent
    build_splits(
        project_root / "data" / "labels",
        project_root / "data" / "splits",
    )
