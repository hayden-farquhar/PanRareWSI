"""
PanRareWSI — Mean-pool patch-level UNI2-h embeddings to slide-level features.

For each cohort with local .h5 files, computes the mean across all patches
to produce a single 1536-dim vector per slide. Saves as .npy + manifest JSON.

THYM is skipped (already mean-pooled via Modal).

Usage:
    python3 -m src.mean_pool_local
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

COHORTS_LOCAL = ["ACC", "UVM", "MESO", "CHOL", "KICH", "DLBC"]


def mean_pool_cohort(cohort: str, embed_dir: Path) -> dict:
    cohort_dir = embed_dir / cohort
    out_npy = cohort_dir / "mean_pooled_features.npy"
    out_manifest = cohort_dir / "mean_pool_manifest.json"

    if out_npy.exists():
        arr = np.load(out_npy)
        manifest = json.loads(out_manifest.read_text())
        logger.info(f"{cohort}: already pooled ({arr.shape[0]} slides) — skipping")
        return {"cohort": cohort, "status": "skipped", "n_slides": arr.shape[0]}

    h5_files = sorted(cohort_dir.glob("*.h5"))
    if not h5_files:
        logger.warning(f"{cohort}: no .h5 files found")
        return {"cohort": cohort, "status": "no_h5_files", "n_slides": 0}

    slide_ids = []
    features_list = []
    n_patches_list = []

    for h5_path in tqdm(h5_files, desc=f"{cohort} pool", unit="slide"):
        with h5py.File(h5_path, "r") as f:
            feats = f["features"][:]  # (1, n_patches, 1536)
        mean_feat = feats[0].mean(axis=0)
        slide_ids.append(h5_path.stem)
        features_list.append(mean_feat)
        n_patches_list.append(feats.shape[1])

    features_matrix = np.array(features_list, dtype=np.float32)
    np.save(out_npy, features_matrix)

    manifest = {
        "slide_ids": slide_ids,
        "n_patches": n_patches_list,
        "source": "local_mean_pool",
    }
    out_manifest.write_text(json.dumps(manifest, indent=2))

    logger.info(f"{cohort}: {features_matrix.shape[0]} slides → {out_npy.name} ({features_matrix.nbytes / 1e6:.1f} MB)")
    return {"cohort": cohort, "status": "ok", "n_slides": features_matrix.shape[0], "shape": features_matrix.shape}


def pool_all(embed_dir: Path | None = None) -> list[dict]:
    if embed_dir is None:
        embed_dir = Path(__file__).resolve().parent.parent / "data" / "embeddings"

    results = []
    for cohort in COHORTS_LOCAL:
        result = mean_pool_cohort(cohort, embed_dir)
        results.append(result)

    logger.info("\nSummary:")
    for r in results:
        logger.info(f"  {r['cohort']:6s}: {r['status']:10s} ({r.get('n_slides', 0)} slides)")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    pool_all()
