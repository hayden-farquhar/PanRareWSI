"""
PanRareWSI — Download and extract UNI2-h patch-level embeddings from HuggingFace.

Downloads gated TCGA cohort archives from MahmoodLab/UNI2-h-features,
extracts .h5 files to data/embeddings/{cohort}/, then deletes the archive
to conserve disk space. Processes cohorts smallest-first.

Requires: huggingface-cli login (gated access for hayden-farquhar).

Each .h5 file contains:
  - features: [1, n_patches, 1536]  (float32 UNI2-h ViT-H/14 embeddings)
  - coords:   [1, n_patches, 2]     (patch coordinates at 20x / 256px)
"""

from __future__ import annotations

import logging
import os
import shutil
import tarfile
from pathlib import Path

import h5py
from huggingface_hub import hf_hub_download
from tqdm import tqdm

logger = logging.getLogger(__name__)

REPO_ID = "MahmoodLab/UNI2-h-features"

COHORTS_ORDERED_BY_SIZE = [
    ("DLBC", "TCGA/TCGA-DLBC.tar.gz", 1_497),
    ("CHOL", "TCGA/TCGA-CHOL.tar.gz", 4_465),
    ("UVM", "TCGA/TCGA-UVM.tar.gz", 4_459),
    ("MESO", "TCGA/TCGA-MESO.tar.gz", 4_810),
    ("KICH", "TCGA/TCGA-KICH.tar.gz", 5_239),
    ("ACC", "TCGA/TCGA-ACC.tar.gz", 10_960),
    ("THYM", "TCGA/TCGA-THYM.tar.gz", 18_113),
]


def _free_gb(path: Path) -> float:
    st = os.statvfs(path)
    return (st.f_bavail * st.f_frsize) / 1e9


def _validate_h5(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        feats = f["features"]
        coords = f["coords"]
        return {
            "features_shape": feats.shape,
            "coords_shape": coords.shape,
            "dtype": str(feats.dtype),
        }


def download_and_extract(
    cohort: str,
    hf_filename: str,
    embed_dir: Path,
    *,
    archive_size_mb: int = 0,
    min_free_gb: float = 5.0,
) -> dict:
    """Download one cohort archive, extract .h5 files, delete archive."""
    cohort_dir = embed_dir / cohort
    manifest_file = cohort_dir / "_manifest.txt"

    if manifest_file.exists():
        n_files = len(list(cohort_dir.glob("*.h5")))
        logger.info(f"{cohort}: already extracted ({n_files} slides) — skipping")
        return {"cohort": cohort, "status": "skipped", "n_slides": n_files}

    # Need space for: archive in cache + extracted .h5 files + 5 GB headroom.
    # Extracted size ≈ compressed size (HDF5 internal compression).
    needed_gb = max(min_free_gb, (archive_size_mb * 2.2) / 1000 + 5)
    free = _free_gb(embed_dir)
    logger.info(f"{cohort}: {free:.1f} GB free, need ~{needed_gb:.1f} GB (archive ~{archive_size_mb/1000:.1f} GB)")
    if free < needed_gb:
        logger.error(
            f"{cohort}: only {free:.1f} GB free (need ~{needed_gb:.1f} GB for "
            f"download + extraction) — skipping"
        )
        return {"cohort": cohort, "status": "insufficient_disk", "free_gb": free}

    logger.info(f"{cohort}: downloading {hf_filename} from HuggingFace...")
    archive_path = hf_hub_download(
        REPO_ID,
        hf_filename,
        repo_type="dataset",
    )
    archive_path = Path(archive_path)
    archive_mb = archive_path.stat().st_size / 1e6
    logger.info(f"{cohort}: downloaded ({archive_mb:.0f} MB cached at {archive_path})")

    cohort_dir.mkdir(parents=True, exist_ok=True)
    n_extracted = 0
    logger.info(f"{cohort}: extracting .h5 files...")
    with tarfile.open(archive_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".h5")]
        for member in tqdm(members, desc=f"{cohort} extract", unit="slide"):
            member_basename = Path(member.name).name
            target = cohort_dir / member_basename
            with tar.extractfile(member) as src:
                with open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            n_extracted += 1

    slide_ids = sorted(p.stem for p in cohort_dir.glob("*.h5"))
    manifest_file.write_text("\n".join(slide_ids) + "\n")

    sample_h5 = next(cohort_dir.glob("*.h5"))
    info = _validate_h5(sample_h5)
    logger.info(
        f"{cohort}: {n_extracted} slides extracted. "
        f"Sample shape: features={info['features_shape']}, dtype={info['dtype']}"
    )

    # Clean up the HF cache entry for this file to reclaim disk space
    _purge_cache_blob(archive_path)

    return {
        "cohort": cohort,
        "status": "ok",
        "n_slides": n_extracted,
        "sample_features_shape": info["features_shape"],
        "sample_coords_shape": info["coords_shape"],
    }


def _purge_cache_blob(cached_path: Path) -> None:
    """Remove the cached blob and its snapshot symlink to free disk space."""
    blob = cached_path.resolve()
    if blob.exists():
        size_gb = blob.stat().st_size / 1e9
        blob.unlink()
        logger.info(f"  Purged cache blob ({size_gb:.1f} GB reclaimed)")
    if cached_path != blob and cached_path.is_symlink():
        cached_path.unlink()


def download_all(embed_dir: Path | None = None, min_free_gb: float = 5.0) -> list[dict]:
    project_root = Path(__file__).resolve().parent.parent
    if embed_dir is None:
        embed_dir = project_root / "data" / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for cohort, hf_filename, _size_mb in COHORTS_ORDERED_BY_SIZE:
        result = download_and_extract(
            cohort, hf_filename, embed_dir,
            archive_size_mb=_size_mb, min_free_gb=min_free_gb,
        )
        results.append(result)
        logger.info(f"  → {result}")

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    total_slides = 0
    for r in results:
        n = r.get("n_slides", 0)
        total_slides += n
        logger.info(f"  {r['cohort']:6s}: {r['status']:20s} ({n} slides)")
    logger.info(f"  {'TOTAL':6s}: {total_slides} slides")
    logger.info(f"  Disk free: {_free_gb(embed_dir):.1f} GB")

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    download_all()
