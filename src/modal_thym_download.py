"""
PanRareWSI — Download THYM embeddings on Modal and return mean-pooled features.

Runs in Modal's datacenter to avoid local disk constraints. Downloads the
18 GB TCGA-THYM archive from HuggingFace, extracts .h5 files, computes
mean-pooled slide-level features (1536-dim), and returns them locally.

Also saves patch-level embeddings to a Modal Volume for later ABMIL runs.

Usage:
    modal run src/modal_thym_download.py

Requires: `modal` CLI authenticated (`modal profile list`).
"""

import modal

app = modal.App("panrarewsi-thym")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub>=0.20", "h5py>=3.9", "numpy>=1.24", "tqdm>=4.65")
)

volume = modal.Volume.from_name("panrarewsi-embeddings", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/embeddings": volume},
    timeout=1800,
    memory=8192,
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
def download_and_pool_thym() -> dict:
    import json
    import os
    import shutil
    import tarfile
    from pathlib import Path

    import h5py
    import numpy as np
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm

    cohort_dir = Path("/embeddings/THYM")
    manifest_file = cohort_dir / "_manifest.txt"

    if manifest_file.exists():
        print(f"THYM already extracted on volume ({len(list(cohort_dir.glob('*.h5')))} slides)")
    else:
        print("Downloading TCGA-THYM.tar.gz from HuggingFace...")
        archive_path = hf_hub_download(
            "MahmoodLab/UNI2-h-features",
            "TCGA/TCGA-THYM.tar.gz",
            repo_type="dataset",
            token=os.environ["HF_TOKEN"],
        )
        archive_path = Path(archive_path)
        print(f"Downloaded: {archive_path.stat().st_size / 1e9:.1f} GB")

        cohort_dir.mkdir(parents=True, exist_ok=True)
        n_extracted = 0
        print("Extracting .h5 files...")
        with tarfile.open(archive_path, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(".h5")]
            for member in tqdm(members, desc="THYM extract", unit="slide"):
                target = cohort_dir / Path(member.name).name
                with tar.extractfile(member) as src:
                    with open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                n_extracted += 1

        slide_ids = sorted(p.stem for p in cohort_dir.glob("*.h5"))
        manifest_file.write_text("\n".join(slide_ids) + "\n")
        print(f"Extracted {n_extracted} slides")
        volume.commit()

    # Mean-pool all slides
    print("Computing mean-pooled slide features...")
    h5_files = sorted(cohort_dir.glob("*.h5"))
    results = {}
    for h5_path in tqdm(h5_files, desc="Mean-pool", unit="slide"):
        with h5py.File(h5_path, "r") as f:
            features = f["features"][:]  # (1, n_patches, 1536)
            coords = f["coords"][:]
        mean_feat = features[0].mean(axis=0)  # (1536,)
        slide_id = h5_path.stem
        results[slide_id] = {
            "mean_features": mean_feat.tolist(),
            "n_patches": features.shape[1],
            "feature_dim": features.shape[2],
        }

    print(f"Pooled {len(results)} slides, dim={results[next(iter(results))]['feature_dim']}")
    return results


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    print("Launching THYM download + mean-pool on Modal...")
    results = download_and_pool_thym.remote()

    # Save locally
    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / "data" / "embeddings" / "THYM"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save mean-pooled features as numpy arrays
    import numpy as np

    slide_ids = sorted(results.keys())
    features_matrix = np.array([results[sid]["mean_features"] for sid in slide_ids], dtype=np.float32)
    n_patches = [results[sid]["n_patches"] for sid in slide_ids]

    np.save(out_dir / "mean_pooled_features.npy", features_matrix)

    manifest = {"slide_ids": slide_ids, "n_patches": n_patches, "source": "modal_mean_pool"}
    with open(out_dir / "mean_pool_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Write standard manifest for linkage check compatibility
    (out_dir / "_manifest.txt").write_text("\n".join(slide_ids) + "\n")

    print(f"\nSaved to {out_dir}:")
    print(f"  mean_pooled_features.npy: {features_matrix.shape} ({features_matrix.nbytes / 1e6:.1f} MB)")
    print(f"  mean_pool_manifest.json: {len(slide_ids)} slides")
    print(f"  _manifest.txt: {len(slide_ids)} entries")
    print(f"\nPatch-level embeddings stored on Modal Volume 'panrarewsi-embeddings'")
    print(f"  (access via `modal volume ls panrarewsi-embeddings /THYM/`)")
