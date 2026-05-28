"""
PanRareWSI — Download TCGA-COAD UNI2-h embeddings on Modal, return mean-pooled
slide features for the TMB-high cross-cohort transfer experiment (RQ3).

Usage:
    modal run src/modal_coad_meanpool.py
"""

import modal

app = modal.App("panrarewsi-coad")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub>=0.20", "h5py>=3.9", "numpy>=1.24", "tqdm>=4.65")
)
volume = modal.Volume.from_name("panrarewsi-embeddings", create_if_missing=True)


@app.function(image=image, volumes={"/vol": volume}, timeout=3600, memory=8192,
              secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])])
def download_and_pool() -> dict:
    import os, shutil, tarfile
    from pathlib import Path
    import h5py, numpy as np
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm

    cohort_dir = Path("/vol/patches/COAD")
    manifest = cohort_dir / "_manifest.txt"
    if not manifest.exists():
        print("Downloading TCGA-COAD.tar.gz...")
        arch = Path(hf_hub_download("MahmoodLab/UNI2-h-features", "TCGA/TCGA-COAD.tar.gz",
                                    repo_type="dataset", token=os.environ["HF_TOKEN"]))
        print(f"Downloaded {arch.stat().st_size/1e9:.1f} GB; extracting...")
        cohort_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(arch, "r:gz") as tar:
            for m in tqdm([m for m in tar.getmembers() if m.name.endswith(".h5")]):
                with tar.extractfile(m) as s, open(cohort_dir / Path(m.name).name, "wb") as d:
                    shutil.copyfileobj(s, d)
        manifest.write_text("\n".join(sorted(p.stem for p in cohort_dir.glob("*.h5"))) + "\n")
        volume.commit()

    h5s = sorted(cohort_dir.glob("*.h5"))
    print(f"Mean-pooling {len(h5s)} COAD slides...")
    out = {}
    for h5 in tqdm(h5s):
        with h5py.File(h5, "r") as f:
            out[h5.stem] = f["features"][:][0].mean(axis=0).tolist()
    return out


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path
    import numpy as np

    res = download_and_pool.remote()
    pr = Path(__file__).resolve().parent.parent
    out_dir = pr / "data" / "embeddings" / "COAD"
    out_dir.mkdir(parents=True, exist_ok=True)
    sids = sorted(res.keys())
    feats = np.array([res[s] for s in sids], dtype=np.float32)
    np.save(out_dir / "mean_pooled_features.npy", feats)
    (out_dir / "mean_pool_manifest.json").write_text(json.dumps({"slide_ids": sids}, indent=2))
    print(f"Saved {feats.shape} COAD mean-pooled features to {out_dir}")
