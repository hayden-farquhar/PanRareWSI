"""
PanRareWSI — Populate Modal Volume with rare-cohort patch embeddings for ABMIL.

Downloads the 6 cohorts not already on the Volume (THYM + cptac_coad already
present) directly from HuggingFace into the Volume. Idempotent: skips cohorts
already extracted. Datacenter bandwidth makes this far faster than uploading
~32 GB from a home connection.

Usage:
    modal run --detach src/modal_populate_volume.py
"""

import modal

app = modal.App("panrarewsi-populate")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub>=0.20", "tqdm>=4.65")
)

volume = modal.Volume.from_name("panrarewsi-embeddings", create_if_missing=True)

COHORTS = {
    "ACC": "TCGA/TCGA-ACC.tar.gz",
    "UVM": "TCGA/TCGA-UVM.tar.gz",
    "MESO": "TCGA/TCGA-MESO.tar.gz",
    "CHOL": "TCGA/TCGA-CHOL.tar.gz",
    "KICH": "TCGA/TCGA-KICH.tar.gz",
    "DLBC": "TCGA/TCGA-DLBC.tar.gz",
}


@app.function(image=image, timeout=3600, memory=8192, volumes={"/vol": volume},
              secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])])
def download_cohort(cohort: str, hf_path: str) -> dict:
    import os, shutil, tarfile
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm

    cohort_dir = Path(f"/vol/patches/{cohort}")
    manifest = cohort_dir / "_manifest.txt"
    if manifest.exists():
        n = len(list(cohort_dir.glob("*.h5")))
        print(f"{cohort}: already on volume ({n} slides)")
        return {"cohort": cohort, "status": "skipped", "n": n}

    print(f"{cohort}: downloading {hf_path}...")
    arch = Path(hf_hub_download("MahmoodLab/UNI2-h-features", hf_path,
                                repo_type="dataset", token=os.environ["HF_TOKEN"]))
    print(f"{cohort}: {arch.stat().st_size/1e9:.1f} GB, extracting...")
    cohort_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(arch, "r:gz") as tar:
        for m in tqdm([m for m in tar.getmembers() if m.name.endswith(".h5")]):
            with tar.extractfile(m) as src, open(cohort_dir / Path(m.name).name, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    manifest.write_text("\n".join(sorted(p.stem for p in cohort_dir.glob("*.h5"))) + "\n")
    volume.commit()
    print(f"{cohort}: {n} slides extracted to volume")
    return {"cohort": cohort, "status": "ok", "n": n}


@app.local_entrypoint()
def main():
    # Sequential to avoid HuggingFace 429 rate-limiting. Idempotent: skips done cohorts.
    for c, p in COHORTS.items():
        try:
            r = download_cohort.remote(c, p)
            print(f"  {r['cohort']}: {r['status']} ({r['n']} slides)")
        except Exception as e:
            print(f"  {c}: FAILED {e}")
