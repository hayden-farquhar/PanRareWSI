"""
PanRareWSI — Phase 5: Cross-cohort zero-shot transfer.

Trains TP53 classifier on CPTAC COAD (common cohort) and tests on rare cohorts.
Pre-registration RQ3.

Usage:
    modal run src/phase5_transfer.py
"""

import modal

app = modal.App("panrarewsi-phase5")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub>=0.20", "h5py>=3.9", "numpy>=1.24", "tqdm>=4.65",
        "scikit-learn>=1.3", "pandas>=2.0",
    )
)

volume = modal.Volume.from_name("panrarewsi-embeddings", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/embeddings": volume},
    timeout=1800,
    memory=8192,
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
def run_transfer() -> dict:
    import json
    import os
    from pathlib import Path

    import h5py
    import numpy as np
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from tqdm import tqdm

    # Step 1: Load CPTAC COAD mean-pooled features (already on volume from Phase 3)
    coad_dir = Path("/embeddings/cptac_coad")
    h5_files = sorted(coad_dir.glob("*.h5"))
    print(f"CPTAC COAD: {len(h5_files)} slides on volume")

    coad_features = {}
    for h5_path in tqdm(h5_files, desc="Pool COAD", unit="slide"):
        with h5py.File(h5_path, "r") as f:
            feats = f["features"][:]
        coad_features[h5_path.stem] = feats[0].mean(axis=0)

    # Step 2: Load CPTAC COAD TP53 labels
    tp53_path = hf_hub_download(
        "MahmoodLab/Patho-Bench", "cptac_coad/TP53_mutation/k=all.tsv", repo_type="dataset"
    )
    tp53_df = pd.read_csv(tp53_path, sep="\t")

    # Match
    slide_map = {}
    for full_id in coad_features:
        for _, row in tp53_df.iterrows():
            if full_id.startswith(row["slide_id"]):
                slide_map[row["slide_id"]] = full_id
                break
    tp53_df["full_id"] = tp53_df["slide_id"].map(slide_map)
    matched = tp53_df.dropna(subset=["full_id"])
    print(f"Matched {len(matched)}/{len(tp53_df)} TP53 slides")

    X_source = np.array([coad_features[sid] for sid in matched["full_id"]])
    y_source = matched["TP53_mutation"].values.astype(float)

    # Train on ALL CPTAC COAD TP53
    scaler = StandardScaler()
    X_source_s = scaler.fit_transform(X_source)

    clf = LogisticRegressionCV(
        Cs=10, cv=3, penalty="l2", scoring="roc_auc",
        solver="lbfgs", max_iter=5000, random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_source_s, y_source)
    print(f"Source model trained: CPTAC COAD TP53 (n={len(y_source)}, prev={y_source.mean():.2f})")

    # Step 3: Zero-shot transfer to rare cohorts with TP53
    rare_tp53_cohorts = {
        "ACC": {"col": "mut_TP53"},
        "MESO": {"col": "mut_TP53"},
        "KICH": {"col": "mut_TP53"},
        "DLBC": {"col": "mut_TP53"},
    }

    # Load rare-cohort features from volume (THYM) or local data won't be here
    # We need to send features from local to Modal. Alternative: compute on volume.
    # THYM is on volume. Others are not. Let's handle what we have.

    thym_dir = Path("/embeddings/THYM")
    if thym_dir.exists():
        thym_h5 = sorted(thym_dir.glob("*.h5"))
        if thym_h5:
            thym_features = {}
            for h5_path in tqdm(thym_h5, desc="Pool THYM", unit="slide"):
                with h5py.File(h5_path, "r") as f:
                    feats = f["features"][:]
                thym_features[h5_path.stem] = feats[0].mean(axis=0)
            print(f"THYM: {len(thym_features)} slides pooled")

    # For this function, return the trained model parameters so local code can apply them
    return {
        "source_task": "CPTAC COAD TP53",
        "source_n": len(y_source),
        "source_prevalence": float(y_source.mean()),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "best_C": float(clf.C_[0]),
    }


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    print("Training source model on Modal...")
    model_params = run_transfer.remote()

    project_root = Path(__file__).resolve().parent.parent

    # Apply to local rare cohorts
    coef = np.array(model_params["coef"])
    intercept = model_params["intercept"]
    scaler_mean = np.array(model_params["scaler_mean"])
    scaler_scale = np.array(model_params["scaler_scale"])

    def predict_proba(X):
        X_s = (X - scaler_mean) / scaler_scale
        logits = X_s @ coef + intercept
        probs = 1 / (1 + np.exp(-logits))
        return probs

    results = []
    for cohort in ["ACC", "MESO", "KICH", "DLBC", "THYM"]:
        embed_dir = project_root / "data" / "embeddings" / cohort
        features = np.load(embed_dir / "mean_pooled_features.npy")
        manifest = json.loads((embed_dir / "mean_pool_manifest.json").read_text())
        slide_ids = manifest["slide_ids"]

        labels = pd.read_parquet(project_root / "data" / "labels" / f"{cohort.lower()}_labels.parquet")

        # Parse patient IDs and aggregate
        patient_features = {}
        for i, sid in enumerate(slide_ids):
            parts = sid.split(".")
            barcode = parts[0]
            pid = "-".join(barcode.split("-")[:3])
            if pid not in patient_features:
                patient_features[pid] = []
            patient_features[pid].append(features[i])
        patient_mean = {pid: np.mean(f, axis=0) for pid, f in patient_features.items()}

        label_map = dict(zip(labels["patient_id"], labels["mut_TP53"]))

        pids, X, y = [], [], []
        for pid, feat in patient_mean.items():
            if pid in label_map and pd.notna(label_map[pid]):
                pids.append(pid)
                X.append(feat)
                y.append(float(label_map[pid]))

        if not pids:
            continue

        X, y = np.array(X), np.array(y)
        probas = predict_proba(X)

        if len(np.unique(y)) < 2:
            auroc = float("nan")
        else:
            auroc = roc_auc_score(y, probas)

        result = {
            "target_cohort": cohort,
            "biomarker": "TP53",
            "transfer_auroc": round(auroc, 4),
            "n": len(y),
            "n_pos": int(y.sum()),
            "prevalence": round(y.mean(), 3),
        }
        results.append(result)
        print(f"  {cohort} TP53: AUROC={auroc:.3f} (n={len(y)}, prev={y.mean():.2f})")

    # Compare with within-cohort results
    phase4 = json.loads((project_root / "results" / "phase4_benchmark.json").read_text())
    within_map = {}
    for r in phase4:
        if r.get("biomarker") == "TP53" and r["status"] == "ok":
            within_map[r["cohort"]] = r["pooled_auroc"]

    print("\n=== Transfer vs Within-Cohort ===")
    for r in results:
        within = within_map.get(r["target_cohort"], "N/A")
        diff = r["transfer_auroc"] - within if isinstance(within, float) else "N/A"
        print(f"  {r['target_cohort']:6s}: transfer={r['transfer_auroc']:.3f}  within={within}  diff={diff}")
        r["within_auroc"] = within
        r["delta"] = round(diff, 4) if isinstance(diff, float) else None

    # Save
    out = {"source": model_params, "transfer_results": results}
    out_path = project_root / "results" / "phase5_transfer.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")
