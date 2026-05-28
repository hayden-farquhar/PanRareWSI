"""
PanRareWSI — Phase 3 baseline replication on CPTAC COAD MSI_H.

Downloads CPTAC COAD UNI2-h embeddings via Modal, mean-pools slide features,
and runs L2-regularised linear probe across Patho-Bench's 50 train/test splits.

Pre-registration §9: AUROC within ±0.03 of published value, or within
literature range if no exact UNI2-h number available.

Usage:
    modal run src/phase3_baseline.py
"""

import modal

app = modal.App("panrarewsi-phase3")

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
    timeout=3600,
    memory=16384,
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
def run_baseline() -> dict:
    import json
    import os
    import shutil
    import tarfile
    from pathlib import Path

    import h5py
    import numpy as np
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    from sklearn.preprocessing import StandardScaler
    from tqdm import tqdm

    cohort_dir = Path("/embeddings/cptac_coad")
    manifest_file = cohort_dir / "_manifest.txt"

    # Step 1: Download and extract CPTAC COAD embeddings
    if manifest_file.exists():
        print(f"CPTAC COAD already extracted ({len(list(cohort_dir.glob('*.h5')))} slides)")
    else:
        print("Downloading CPTAC/cptac_coad.tar.gz...")
        archive_path = hf_hub_download(
            "MahmoodLab/UNI2-h-features",
            "CPTAC/cptac_coad.tar.gz",
            repo_type="dataset",
            token=os.environ["HF_TOKEN"],
        )
        archive_path = Path(archive_path)
        print(f"Downloaded: {archive_path.stat().st_size / 1e9:.1f} GB")

        cohort_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        with tarfile.open(archive_path, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(".h5")]
            for member in tqdm(members, desc="Extract", unit="slide"):
                target = cohort_dir / Path(member.name).name
                with tar.extractfile(member) as src:
                    with open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                n += 1
        slide_ids = sorted(p.stem for p in cohort_dir.glob("*.h5"))
        manifest_file.write_text("\n".join(slide_ids) + "\n")
        print(f"Extracted {n} slides")
        volume.commit()

    # Step 2: Mean-pool all slides
    print("Mean-pooling slide features...")
    h5_files = sorted(cohort_dir.glob("*.h5"))
    slide_features = {}
    for h5_path in tqdm(h5_files, desc="Pool", unit="slide"):
        with h5py.File(h5_path, "r") as f:
            features = f["features"][:]  # (1, n_patches, 1536)
        slide_features[h5_path.stem] = features[0].mean(axis=0)
    print(f"Pooled {len(slide_features)} slides")

    # Step 3: Load Patho-Bench splits
    splits_path = hf_hub_download(
        "MahmoodLab/Patho-Bench",
        "cptac_coad/MSI_H/k=all.tsv",
        repo_type="dataset",
    )
    df = pd.read_csv(splits_path, sep="\t")

    # Match slide_id to embeddings (slide_id in splits is truncated)
    slide_id_map = {}
    for full_id in slide_features:
        for _, row in df.iterrows():
            if full_id.startswith(row["slide_id"]):
                slide_id_map[row["slide_id"]] = full_id
                break

    df["full_slide_id"] = df["slide_id"].map(slide_id_map)
    matched = df.dropna(subset=["full_slide_id"])
    print(f"Matched {len(matched)}/{len(df)} slides to embeddings")

    # Step 4: Run linear probe across 50 folds
    fold_cols = [c for c in df.columns if c.startswith("fold_")]
    results_per_fold = []

    for fold_col in tqdm(fold_cols, desc="Folds", unit="fold"):
        train_mask = matched[fold_col] == "train"
        test_mask = matched[fold_col] == "test"

        X_train = np.array([slide_features[sid] for sid in matched.loc[train_mask, "full_slide_id"]])
        y_train = matched.loc[train_mask, "MSI_H"].values.astype(float)
        X_test = np.array([slide_features[sid] for sid in matched.loc[test_mask, "full_slide_id"]])
        y_test = matched.loc[test_mask, "MSI_H"].values.astype(float)

        if len(np.unique(y_test)) < 2 or len(np.unique(y_train)) < 2:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        n_pos = int(y_train.sum())
        inner_cv = min(3, n_pos, int(len(y_train) - n_pos))
        if inner_cv < 2:
            inner_cv = 2

        clf = LogisticRegressionCV(
            Cs=10, cv=inner_cv, penalty="l2", scoring="roc_auc",
            solver="lbfgs", max_iter=5000, random_state=42,
            class_weight="balanced",
        )
        clf.fit(X_train_s, y_train)
        probas = clf.predict_proba(X_test_s)[:, 1]

        auroc = roc_auc_score(y_test, probas)
        auprc = average_precision_score(y_test, probas)
        brier = brier_score_loss(y_test, probas)

        results_per_fold.append({
            "fold": fold_col,
            "auroc": auroc,
            "auprc": auprc,
            "brier": brier,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "n_pos_test": int(y_test.sum()),
            "best_C": float(clf.C_[0]),
        })

    # Step 5: Aggregate results
    aurocs = [r["auroc"] for r in results_per_fold]
    auprcs = [r["auprc"] for r in results_per_fold]
    briers = [r["brier"] for r in results_per_fold]

    summary = {
        "task": "CPTAC COAD MSI_H",
        "model": "UNI2-h mean-pool + L2 logistic regression",
        "n_folds_run": len(results_per_fold),
        "n_folds_total": len(fold_cols),
        "mean_auroc": float(np.mean(aurocs)),
        "std_auroc": float(np.std(aurocs)),
        "median_auroc": float(np.median(aurocs)),
        "mean_auprc": float(np.mean(auprcs)),
        "mean_brier": float(np.mean(briers)),
        "per_fold": results_per_fold,
    }

    print(f"\n{'='*50}")
    print(f"PHASE 3 BASELINE RESULTS")
    print(f"{'='*50}")
    print(f"Task: CPTAC COAD MSI_H (Patho-Bench)")
    print(f"Model: UNI2-h mean-pool → L2 LogReg")
    print(f"Folds: {len(results_per_fold)}/{len(fold_cols)} completed")
    print(f"AUROC: {np.mean(aurocs):.3f} ± {np.std(aurocs):.3f} (median {np.median(aurocs):.3f})")
    print(f"AUPRC: {np.mean(auprcs):.3f} ± {np.std(auprcs):.3f}")
    print(f"Brier: {np.mean(briers):.3f} ± {np.std(briers):.3f}")
    print(f"Literature range for COAD MSI (FMs): 0.73–0.93")

    return summary


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    print("Launching Phase 3 baseline on Modal...")
    summary = run_baseline.remote()

    project_root = Path(__file__).resolve().parent.parent
    out_path = project_root / "results" / "phase3_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"Mean AUROC: {summary['mean_auroc']:.3f} ± {summary['std_auroc']:.3f}")
