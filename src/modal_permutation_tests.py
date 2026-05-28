"""
PanRareWSI — 10K permutation tests on Modal (parallelised across cells).

Each cell runs as an independent Modal function call, enabling parallel execution.
Returns p-values for all 33 cells, with BH-FDR correction applied locally.

Usage:
    modal run src/modal_permutation_tests.py
"""

import modal

app = modal.App("panrarewsi-permutations")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.24", "scikit-learn>=1.3", "pandas>=2.0", "tqdm>=4.65",
    )
)


volume = modal.Volume.from_name("panrarewsi-embeddings", create_if_missing=True)


@app.function(image=image, timeout=7200, memory=4096, volumes={"/results": volume})
def run_cell_permutation(
    X: list, y: list, folds: list,
    observed_auroc: float, cell_name: str, tier: str,
    n_perms: int = 5000, seed: int = 42,
) -> dict:
    import numpy as np
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    X = np.array(X)
    y = np.array(y)
    folds = np.array(folds)
    rng = np.random.RandomState(seed)

    def cv_auroc(X_data, y_data, fold_data):
        all_probas = np.full(len(y_data), np.nan)
        for fold_idx in range(5):
            test_mask = fold_data == fold_idx
            train_mask = ~test_mask
            y_tr, y_te = y_data[train_mask], y_data[test_mask]
            if len(np.unique(y_te)) < 2 or len(np.unique(y_tr)) < 2:
                continue
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_data[train_mask])
            X_te = scaler.transform(X_data[test_mask])
            n_pos = int(y_tr.sum())
            inner_cv = min(3, n_pos, int(len(y_tr) - n_pos))
            if inner_cv < 2:
                inner_cv = 2
            clf = LogisticRegressionCV(
                Cs=5, cv=inner_cv, penalty="l2", scoring="roc_auc",
                solver="lbfgs", max_iter=2000, random_state=42,
                class_weight="balanced",
            )
            clf.fit(X_tr, y_tr)
            all_probas[test_mask] = clf.predict_proba(X_te)[:, 1]
        valid = ~np.isnan(all_probas)
        if valid.sum() < 10 or len(np.unique(y_data[valid])) < 2:
            return 0.5
        return roc_auc_score(y_data[valid], all_probas[valid])

    null_aurocs = []
    for i in range(n_perms):
        y_perm = rng.permutation(y)
        null_aurocs.append(cv_auroc(X, y_perm, folds))
        if (i + 1) % 1000 == 0:
            print(f"  {cell_name}: {i+1}/{n_perms} perms done")

    null_aurocs = np.array(null_aurocs)
    p_value = (np.sum(null_aurocs >= observed_auroc) + 1) / (n_perms + 1)

    result = {
        "cell": cell_name,
        "tier": tier,
        "observed_auroc": observed_auroc,
        "p_value": float(p_value),
        "null_mean": float(np.mean(null_aurocs)),
        "null_std": float(np.std(null_aurocs)),
        "n_perms": n_perms,
        "n": len(y),
    }

    # Persist to volume so results survive if the local client disconnects
    import json
    from pathlib import Path
    out_dir = Path("/results/permutation_cells")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = cell_name.replace("/", "_").replace(" ", "_")
    (out_dir / f"{safe_name}.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    print(f"  {cell_name}: DONE p={p_value:.5f}")

    return result


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    project_root = Path(__file__).resolve().parent.parent

    # Import locally to prepare data
    from src.phase4_benchmark import (
        BIOMARKER_CELLS, COHORTS, load_cohort_data, parse_patient_id, _get_binary_labels,
    )

    phase4 = json.loads((project_root / "results" / "phase4_benchmark.json").read_text())
    ok_cells = {r["cell"]: r for r in phase4 if r["status"] == "ok"}

    # Prepare all cell data for parallel dispatch
    cell_inputs = []

    for cohort in COHORTS:
        try:
            features, slide_ids, labels, splits = load_cohort_data(cohort, project_root)
        except Exception as e:
            print(f"{cohort}: {e}")
            continue

        slide_patient = {sid: parse_patient_id(sid) for sid in slide_ids}
        patient_features = {}
        for i, sid in enumerate(slide_ids):
            pid = slide_patient[sid]
            if pid not in patient_features:
                patient_features[pid] = []
            patient_features[pid].append(features[i])
        patient_mean = {pid: np.mean(f, axis=0) for pid, f in patient_features.items()}

        split_map = dict(zip(splits["patient_id"], splits["fold"]))

        for cell in BIOMARKER_CELLS.get(cohort, []):
            cell_name = f"{cohort}/{cell['name']}"
            if cell_name not in ok_cells:
                continue

            observed = ok_cells[cell_name]["pooled_auroc"]
            y_series = _get_binary_labels(labels, cell)
            label_map = dict(zip(labels["patient_id"], y_series))

            pids, X, y, folds = [], [], [], []
            for pid, feat in patient_mean.items():
                if pid in label_map and pid in split_map and pd.notna(label_map[pid]):
                    pids.append(pid)
                    X.append(feat.tolist())
                    y.append(float(label_map[pid]))
                    folds.append(int(split_map[pid]))

            cell_inputs.append({
                "X": X, "y": y, "folds": folds,
                "observed_auroc": observed,
                "cell_name": cell_name,
                "tier": cell["tier"],
            })

    n_perms = 5000
    print(f"Running {len(cell_inputs)} cells on Modal ({n_perms} perms each, sequential)...")

    # Sequential dispatch to avoid concurrency limits
    results = []
    for i, ci in enumerate(cell_inputs):
        cell_name = ci["cell_name"]
        print(f"  [{i+1}/{len(cell_inputs)}] {cell_name}...")
        try:
            result = run_cell_permutation.remote(
                ci["X"], ci["y"], ci["folds"],
                ci["observed_auroc"], ci["cell_name"], ci["tier"],
                n_perms=n_perms,
            )
            results.append(result)
            print(f"    p={result['p_value']:.5f}")
        except Exception as e:
            print(f"    FAILED: {e}")
    print(f"Completed {len(results)}/{len(cell_inputs)} cells")

    # BH-FDR on primary cells
    primary = [r for r in results if r["tier"] == "primary"]
    pvals = [r["p_value"] for r in primary]
    n = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    significant = [False] * n
    for rank, (orig_idx, p) in enumerate(indexed, 1):
        if p <= (rank / n) * 0.05:
            significant[orig_idx] = True
        else:
            break
    for r, sig in zip(primary, significant):
        r["bh_significant"] = sig
    for r in results:
        if r["tier"] != "primary":
            r["bh_significant"] = None

    # Save
    out_path = project_root / "results" / "permutation_tests.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    n_sig = sum(1 for r in primary if r["bh_significant"])
    print(f"\n{'='*60}")
    print(f"PERMUTATION TEST RESULTS (10K perms, BH-FDR α=0.05)")
    print(f"{'='*60}")
    print(f"Primary cells: {n_sig}/{len(primary)} significant")
    for r in sorted(results, key=lambda x: x["p_value"]):
        sig = "***" if r.get("bh_significant") else "   "
        print(f"  {sig} {r['cell']:30s} p={r['p_value']:.5f} AUROC={r['observed_auroc']:.3f} ({r['tier']})")
    print(f"\nSaved: {out_path}")
