"""
PanRareWSI — Run remaining 19 permutation cells on Modal (2 batches).
"""

import modal

app = modal.App("panrarewsi-perms-r2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.24", "scikit-learn>=1.3", "pandas>=2.0")
)

volume = modal.Volume.from_name("panrarewsi-embeddings", create_if_missing=True)

N_PERMS = 5000

MISSING = [
    "ACC/CTNNB1", "ACC/TP53", "DLBC/MSI-H", "DLBC/MYD88", "DLBC/TP53",
    "KICH/Aneuploidy-high", "KICH/Immune C3", "KICH/MSI-H", "KICH/TP53",
    "MESO/Immune C1", "THYM/GTF2I", "THYM/HRAS", "THYM/TMB-high",
    "UVM/BAP1", "UVM/Chr3 loss", "UVM/GNA11", "UVM/GNAQ", "UVM/Immune C4",
    "UVM/SF3B1",
]


def _cv_auroc(X, y, folds):
    import numpy as np
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    all_probas = np.full(len(y), np.nan)
    for fi in range(5):
        te = folds == fi
        tr = ~te
        y_tr, y_te = y[tr], y[te]
        if len(set(y_te)) < 2 or len(set(y_tr)) < 2:
            continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        n_pos = int(y_tr.sum())
        icv = min(3, n_pos, int(len(y_tr) - n_pos))
        if icv < 2:
            icv = 2
        clf = LogisticRegressionCV(
            Cs=5, cv=icv, penalty="l2", scoring="roc_auc",
            solver="lbfgs", max_iter=2000, random_state=42,
            class_weight="balanced",
        )
        clf.fit(Xtr, y_tr)
        all_probas[te] = clf.predict_proba(Xte)[:, 1]
    v = ~np.isnan(all_probas)
    if v.sum() < 10 or len(set(y[v])) < 2:
        return 0.5
    return roc_auc_score(y[v], all_probas[v])


@app.function(image=image, timeout=7200, memory=4096, volumes={"/vol": volume})
def run_batch(cells_data: list[dict], batch_id: int) -> list[dict]:
    import json
    import numpy as np
    from pathlib import Path

    results = []
    out_dir = Path("/vol/perm_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    for ci, cell in enumerate(cells_data):
        X = np.array(cell["X"])
        y = np.array(cell["y"])
        folds = np.array(cell["folds"])
        observed = cell["observed_auroc"]
        name = cell["cell_name"]
        tier = cell["tier"]

        print(f"  B{batch_id} [{ci+1}/{len(cells_data)}] {name} (n={len(y)})...")

        rng = np.random.RandomState(42)
        null_aurocs = []
        for i in range(N_PERMS):
            null_aurocs.append(_cv_auroc(X, rng.permutation(y), folds))
            if (i + 1) % 500 == 0:
                print(f"    {name}: {i+1}/{N_PERMS}")

        null_aurocs = np.array(null_aurocs)
        p = (np.sum(null_aurocs >= observed) + 1) / (N_PERMS + 1)

        result = {
            "cell": name, "tier": tier,
            "observed_auroc": observed, "p_value": float(p),
            "null_mean": float(np.mean(null_aurocs)),
            "null_std": float(np.std(null_aurocs)),
            "n_perms": N_PERMS, "n": len(y),
        }
        results.append(result)
        print(f"    {name}: p={p:.5f}")

        safe = name.replace("/", "_").replace(" ", "_")
        (out_dir / f"{safe}.json").write_text(json.dumps(result, indent=2))
        volume.commit()

    print(f"  B{batch_id} complete: {len(results)} cells")
    return results


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path
    import math
    import numpy as np
    import pandas as pd

    project_root = Path(__file__).resolve().parent.parent
    from src.phase4_benchmark import (
        BIOMARKER_CELLS, COHORTS, load_cohort_data, parse_patient_id, _get_binary_labels,
    )

    phase4 = json.loads((project_root / "results" / "phase4_benchmark.json").read_text())
    ok_cells = {r["cell"]: r for r in phase4 if r["status"] == "ok"}

    all_cells = []
    for cohort in COHORTS:
        try:
            features, slide_ids, labels, splits = load_cohort_data(cohort, project_root)
        except Exception as e:
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
            if cell_name not in ok_cells or cell_name not in MISSING:
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
            all_cells.append({
                "X": X, "y": y, "folds": folds,
                "observed_auroc": observed,
                "cell_name": cell_name,
                "tier": cell["tier"],
            })

    all_cells.sort(key=lambda c: len(c["y"]))
    n_batches = 2
    batch_size = math.ceil(len(all_cells) / n_batches)
    batches = [all_cells[i:i + batch_size] for i in range(0, len(all_cells), batch_size)]

    print(f"Running {len(all_cells)} remaining cells in {len(batches)} batches...")
    for i, b in enumerate(batches):
        print(f"  B{i}: {[c['cell_name'] for c in b]}")

    handles = [run_batch.spawn(batch, i) for i, batch in enumerate(batches)]
    all_results = []
    for i, h in enumerate(handles):
        try:
            all_results.extend(h.get())
            print(f"  B{i}: collected")
        except Exception as e:
            print(f"  B{i} handle failed: {e}")

    # Merge with existing results from Volume
    existing = []
    perm_dir = project_root / "results" / "perm_cells" / "perm_results"
    for f in perm_dir.glob("*.json"):
        existing.append(json.loads(f.read_text()))

    merged = {r["cell"]: r for r in existing}
    for r in all_results:
        merged[r["cell"]] = r

    final = list(merged.values())

    # BH-FDR
    primary = [r for r in final if r["tier"] == "primary"]
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
    for r in final:
        if r["tier"] != "primary":
            r["bh_significant"] = None

    out_path = project_root / "results" / "permutation_tests.json"
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)

    n_sig = sum(1 for r in primary if r["bh_significant"])
    print(f"\n{'='*60}")
    print(f"PERMUTATION TESTS ({N_PERMS} perms, BH-FDR α=0.05)")
    print(f"{'='*60}")
    print(f"Total cells: {len(final)}")
    print(f"Primary significant: {n_sig}/{len(primary)}")
    for r in sorted(final, key=lambda x: x["p_value"]):
        sig = "***" if r.get("bh_significant") else "   "
        print(f"  {sig} {r['cell']:30s} p={r['p_value']:.5f} AUROC={r['observed_auroc']:.3f} ({r['tier']})")
    print(f"\nSaved: {out_path}")
