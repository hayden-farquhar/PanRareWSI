"""
PanRareWSI — Run remaining permutation cells on Modal in DETACH mode.

Each cell is an independent function call (1 cell per worker). In detach mode
(`modal run --detach`), the app persists server-side even after the local
client exits, so the ~30-min app-runtime limit on the foreground client
does not kill the workers. Each cell saves to the Volume on completion.

Run with:
    modal run --detach src/modal_perm_detach.py

Then poll the Volume:
    modal volume ls panrarewsi-embeddings perm_results
"""

import modal

app = modal.App("panrarewsi-perms-detach")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.24", "scikit-learn>=1.3", "pandas>=2.0")
)

volume = modal.Volume.from_name("panrarewsi-embeddings", create_if_missing=True)

N_PERMS = 2000

ALL_CELLS = [
    "ACC/TP53", "ACC/CTNNB1", "ACC/TMB-high", "ACC/FGA-high", "ACC/Immune C4",
    "UVM/GNAQ", "UVM/GNA11", "UVM/SF3B1", "UVM/BAP1", "UVM/EIF1AX",
    "UVM/Chr3 loss", "UVM/Immune C4",
    "MESO/Histology epi", "MESO/NF2", "MESO/BAP1", "MESO/TP53",
    "MESO/CDKN2A homdel", "MESO/Immune C1",
    "CHOL/IDH1", "CHOL/Immune C3",
    "THYM/GTF2I", "THYM/TMB-high", "THYM/HRAS",
    "KICH/TP53", "KICH/Aneuploidy-high", "KICH/Immune C3", "KICH/MSI-H",
    "DLBC/MSI-H", "DLBC/TMB-high", "DLBC/Hans non-GCB", "DLBC/MYD88",
    "DLBC/Aneuploidy-high", "DLBC/TP53",
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


@app.function(image=image, timeout=3600, memory=4096, volumes={"/vol": volume})
def run_one_cell(cell: dict) -> dict:
    """Run permutation test for a single cell. Saves to volume on completion."""
    import json
    import numpy as np
    from pathlib import Path

    name = cell["cell_name"]
    safe = name.replace("/", "_").replace(" ", "_")
    out_path = Path(f"/vol/perm_results/{safe}.json")

    # Skip if already done
    if out_path.exists():
        print(f"  {name}: already done, skipping")
        return json.loads(out_path.read_text())

    X = np.array(cell["X"])
    y = np.array(cell["y"])
    folds = np.array(cell["folds"])
    observed = cell["observed_auroc"]
    tier = cell["tier"]

    print(f"  {name} (n={len(y)}): starting {N_PERMS} perms...")
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    volume.commit()
    print(f"  {name}: DONE p={p:.5f}")
    return result


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path
    import numpy as np
    import pandas as pd

    project_root = Path(__file__).resolve().parent.parent
    from src.phase4_benchmark import (
        BIOMARKER_CELLS, COHORTS, load_cohort_data, parse_patient_id, _get_binary_labels,
    )

    phase4 = json.loads((project_root / "results" / "phase4_benchmark.json").read_text())
    ok_cells = {r["cell"]: r for r in phase4 if r["status"] == "ok"}

    # Build all cell inputs
    cell_data = {}
    for cohort in COHORTS:
        try:
            features, slide_ids, labels, splits = load_cohort_data(cohort, project_root)
        except Exception:
            continue
        slide_patient = {sid: parse_patient_id(sid) for sid in slide_ids}
        patient_features = {}
        for i, sid in enumerate(slide_ids):
            pid = slide_patient[sid]
            patient_features.setdefault(pid, []).append(features[i])
        patient_mean = {pid: np.mean(f, axis=0) for pid, f in patient_features.items()}
        split_map = dict(zip(splits["patient_id"], splits["fold"]))

        for cell in BIOMARKER_CELLS.get(cohort, []):
            cell_name = f"{cohort}/{cell['name']}"
            if cell_name not in ok_cells:
                continue
            observed = ok_cells[cell_name]["pooled_auroc"]
            y_series = _get_binary_labels(labels, cell)
            label_map = dict(zip(labels["patient_id"], y_series))
            X, y, folds = [], [], []
            for pid, feat in patient_mean.items():
                if pid in label_map and pid in split_map and pd.notna(label_map[pid]):
                    X.append(feat.tolist())
                    y.append(float(label_map[pid]))
                    folds.append(int(split_map[pid]))
            cell_data[cell_name] = {
                "X": X, "y": y, "folds": folds,
                "observed_auroc": observed,
                "cell_name": cell_name,
                "tier": cell["tier"],
            }

    inputs = [cell_data[name] for name in ALL_CELLS if name in cell_data]
    print(f"Dispatching {len(inputs)} cells as individual workers (detach-safe, {N_PERMS} perms)...")
    print("Each cell skips if already saved to volume. Workers persist in detach mode.")

    # .map() with a generous timeout; workers save to volume regardless
    results = list(run_one_cell.map(inputs))
    print(f"Returned {len(results)} results (also saved to volume)")
