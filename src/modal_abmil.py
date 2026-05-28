"""
PanRareWSI — ABMIL secondary model (pre-registration §6.2) on Modal.

Gated-attention multiple-instance learning (Ilse et al. 2018) on patch-level
UNI2-h embeddings. Applied to all primary + exploratory cells, 5-fold patient-
level CV, class-weighted BCE, early stopping on a validation subset. Reports
pooled out-of-fold AUROC with 1000x bootstrap CIs. Idempotent per cell
(saves to Volume), so re-runs skip completed cells (Modal app-runtime limit).

Patches must be on the Volume under /vol/patches/{COHORT}/ and /vol/THYM/.

Usage:
    modal run --detach src/modal_abmil.py
"""

import modal

app = modal.App("panrarewsi-abmil")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.1", "h5py>=3.9", "numpy>=1.24",
                 "scikit-learn>=1.3", "pandas>=2.0", "pyarrow>=14.0")
)

volume = modal.Volume.from_name("panrarewsi-embeddings", create_if_missing=True)

# Cell definitions mirror phase4_benchmark BIOMARKER_CELLS (binary cells only)
CELL_SPECS = {
    "ACC": [("TP53", "mut_TP53", None, "primary"), ("CTNNB1", "mut_CTNNB1", None, "primary"),
            ("TMB-high", "label_tmb_high", None, "primary"), ("FGA-high", "label_fga_high", None, "primary"),
            ("Immune C4", "immune_subtype", "C4", "primary")],
    "UVM": [("GNAQ", "mut_GNAQ", None, "primary"), ("GNA11", "mut_GNA11", None, "primary"),
            ("SF3B1", "mut_SF3B1", None, "primary"), ("BAP1", "mut_BAP1", None, "primary"),
            ("EIF1AX", "mut_EIF1AX", None, "primary"), ("Chr3 loss", "label_chr3_loss", None, "primary"),
            ("Immune C4", "immune_subtype", "C4", "primary")],
    "MESO": [("Histology epi", "label_histology_epi", None, "primary"), ("NF2", "mut_NF2", None, "primary"),
             ("BAP1", "mut_BAP1", None, "primary"), ("TP53", "mut_TP53", None, "primary"),
             ("CDKN2A homdel", "label_cdkn2a_homdel", None, "primary"), ("Immune C1", "immune_subtype", "C1", "primary")],
    "CHOL": [("IDH1", "mut_IDH1", None, "exploratory"), ("Immune C3", "immune_subtype", "C3", "exploratory")],
    "THYM": [("GTF2I", "mut_GTF2I", None, "primary"), ("TMB-high", "label_tmb_high", None, "primary"),
             ("HRAS", "mut_HRAS", None, "exploratory")],
    "KICH": [("TP53", "mut_TP53", None, "primary"), ("Aneuploidy-high", "label_aneuploidy_high", None, "primary"),
             ("Immune C3", "immune_subtype", "C3", "primary"), ("MSI-H", "label_msi_h", None, "exploratory")],
    "DLBC": [("MSI-H", "label_msi_h", None, "primary"), ("TMB-high", "label_tmb_high", None, "primary"),
             ("Hans non-GCB", "label_hans_nongcb", None, "primary"), ("MYD88", "mut_MYD88", None, "primary"),
             ("Aneuploidy-high", "label_aneuploidy_high", None, "primary"), ("TP53", "mut_TP53", None, "exploratory")],
}


def patch_dir_for(cohort):
    # THYM patches live at /vol/THYM; others at /vol/patches/{cohort}
    return f"/vol/THYM" if cohort == "THYM" else f"/vol/patches/{cohort}"


@app.function(image=image, timeout=3600, memory=16384, gpu="T4", volumes={"/vol": volume},
              secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])])
def run_cohort_abmil(cohort: str, labels_parquet_bytes: bytes, splits_csv_bytes: bytes) -> list:
    import io, json
    from pathlib import Path
    import h5py
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    labels = pd.read_parquet(io.BytesIO(labels_parquet_bytes))
    splits = pd.read_csv(io.BytesIO(splits_csv_bytes))

    pdir = Path(patch_dir_for(cohort))
    h5_files = sorted(pdir.glob("*.h5"))

    def parse_pid(stem):
        return "-".join(stem.split(".")[0].split("-")[:3])

    # Load patches per patient (concatenate slides)
    patient_patches = {}
    for h5 in h5_files:
        pid = parse_pid(h5.stem)
        with h5py.File(h5, "r") as f:
            feats = f["features"][:][0]  # (n_patches, 1536)
        patient_patches.setdefault(pid, []).append(feats)
    patient_bags = {pid: np.concatenate(v, axis=0) for pid, v in patient_patches.items()}

    class GatedAttentionMIL(nn.Module):
        def __init__(self, in_dim=1536, hidden=256, att=128):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.25))
            self.att_V = nn.Linear(hidden, att)
            self.att_U = nn.Linear(hidden, att)
            self.att_w = nn.Linear(att, 1)
            self.classifier = nn.Linear(hidden, 1)

        def forward(self, x):  # x: (N, in_dim)
            h = self.fc(x)
            a = self.att_w(torch.tanh(self.att_V(h)) * torch.sigmoid(self.att_U(h)))  # (N,1)
            a = torch.softmax(a, dim=0)
            z = (a * h).sum(dim=0, keepdim=True)  # (1, hidden)
            return self.classifier(z).squeeze(), a

    smap = dict(zip(splits["patient_id"], splits["fold"]))
    out_dir = Path("/vol/abmil_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name, col, binar, tier in CELL_SPECS.get(cohort, []):
        cell_name = f"{cohort}/{name}"
        safe = cell_name.replace("/", "_").replace(" ", "_")
        out_path = out_dir / f"{safe}.json"
        if out_path.exists():
            results.append(json.loads(out_path.read_text()))
            print(f"  {cell_name}: cached")
            continue

        # Build labels
        if binar:
            lab = labels[col].apply(lambda x: 1.0 if x == binar else (0.0 if pd.notna(x) else np.nan))
        else:
            lab = labels[col].astype(float)
        lmap = dict(zip(labels["patient_id"], lab))

        pids, bags, y, folds = [], [], [], []
        for pid, bag in patient_bags.items():
            if pid in lmap and pid in smap and pd.notna(lmap[pid]):
                pids.append(pid); bags.append(bag); y.append(lmap[pid]); folds.append(smap[pid])
        y = np.array(y); folds = np.array(folds)
        if len(y) < 10 or len(np.unique(y)) < 2 or int(y.sum()) < 5 or int((1-y).sum()) < 5:
            res = {"cell": cell_name, "tier": tier, "status": "insufficient_n", "n": len(y),
                   "n_pos": int(y.sum()) if len(y) else 0}
            out_path.write_text(json.dumps(res)); volume.commit()
            results.append(res); print(f"  {cell_name}: insufficient_n")
            continue

        oof = np.full(len(y), np.nan)
        for fi in range(5):
            te = folds == fi; tr = ~te
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            tr_idx = np.where(tr)[0]
            # inner validation split for early stopping (20%)
            rng = np.random.RandomState(42)
            rng.shuffle(tr_idx)
            n_val = max(2, int(0.2 * len(tr_idx)))
            val_idx, fit_idx = tr_idx[:n_val], tr_idx[n_val:]
            if len(np.unique(y[fit_idx])) < 2:
                fit_idx = tr_idx; val_idx = tr_idx[:n_val]

            pos_w = float((y[fit_idx] == 0).sum()) / max(1, (y[fit_idx] == 1).sum())
            model = GatedAttentionMIL().to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
            lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=device))

            best_val, best_state, patience, wait = -1, None, 15, 0
            for epoch in range(100):
                model.train()
                perm = rng.permutation(fit_idx)
                for idx in perm:
                    x = torch.tensor(bags[idx], dtype=torch.float32, device=device)
                    target = torch.tensor(y[idx], dtype=torch.float32, device=device)
                    opt.zero_grad()
                    logit, _ = model(x)
                    loss = lossf(logit.unsqueeze(0), target.unsqueeze(0))
                    loss.backward(); opt.step()
                # validation AUROC
                model.eval()
                with torch.no_grad():
                    vp = [torch.sigmoid(model(torch.tensor(bags[i], dtype=torch.float32, device=device))[0]).item() for i in val_idx]
                if len(np.unique(y[val_idx])) > 1:
                    va = roc_auc_score(y[val_idx], vp)
                    if va > best_val:
                        best_val, best_state, wait = va, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
                    else:
                        wait += 1
                        if wait >= patience:
                            break
            if best_state:
                model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                for i in np.where(te)[0]:
                    oof[i] = torch.sigmoid(model(torch.tensor(bags[i], dtype=torch.float32, device=device))[0]).item()

        valid = ~np.isnan(oof)
        if valid.sum() < 10 or len(np.unique(y[valid])) < 2:
            res = {"cell": cell_name, "tier": tier, "status": "failed", "n": len(y)}
        else:
            auroc = roc_auc_score(y[valid], oof[valid])
            # bootstrap CI
            rng = np.random.RandomState(42)
            boots = []
            yv, ov = y[valid], oof[valid]
            for _ in range(1000):
                bi = rng.randint(0, len(yv), len(yv))
                if len(np.unique(yv[bi])) < 2:
                    continue
                boots.append(roc_auc_score(yv[bi], ov[bi]))
            res = {"cell": cell_name, "tier": tier, "status": "ok", "n": len(y),
                   "n_pos": int(y.sum()), "abmil_auroc": round(float(auroc), 4),
                   "abmil_ci": [round(float(np.percentile(boots, 2.5)), 4),
                                round(float(np.percentile(boots, 97.5)), 4)],
                   "device": device}
            print(f"  {cell_name}: ABMIL AUROC={auroc:.3f}")
        out_path.write_text(json.dumps(res, indent=2)); volume.commit()
        results.append(res)

    return results


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    cohorts = ["ACC", "UVM", "MESO", "CHOL", "THYM", "KICH", "DLBC"]

    args = []
    for c in cohorts:
        lab = (project_root / "data" / "labels" / f"{c.lower()}_labels.parquet").read_bytes()
        spl = (project_root / "data" / "splits" / f"{c.lower()}_splits.csv").read_bytes()
        args.append((c, lab, spl))

    all_results = []
    for r in run_cohort_abmil.starmap(args):
        all_results.extend(r)

    out = project_root / "results" / "abmil_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    ok = [r for r in all_results if r.get("status") == "ok"]
    print(f"\nABMIL: {len(ok)} cells with results")
    for r in sorted(ok, key=lambda x: -x.get("abmil_auroc", 0)):
        print(f"  {r['cell']:28s} ABMIL AUROC={r['abmil_auroc']:.3f} {r['abmil_ci']}")
    print(f"Saved: {out}")
