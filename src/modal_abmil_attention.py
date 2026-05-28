"""
PanRareWSI — Phase 6b: ABMIL attention maps for FDR-significant cells.

Retrains gated-attention ABMIL (full 5-fold CV) on the 5 FDR-significant cells,
and for each patient records out-of-fold prediction + per-patch attention weights
+ patch coordinates. Enables spatial attention maps (attention over WSI coordinate
space) for misclassified vs correctly-classified slides, addressing the
attention-map half of pre-registration RQ4 without needing raw WSI pixels.

Usage:
    modal run src/modal_abmil_attention.py
"""

import modal

app = modal.App("panrarewsi-abmil-attn")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.1", "h5py>=3.9", "numpy>=1.24",
                 "scikit-learn>=1.3", "pandas>=2.0", "pyarrow>=14.0")
)
volume = modal.Volume.from_name("panrarewsi-embeddings", create_if_missing=True)

# (cohort, biomarker, label_col, binarise_target) for the 5 FDR-significant cells
FDR_CELLS = [
    ("DLBC", "MSI-H", "label_msi_h", None),
    ("THYM", "GTF2I", "mut_GTF2I", None),
    ("THYM", "TMB-high", "label_tmb_high", None),
    ("UVM", "EIF1AX", "mut_EIF1AX", None),
    ("UVM", "Chr3 loss", "label_chr3_loss", None),
]


def patch_dir_for(cohort):
    return "/vol/THYM" if cohort == "THYM" else f"/vol/patches/{cohort}"


@app.function(image=image, timeout=3600, memory=16384, gpu="T4", volumes={"/vol": volume})
def attention_for_cohort(cohort: str, cells: list, labels_bytes: bytes, splits_bytes: bytes) -> dict:
    import io, json
    from pathlib import Path
    import h5py, numpy as np, pandas as pd
    import torch, torch.nn as nn
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(42); np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = pd.read_parquet(io.BytesIO(labels_bytes))
    splits = pd.read_csv(io.BytesIO(splits_bytes))

    pdir = Path(patch_dir_for(cohort))
    h5s = sorted(pdir.glob("*.h5"))

    def pid_of(stem): return "-".join(stem.split(".")[0].split("-")[:3])

    # one slide per patient (first); keep coords for attention map
    patient_bag, patient_coords = {}, {}
    for h5 in h5s:
        pid = pid_of(h5.stem)
        if pid in patient_bag:
            continue
        with h5py.File(h5, "r") as f:
            patient_bag[pid] = f["features"][:][0]
            patient_coords[pid] = f["coords"][:][0]

    class GatedAttn(nn.Module):
        def __init__(self, d=1536, h=256, a=128):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.25))
            self.V = nn.Linear(h, a); self.U = nn.Linear(h, a); self.w = nn.Linear(a, 1)
            self.clf = nn.Linear(h, 1)
        def forward(self, x):
            h = self.fc(x)
            a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))
            a = torch.softmax(a, dim=0)
            return self.clf((a * h).sum(0, keepdim=True)).squeeze(), a.squeeze()

    smap = dict(zip(splits["patient_id"], splits["fold"]))
    # Idempotent per-cohort save (survives Modal app-runtime limit)
    out_dir = Path("/vol/attention")
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = out_dir / f"{cohort}.json"
    if cohort_path.exists():
        print(f"{cohort}: already saved, skipping")
        return json.loads(cohort_path.read_text())
    out = {}
    for name, col, binar in cells:
        lab = labels[col].astype(float)
        lmap = dict(zip(labels["patient_id"], lab))
        pids, y, folds = [], [], []
        for pid in patient_bag:
            if pid in lmap and pid in smap and pd.notna(lmap[pid]):
                pids.append(pid); y.append(lmap[pid]); folds.append(smap[pid])
        y = np.array(y); folds = np.array(folds)

        oof_pred = {}
        oof_attn = {}
        for fi in range(5):
            te = folds == fi; tr = ~te
            tr_pids = [pids[i] for i in np.where(tr)[0]]
            te_pids = [pids[i] for i in np.where(te)[0]]
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            pos_w = float((y[tr] == 0).sum()) / max(1, (y[tr] == 1).sum())
            model = GatedAttn().to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
            lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=device))
            rng = np.random.RandomState(42)
            for epoch in range(35):
                for i in rng.permutation(np.where(tr)[0]):
                    x = torch.tensor(patient_bag[pids[i]], dtype=torch.float32, device=device)
                    t = torch.tensor(y[i], dtype=torch.float32, device=device)
                    opt.zero_grad(); logit, _ = model(x)
                    lossf(logit.unsqueeze(0), t.unsqueeze(0)).backward(); opt.step()
            model.eval()
            with torch.no_grad():
                for i in np.where(te)[0]:
                    pid = pids[i]
                    x = torch.tensor(patient_bag[pid], dtype=torch.float32, device=device)
                    logit, attn = model(x)
                    oof_pred[pid] = float(torch.sigmoid(logit).item())
                    # store top-20 attention patches (coord + weight) to keep payload small
                    a = attn.cpu().numpy()
                    coords = patient_coords[pid]
                    topk = np.argsort(-a)[:20]
                    oof_attn[pid] = {"coords": coords[topk].tolist(),
                                     "weights": a[topk].tolist(),
                                     "n_patches": int(len(a))}
        # assemble
        cell_out = {"patients": []}
        for i, pid in enumerate(pids):
            if pid in oof_pred:
                cell_out["patients"].append({
                    "patient_id": pid, "true": int(y[i]),
                    "pred": round(oof_pred[pid], 4),
                    "correct": int((oof_pred[pid] >= 0.5) == bool(y[i])),
                    "top_attn": oof_attn[pid],
                })
        valid_y = [p["true"] for p in cell_out["patients"]]
        valid_p = [p["pred"] for p in cell_out["patients"]]
        if len(set(valid_y)) > 1:
            cell_out["auroc"] = round(roc_auc_score(valid_y, valid_p), 4)
        out[f"{cohort}/{name}"] = cell_out
        print(f"  {cohort}/{name}: {len(cell_out['patients'])} patients, AUROC={cell_out.get('auroc')}")
    cohort_path.write_text(json.dumps(out))
    volume.commit()
    return out


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path
    pr = Path(__file__).resolve().parent.parent

    by_cohort = {}
    for cohort, name, col, binar in FDR_CELLS:
        by_cohort.setdefault(cohort, []).append((name, col, binar))

    all_out = {}
    for cohort, cells in by_cohort.items():
        lab = (pr / "data" / "labels" / f"{cohort.lower()}_labels.parquet").read_bytes()
        spl = (pr / "data" / "splits" / f"{cohort.lower()}_splits.csv").read_bytes()
        res = attention_for_cohort.remote(cohort, cells, lab, spl)
        all_out.update(res)

    out_path = pr / "results" / "abmil_attention.json"
    with open(out_path, "w") as f:
        json.dump(all_out, f, indent=2)
    print(f"Saved attention data for {len(all_out)} cells: {out_path}")
