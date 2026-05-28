"""
PanRareWSI — Figure 4: ABMIL spatial attention maps (Phase 6b, RQ4 attention half).

For the two strongest FDR-significant cells, plots the top-attention patch
coordinates for one correctly-classified and one misclassified slide, showing
where the attention model focuses. Pixel-level morphological categorisation
(necrosis/lymphoid/artefact) requires raw WSI review by a pathologist and is
left to a follow-up; this figure delivers the attention-localisation component.

Requires results/abmil_attention.json (from modal_abmil_attention.py).

Usage:
    python3 -m src.make_attention_fig
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build(project_root: Path | None = None):
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    data = json.loads((project_root / "results" / "abmil_attention.json").read_text())

    # pick two showcase cells
    cells = [c for c in ["THYM/GTF2I", "DLBC/MSI-H"] if c in data]
    if not cells:
        cells = list(data.keys())[:2]

    fig, axes = plt.subplots(len(cells), 2, figsize=(11, 5.2 * len(cells)))
    if len(cells) == 1:
        axes = axes.reshape(1, 2)

    for row, cell in enumerate(cells):
        patients = data[cell]["patients"]
        correct = [p for p in patients if p["correct"] == 1]
        wrong = [p for p in patients if p["correct"] == 0]
        # most confident correct and most confident wrong
        correct.sort(key=lambda p: -abs(p["pred"] - 0.5))
        wrong.sort(key=lambda p: -abs(p["pred"] - 0.5))

        for col, (group, label) in enumerate([(correct, "correct"), (wrong, "misclassified")]):
            ax = axes[row, col]
            if not group:
                ax.set_visible(False)
                continue
            p = group[0]
            coords = np.array(p["top_attn"]["coords"])
            weights = np.array(p["top_attn"]["weights"])
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=weights, s=80,
                            cmap="hot", edgecolor="k", linewidth=0.3)
            ax.invert_yaxis()  # WSI coords: y increases downward
            ax.set_title(f"{cell} — {label}\npatient {p['patient_id']}, "
                         f"true={p['true']}, pred={p['pred']:.2f} "
                         f"({p['top_attn']['n_patches']} patches)", fontsize=9)
            ax.set_xlabel("patch x (px)", fontsize=8)
            ax.set_ylabel("patch y (px)", fontsize=8)
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="attention")

    fig.suptitle("ABMIL top-20 attention patches by WSI location (correct vs misclassified slides)", fontsize=12)
    plt.tight_layout()
    out_dir = project_root / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "fig4_attention_maps.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / "fig4_attention_maps.pdf", bbox_inches="tight")
    print(f"Saved: {out_dir / 'fig4_attention_maps.png'}")


if __name__ == "__main__":
    build()
