"""
PanRareWSI — Figure 3: paired scatter showing neither alternative beats the
linear probe. Left panel: ABMIL vs linear probe. Right panel: Prov-GigaPath vs
UNI2-h linear probe. Diagonal = parity; points below the line = alternative worse.

Usage:
    python3 -m src.make_fig3
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build(project_root: Path | None = None):
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    p4 = {r["cell"]: r for r in json.loads((project_root / "results" / "phase4_benchmark.json").read_text()) if r["status"] == "ok"}
    abmil = {r["cell"]: r for r in json.loads((project_root / "results" / "abmil_results.json").read_text()) if r.get("status") == "ok"}
    giga = {r["cell"]: r for r in json.loads((project_root / "results" / "phase5b_gigapath.json").read_text())["cells"]}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))

    # Panel A: ABMIL vs linear probe
    ax = axes[0]
    xs = [p4[c]["pooled_auroc"] for c in abmil if c in p4]
    ys = [abmil[c]["abmil_auroc"] for c in abmil if c in p4]
    ax.scatter(xs, ys, c="tab:purple", s=45, alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.plot([0.3, 1.0], [0.3, 1.0], "k--", lw=1, alpha=0.6)
    ax.fill_between([0.3, 1.0], [0.3, 1.0], 0.3, color="tab:red", alpha=0.04)
    ax.set_xlabel("Linear probe AUROC", fontsize=11)
    ax.set_ylabel("ABMIL AUROC", fontsize=11)
    ax.set_title(f"A. ABMIL vs linear probe\n(mean Δ = {np.mean([y-x for x,y in zip(xs,ys)]):+.3f}; below line = ABMIL worse)", fontsize=10)
    ax.set_xlim(0.3, 1.0); ax.set_ylim(0.3, 1.0); ax.set_aspect("equal"); ax.grid(alpha=0.3)

    # Panel B: GigaPath vs UNI2-h (using the matched-subset UNI2 values for fairness)
    ax = axes[1]
    xs = [giga[c]["uni2_auroc"] for c in giga]
    ys = [giga[c]["gigapath_auroc"] for c in giga]
    ax.scatter(xs, ys, c="tab:green", s=45, alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.plot([0.3, 1.0], [0.3, 1.0], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("UNI2-h AUROC (matched subset)", fontsize=11)
    ax.set_ylabel("Prov-GigaPath AUROC", fontsize=11)
    ax.set_title(f"B. Prov-GigaPath vs UNI2-h\n(mean {np.mean(ys):.3f} vs {np.mean(xs):.3f}; near parity)", fontsize=10)
    ax.set_xlim(0.3, 1.0); ax.set_ylim(0.3, 1.0); ax.set_aspect("equal"); ax.grid(alpha=0.3)

    fig.suptitle("Neither attention-MIL nor a second foundation model improves on the linear probe at rare-cohort scale", fontsize=12)
    plt.tight_layout()
    out_dir = project_root / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "fig3_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / "fig3_model_comparison.pdf", bbox_inches="tight")
    print(f"Saved: {out_dir / 'fig3_model_comparison.png'}")


if __name__ == "__main__":
    build()
