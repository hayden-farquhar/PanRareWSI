"""
PanRareWSI — Master cohort × biomarker AUROC heatmap (pre-registration deliverable).

Produces the headline figure: a heatmap of pooled AUROC across all cohort ×
biomarker cells, annotated with significance (BH-FDR) and tier. Cells are
coloured by AUROC; FDR-significant cells are marked. Insufficient-n cells
are shown as grey.

Usage:
    python3 -m src.make_heatmap
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

COHORT_ORDER = ["ACC", "UVM", "MESO", "CHOL", "THYM", "KICH", "DLBC"]


def build_heatmap(project_root: Path | None = None):
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    phase4 = json.loads((project_root / "results" / "phase4_benchmark.json").read_text())
    perm = json.loads((project_root / "results" / "permutation_tests.json").read_text())
    perm_map = {r["cell"]: r for r in perm}

    # Build cell grid: rows = biomarkers (union), cols = cohorts
    ok = [r for r in phase4 if r["status"] == "ok"]

    # Organise by cohort
    cohort_cells = {c: [] for c in COHORT_ORDER}
    for r in ok:
        cohort_cells[r["cohort"]].append(r)
    for c in cohort_cells:
        cohort_cells[c].sort(key=lambda r: -r["pooled_auroc"])

    # Max rows
    max_rows = max(len(v) for v in cohort_cells.values())

    fig, ax = plt.subplots(figsize=(11, 9))
    cmap = plt.cm.RdYlBu_r
    # Anchor the colour midpoint at AUROC 0.5 (chance): blue below, red above.
    norm = mcolors.TwoSlopeNorm(vmin=0.35, vcenter=0.5, vmax=1.0)

    for col_idx, cohort in enumerate(COHORT_ORDER):
        cells = cohort_cells[cohort]
        for row_idx, cell in enumerate(cells):
            auroc = cell["pooled_auroc"]
            ci = cell["auroc_ci"]
            tier = cell["tier"]
            pm = perm_map.get(cell["cell"], {})
            sig = pm.get("bh_significant", False)

            color = cmap(norm(auroc))
            rect = plt.Rectangle((col_idx, max_rows - 1 - row_idx), 1, 1,
                                 facecolor=color, edgecolor="white", linewidth=1.5)
            ax.add_patch(rect)

            # Significance marker
            star = " *" if sig else ""
            label = f"{cell['biomarker']}{star}\n{auroc:.2f}\n[{ci[0]:.2f},{ci[1]:.2f}]"
            txt_color = "white" if (auroc > 0.82 or auroc < 0.5) else "black"
            ax.text(col_idx + 0.5, max_rows - 1 - row_idx + 0.5, label,
                    ha="center", va="center", fontsize=6.5, color=txt_color,
                    fontweight="bold" if tier == "primary" else "normal")

    ax.set_xlim(0, len(COHORT_ORDER))
    ax.set_ylim(0, max_rows)
    ax.set_xticks([i + 0.5 for i in range(len(COHORT_ORDER))])
    ax.set_xticklabels([f"{c}\n(n={cohort_cells[c][0]['n'] if cohort_cells[c] else 0})"
                        for c in COHORT_ORDER], fontsize=10, fontweight="bold")
    ax.set_yticks([])
    ax.set_title("UNI2-h linear-probe AUROC across rare TCGA cohorts × biomarkers\n"
                 "* = BH-FDR significant (α=0.05); bold = primary cell",
                 fontsize=12, pad=15)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Pooled out-of-fold AUROC", fontsize=10)

    # Reference line annotation for chance
    ax.text(len(COHORT_ORDER) + 0.15, max_rows * 0.02,
            "AUROC 0.50 = chance", fontsize=7, rotation=90, color="gray")

    plt.tight_layout()
    out_dir = project_root / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "master_heatmap.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / "master_heatmap.pdf", bbox_inches="tight")
    print(f"Saved: {out_path}")
    print(f"Saved: {out_dir / 'master_heatmap.pdf'}")

    return out_path


if __name__ == "__main__":
    build_heatmap()
