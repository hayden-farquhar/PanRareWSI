"""
PanRareWSI — Reliability diagrams for FDR-significant cells (pre-reg §7.3).

10-bin reliability curves, pre- and post-Platt calibration, with calibration-set
sizes annotated (small-n warning per §7.3).

Usage:
    python3 -m src.make_reliability
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.calibration import (
    expected_calibration_error, reliability_curve, recalibrate_cv, get_oof_predictions,
)
from src.phase4_benchmark import BIOMARKER_CELLS, load_cohort_data

FDR_CELLS = [
    ("DLBC", "MSI-H"), ("THYM", "GTF2I"), ("THYM", "TMB-high"),
    ("UVM", "EIF1AX"), ("UVM", "Chr3 loss"),
]


def build(project_root: Path | None = None):
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    cell_lookup = {}
    for cohort, cells in BIOMARKER_CELLS.items():
        for c in cells:
            cell_lookup[(cohort, c["name"])] = c

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6))

    for ax, (cohort, bm) in zip(axes, FDR_CELLS):
        cell = cell_lookup[(cohort, bm)]
        features, slide_ids, labels, splits = load_cohort_data(cohort, project_root)
        y_true, y_prob = get_oof_predictions(cohort, cell, features, slide_ids, labels, splits)

        platt = recalibrate_cv(y_true, y_prob, "platt")
        ece_pre = expected_calibration_error(y_true, y_prob, 15)
        ece_post = expected_calibration_error(y_true, platt, 15)

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect")
        for probs, lbl, color in [(y_prob, f"Raw (ECE {ece_pre:.2f})", "tab:red"),
                                   (platt, f"Platt (ECE {ece_post:.2f})", "tab:blue")]:
            centers, accs, counts = reliability_curve(y_true, probs, 10)
            valid = ~np.isnan(accs)
            ax.plot(centers[valid], accs[valid], "o-", color=color, label=lbl, markersize=5)

        ax.set_title(f"{cohort}/{bm}\n(n={len(y_true)})", fontsize=11)
        ax.set_xlabel("Predicted probability", fontsize=9)
        ax.set_ylabel("Observed frequency", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)

    fig.suptitle("Reliability diagrams — FDR-significant cells (10-bin, pre/post Platt scaling)", fontsize=13)
    plt.tight_layout()
    out_dir = project_root / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "reliability_diagrams.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / "reliability_diagrams.pdf", bbox_inches="tight")
    print(f"Saved: {out_dir / 'reliability_diagrams.png'}")


if __name__ == "__main__":
    build()
