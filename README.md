# What pathology foundation models can and cannot recover from H&E in rare cancers

Code and derived-data repository for:

**What pathology foundation models can and cannot recover from H&E in rare cancers: a pre-registered molecular-biomarker benchmark across seven TCGA cohorts**

Hayden Farquhar MBBS MPHTM — Independent researcher, Finley, New South Wales, Australia
ORCID: [0009-0002-6226-440X](https://orcid.org/0009-0002-6226-440X)

- **Pre-registration:** https://doi.org/10.17605/OSF.IO/Y6HVP
- **OSF project:** https://osf.io/dqp6r/
- **Preprint:** https://doi.org/10.5281/zenodo.20427794

## Overview

This study benchmarks frozen pathology foundation-model slide embeddings (UNI2-h as the primary model; Prov-GigaPath as a comparison) for predicting molecular biomarkers from H&E histology across seven rare TCGA cohorts (ACC, UVM, MESO, CHOL, THYM, KICH, DLBC; 524 patients with matched embeddings and labels) that existing foundation-model benchmarks omit. Every analysis is pre-registered. The deliverable is a calibrated, FDR-controlled "recoverability map" — which cohort × biomarker cells are recoverable and which are indeterminate at these sample sizes — rather than a single headline performance figure.

This repository contains the analysis code and the derived outputs needed to reproduce every number, table, and figure in the paper. It does **not** contain the foundation-model embeddings themselves (see *Data Sources*).

## Data Sources

| Source | URL | Access | License | In this repo? |
|--------|-----|--------|---------|---------------|
| UNI2-h embeddings (`MahmoodLab/UNI2-h-features`, commit `1fdc2c03`) | https://huggingface.co/datasets/MahmoodLab/UNI2-h-features | Gated (user agreement) | CC BY-NC-ND 4.0 | **No** — not redistributable |
| Prov-GigaPath TCGA embeddings (`seandavis/tcga_provgigapath_embeddings`) | https://huggingface.co/datasets/seandavis/tcga_provgigapath_embeddings | Public | CC-BY-4.0 | No — obtain from source |
| Molecular labels (cBioPortal PanCancer Atlas 2018) | https://www.cbioportal.org/ | Public | CC0 | **Yes** (`data/labels/`) |
| Supplementary labels (TCGA primary publications via UCSC Xena, Broad Firehose GDAC) | https://xenabrowser.net/ ; https://gdac.broadinstitute.org/ | Public | TCGA data-use terms | **Yes** (merged into `data/labels/`) |
| CPTAC COAD (baseline-replication gate) | https://proteomics.cancer.gov/programs/cptac | Public | CC0 | No — obtain from source |

**Embeddings are deliberately excluded.** The UNI2-h embeddings are CC BY-NC-ND 4.0 (no derivatives, non-commercial) and cannot be redistributed. To re-run the pipeline from raw features, download them from the HuggingFace dataset above (commit `1fdc2c03`) after accepting the user agreement, and place the per-cohort `.h5` files under `data/embeddings/<COHORT>/`. The attention weights, out-of-fold predictions, results tables, and CV splits derived from those embeddings *are* released here.

## Requirements

Python 3.11+ (the analysis ran across CPython 3.11 on cloud workers and 3.14 locally; the linear-probe path is scikit-learn only and runs anywhere). Heavy steps (ABMIL training, 10k-permutation tests) were run on [Modal](https://modal.com/) with a T4 GPU and require a Modal account; all linear-probe analyses run on a laptop CPU.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

All experiments use a fixed random seed of **42**. Package versions are recorded in `requirements.txt`.

## Reproduction

Run from the repository root as a package (e.g. `python -m src.phase4_benchmark`). Derived outputs are already included under `results/` and `outputs/`, so each step can also be inspected without re-running.

```bash
# 1. Labels and splits (public data; no embeddings needed)
python -m src.data_loaders            # cBioPortal labels  -> data/labels/*.parquet
python -m src.supplementary_labels    # immune/chr3/Hans/CDKN2A labels (merged in)
python -m src.build_splits            # frozen 5-fold CV    -> data/splits/

# 2. Embeddings (requires the gated UNI2-h download placed in data/embeddings/)
python -m src.download_embeddings     # fetch UNI2-h-features (gated)
python -m src.mean_pool_local         # mean-pool patches   -> data/embeddings/<C>/mean_pooled_features.npy
python -m src.linkage_check           # barcode linkage audit

# 3. Baseline-replication gate (non-negotiable; must pass before rare-cohort work)
python -m src.phase3_baseline         # CPTAC COAD MSI      -> results/phase3_baseline.json

# 4. Main benchmark + derived predictions
python -m src.phase4_benchmark        # linear probe        -> results/phase4_benchmark.json
python -m src.export_oof_predictions  # per-patient OOF     -> results/oof_predictions.csv
python -m src.calibration             # ECE/Platt/isotonic  -> results/calibration.json

# 5. Significance, sensitivity, confounds
python -m src.permutation_tests       # (or the Modal variants for the full 10k) -> results/permutation_tests.json
python -m src.sensitivity_analyses    # S1/S3/S5/S6         -> results/sensitivity_analyses.json
python -m src.s2_meanvar_pool         # S2 mean+var pooling -> results/s2_meanvar.json
python -m src.site_analyses           # S7/S8 + TSS         -> results/site_analyses.json
python -m src.failure_modes           # error enrichment    -> results/failure_modes.json

# 6. Transfer + comparison foundation model
python -m src.phase5_transfer         # TP53 zero-shot      -> results/phase5_transfer.json
python -m src.tmb_transfer            # TMB-high transfer   -> results/tmb_transfer.json
python -m src.phase5b_gigapath        # Prov-GigaPath + ens -> results/phase5b_gigapath.json

# 7. Secondary model (ABMIL) — Modal/GPU
modal run src/modal_abmil.py          # ABMIL 5-fold        -> results/abmil_results.json
modal run src/modal_abmil_attention.py# attention maps      -> results/abmil_attention.json

# 8. Figures and tables
python -m src.make_heatmap            # Figure 1
python -m src.make_reliability        # Figure 2
python -m src.make_fig3               # Figure 3
python -m src.make_attention_fig      # Figure 4
python -m src.make_tables             # supplementary tables -> outputs/tables/
```

Linear-probe steps (1, 3, 4, 5, 6, 8) complete in minutes on a laptop. ABMIL and the full 10,000-permutation tests were run on a Modal T4 GPU; reduced local variants are available.

## Script Descriptions

| Script | Description | Key output |
|--------|-------------|-----------|
| `src/data_loaders.py` | Pull molecular labels from cBioPortal PanCancer Atlas 2018 | `data/labels/*.parquet` |
| `src/supplementary_labels.py` | Add immune subtype, UVM chr3, DLBC Hans, MESO CDKN2A labels | merged labels |
| `src/build_splits.py` | Patient-level stratified 5-fold CV (seed 42), frozen | `data/splits/*.csv`, `split_metadata.json` |
| `src/download_embeddings.py` | Fetch gated UNI2-h-features | `data/embeddings/` (not in repo) |
| `src/mean_pool_local.py` | Mean-pool patch embeddings to slide vectors | `mean_pooled_features.npy` |
| `src/linkage_check.py` | TCGA barcode → patient linkage audit | console / report |
| `src/phase3_baseline.py` | Baseline-replication gate (CPTAC COAD MSI) | `results/phase3_baseline.json` |
| `src/phase4_benchmark.py` | Linear probe across all cells, bootstrap CIs | `results/phase4_benchmark.json` |
| `src/export_oof_predictions.py` | Regenerate + verify per-patient OOF probabilities | `results/oof_predictions.csv` |
| `src/calibration.py` | ECE, Platt, isotonic recalibration (nested CV) | `results/calibration.json` |
| `src/permutation_tests.py` / `src/modal_perm*.py` | Label-permutation significance + BH-FDR | `results/permutation_tests.json` |
| `src/sensitivity_analyses.py` | PCA (S1), LOOCV (S3), BY-FDR (S5), threshold (S6) | `results/sensitivity_analyses.json` |
| `src/s2_meanvar_pool.py` | Mean+variance pooling sensitivity (S2) | `results/s2_meanvar.json` |
| `src/site_analyses.py` | Site-aware CV (S7), site-as-covariate (S8) | `results/site_analyses.json` |
| `src/failure_modes.py` | Per-site high-confidence error enrichment | `results/failure_modes.json` |
| `src/phase5_transfer.py` | TP53 zero-shot cross-cohort transfer | `results/phase5_transfer.json` |
| `src/tmb_transfer.py` | TMB-high cross-cohort transfer (TCGA-COAD source) | `results/tmb_transfer.json` |
| `src/phase5b_gigapath.py` | Prov-GigaPath comparison + two-model ensemble | `results/phase5b_gigapath.json` |
| `src/modal_abmil.py` | Gated-attention ABMIL secondary model (GPU) | `results/abmil_results.json` |
| `src/modal_abmil_attention.py` | ABMIL top-attention patches for the 5 FDR cells | `results/abmil_attention.json` |
| `src/modal_coad_meanpool.py`, `src/modal_thym_download.py`, `src/modal_populate_volume.py` | Cloud data staging for transfer/ABMIL | Modal volume |
| `src/models/linear_probe.py`, `src/models/multiclass_probe.py` | Shared probe implementations | — |
| `src/make_heatmap.py`, `make_reliability.py`, `make_fig3.py`, `make_attention_fig.py`, `make_tables.py` | Figure and table generation | `outputs/figures/`, `outputs/tables/` |

## Outputs

| File | Paper reference |
|------|----------------|
| `outputs/figures/master_heatmap.{png,pdf}` | Figure 1 (recoverability map) |
| `outputs/figures/reliability_diagrams.{png,pdf}` | Figure 2 (calibration) |
| `outputs/figures/fig3_model_comparison.{png,pdf}` | Figure 3 (ABMIL/GigaPath vs probe) |
| `outputs/figures/fig4_attention_maps.{png,pdf}` | Figure 4 (attention maps) |
| `results/phase4_benchmark.json` | Table 1 + Table S2 (per-cell metrics) |
| `results/oof_predictions.csv` | Per-patient out-of-fold probabilities (underpin AUROC/CIs/calibration) |
| `results/calibration.json` | Table S5 (ECE pre/post calibration) |
| `results/abmil_results.json` | Table S3 (ABMIL vs probe) |
| `results/phase5b_gigapath.json` | Table S4 (UNI2-h vs Prov-GigaPath vs ensemble) |
| `results/sensitivity_analyses.json`, `s2_meanvar.json`, `site_analyses.json` | Table S6 (sensitivity registry) |
| `results/phase5_transfer.json` | Table S7 (TP53 transfer) |
| `results/tmb_transfer.json` | Table S8 (TMB-high transfer) |
| `results/abmil_attention.json` | Figure 4 source (top-attention patches + coordinates) |

See `data_dictionary.md` for the columns of every released data file.

## Citation

If you use this code or the released data, please cite the accompanying paper and the pre-registration:

```
Farquhar H. What pathology foundation models can and cannot recover from H&E in
rare cancers: a pre-registered molecular-biomarker benchmark across seven TCGA
cohorts. Preprint: https://doi.org/10.5281/zenodo.20427794. Pre-registration: https://doi.org/10.17605/OSF.IO/Y6HVP
```

## License

- **Code** (`src/`): MIT License — see `LICENSE`.
- **Released data and documentation** (`data/`, `results/`, `outputs/`, `docs/`, this README): CC-BY-4.0.
- The UNI2-h and Prov-GigaPath embeddings are governed by their own licences (CC BY-NC-ND 4.0 and CC-BY-4.0 respectively) and are not included here.
