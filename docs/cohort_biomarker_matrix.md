# PanRareWSI — Finalised Cohort × Biomarker Matrix

**Date finalised:** 2026-05-27 (supplementary labels added 2026-05-27 session 2)
**Data source:** cBioPortal PanCancer Atlas 2018 + Firehose Legacy + UCSC Xena (Thorsson immune subtypes) + Firehose cytogenetic/CNA/IHC data (queried 2026-05-27)
**Decision rule:** Viable (✓) = minority class ≥10 cases AND ≥10% prevalence. Borderline (?) = minority class 5–9 OR prevalence 5–10%. Excluded (✗) = minority class <5 OR prevalence <5%.

## Summary

| Cohort | n | Primary (viable) | Exploratory (borderline) | Excluded |
|--------|---|------------------|--------------------------|----------|
| ACC    | 93  | 5 | 1 | 1 |
| UVM    | 80  | 8 | 0 | 0 |
| MESO   | 87  | 6 | 0 | 0 |
| CHOL   | 36–51 | 0 | 4 | 3 |
| THYM   | 124 | 3 | 1 | 1 |
| KICH   | 65–113 | 3 | 1 | 1 |
| DLBC   | 48  | 5 | 2 | 0 |
| **Total** | **596** | **30** | **9** | **6** |

**Note:** CHOL has 0 primary cells (all biomarkers have minority class <10). DLBC MYD88 (minority=7) moved from primary to exploratory. See `osf_preregistration.md` §4 for full rationale.

**Changes from supplementary labels (session 2):** +8 primary cells (MESO CDKN2A homdel, UVM chr3 loss, DLBC Hans non-GCB, immune subtype binarised in ACC/UVM/MESO/KICH), +1 exploratory (CHOL immune), MESO CDKN2A moved from ✗ to ✓ (CNA vs mutation detection).

## Detailed Matrix

### ACC — Adrenocortical Carcinoma (n=93)

| Biomarker | Label type | Positive/Total | Prevalence | Viability | Notes |
|-----------|-----------|----------------|------------|-----------|-------|
| TP53 mutation | Binary | 18/93 | 19% | ✓ | Prognostic; associated with aggressive phenotype |
| CTNNB1 mutation | Binary | 14/93 | 15% | ✓ | Wnt pathway; defines molecular cluster |
| TMB-high (≥Q75) | Binary | 23/91 | 25% | ✓ | Immunotherapy-relevant |
| FGA-high (≥Q75) | Binary | 23/89 | 26% | ✓ | Proxy for chromosomal instability |
| Immune subtype (C4 vs non-C4) | Binary | C4 49, non-C4 29 / 78 mapped | 37% non-C4 | ✓ | Thorsson C1–C6; binarised at dominant subtype. 15/93 unmapped |
| MEN1 mutation | Binary | 7/93 | 8% | ? | Borderline n; clinically interesting |
| ZNRF3 mutation | Binary | 4/93 | 4% | ✗ | Too few positive cases |

### UVM — Uveal Melanoma (n=80)

| Biomarker | Label type | Positive/Total | Prevalence | Viability | Notes |
|-----------|-----------|----------------|------------|-----------|-------|
| GNAQ mutation | Binary | 40/80 | 50% | ✓ | Near-balanced; MAPK pathway |
| GNA11 mutation | Binary | 36/80 | 45% | ✓ | Near-balanced; MAPK pathway |
| SF3B1 mutation | Binary | 18/80 | 22% | ✓ | Splicing factor; prognostic |
| BAP1 mutation | Binary | 13/80 | 16% | ✓ | Tumour suppressor; poor prognosis marker |
| EIF1AX mutation | Binary | 10/80 | 12% | ✓ | Good prognosis marker |
| Histological subtype | Multi-class | Spindle 30 / Mixed 37 / Epithelioid 13 | 3 classes | ✓ | WHO morphological classification |
| Chromosome 3 loss | Binary | 31/52 mapped | 60% loss | ✓ | Robertson 2017; key prognostic division. 28/80 unmapped (35% missing) |
| Immune subtype (C4 vs non-C4) | Binary | C4 48, non-C4 32 / 80 mapped | 40% non-C4 | ✓ | Thorsson C1–C6; all 80 patients mapped |

### MESO — Mesothelioma (n=87)

| Biomarker | Label type | Positive/Total | Prevalence | Viability | Notes |
|-----------|-----------|----------------|------------|-----------|-------|
| Histological subtype (binary) | Binary | Epithelioid 62 / Non-epithelioid 25 | 29% non-epi | ✓ | Prognostic; treatment-relevant |
| NF2 mutation | Binary | 20/87 | 23% | ✓ | Tumour suppressor; Hippo pathway |
| BAP1 mutation | Binary | 18/87 | 21% | ✓ | Chromatin remodelling; prognostic |
| TP53 mutation | Binary | 14/87 | 16% | ✓ | Prognostic |
| CDKN2A homdel | Binary | 39/87 | 45% | ✓ | GISTIC CNA endpoint (not mutation); near-balanced. Previously ✗ from mutation data |
| Immune subtype (C1 vs non-C1) | Binary | C1 32, non-C1 51 / 83 mapped | 39% C1 | ✓ | Thorsson C1–C6; binarised at dominant subtype. 4/87 unmapped |

### CHOL — Cholangiocarcinoma (n=36 molecular, 51 total)

| Biomarker | Label type | Positive/Total | Prevalence | Viability | Notes |
|-----------|-----------|----------------|------------|-----------|-------|
| Anatomic subtype | Binary | Intrahepatic 30 / Other 6 | 17% other | ✓ | From Firehose (45 patients); clinically distinct |
| IDH1 mutation | Binary | 5/51 | 10% | ? | Targetable; borderline n |
| ARID1A mutation | Binary | 4/51 | 8% | ? | Chromatin remodelling; borderline |
| Immune subtype (C3 vs non-C3) | Binary | C3 17, non-C3 18 / 35 mapped | 49% C3 | ? | Thorsson C1–C6; near-balanced but only n=35 mapped (16/51 unmapped) |
| TP53 mutation | Binary | 4/51 | 8% | ✗ | Too few positive cases at n=51 |
| KRAS mutation | Binary | 2/51 | 4% | ✗ | Too few |
| IDH2 mutation | Binary | 2/51 | 4% | ✗ | Too few |

### THYM — Thymoma (n=124)

| Biomarker | Label type | Positive/Total | Prevalence | Viability | Notes |
|-----------|-----------|----------------|------------|-----------|-------|
| GTF2I mutation | Binary | 60/124 | 48% | ✓ | Hallmark driver; near-balanced |
| WHO histological subtype | Multi-class | AB 38, B2 28, A 15, B1 14, B3 12, TC 9, other 8 | 5+ classes | ✓ | WHO classification; prognostic |
| TMB-high (≥Q75) | Binary | 32/123 | 26% | ✓ | Immunotherapy-relevant |
| HRAS mutation | Binary | 10/124 | 8% | ? | Borderline; RAS pathway |
| TP53 mutation | Binary | 4/124 | 3% | ✗ | Too few |

### KICH — Kidney Chromophobe (n=65 PanCan molecular, 113 total)

| Biomarker | Label type | Positive/Total | Prevalence | Viability | Notes |
|-----------|-----------|----------------|------------|-----------|-------|
| TP53 mutation | Binary | 21/113 | 19% | ✓ | Prognostic |
| Aneuploidy-high (≥Q75) | Binary | 17/65 | 26% | ✓ | Chromosomal instability proxy |
| Immune subtype (C3 vs non-C3) | Binary | C3 38, non-C3 27 / 65 mapped | 42% non-C3 | ✓ | Thorsson C1–C6; binarised at dominant subtype. 48/113 unmapped (42% missing) |
| MSI-H (MSIsensor ≥3.5) | Binary | 7/65 | 11% | ? | Borderline; 0 by MANTIS |
| PTEN mutation | Binary | 6/113 | 5% | ✗ | Too few |

### DLBC — Diffuse Large B-Cell Lymphoma (n=48)

| Biomarker | Label type | Positive/Total | Prevalence | Viability | Notes |
|-----------|-----------|----------------|------------|-----------|-------|
| MSI-H (MANTIS ≥0.4) | Binary | 11/48 | 23% | ✓ | Highest MSI-H prevalence in the panel |
| TMB-high (≥Q75) | Binary | 11/41 | 27% | ✓ | Immunotherapy-relevant |
| Hans non-GCB | Binary | non-GCB 23, GCB 14 / 37 mapped | 38% GCB | ✓ | Hans algorithm (CD10/BCL6/MUM1 IHC). 11/48 unmapped |
| MYD88 mutation | Binary | 7/48 | 15% | ✓ | ABC-DLBCL associated; ibrutinib target |
| Aneuploidy-high (≥Q75) | Binary | 14/47 | 30% | ✓ | Chromosomal instability |
| CD79B mutation | Binary | 5/48 | 10% | ? | ABC-DLBCL associated; borderline n |
| TP53 mutation | Binary | 5/48 | 10% | ? | Prognostic; borderline n |

## Additional labels — sourcing status

1. ~~**Pan-cancer immune subtypes (C1–C6)**~~ — **Done.** Thorsson et al. 2018 via UCSC Xena. Mapped to ACC (78/93), UVM (80/80), MESO (83/87), CHOL (35/51), KICH (65/113). THYM and DLBC excluded from Thorsson (expected for thymic/haematological tumours).
2. ~~**DLBC ABC/GCB molecular subtype**~~ — **Done.** Hans algorithm (CD10/BCL6/MUM1 IHC) from Firehose. 37/48 mapped (23 non-GCB, 14 GCB).
3. ~~**UVM chromosome 3 status**~~ — **Done.** Firehose cytogenetic abnormality data. 52/80 mapped (31 loss, 21 intact).
4. **ACC CIMP status** — Not yet sourced. Requires Zheng et al. 2016 methylation supplementary.
5. **MESO molecular subtypes** — Not yet sourced. Requires Hmeljak et al. 2018 supplementary.
6. ~~**CDKN2A homozygous deletion (MESO)**~~ — **Done.** GISTIC CNA endpoint. 39/87 (45%) — now a primary cell.

## Decision gate outcomes

- **30 primary cells** across 6 cohorts (ACC 5, UVM 8, MESO 6, THYM 3, KICH 3, DLBC 5) — up from 22 after supplementary label sourcing.
- **9 exploratory cells** across 4 cohorts (ACC, CHOL, THYM, KICH, DLBC) — reported separately, not included in FDR correction.
- **CHOL has 0 primary cells** (all biomarkers have minority class <10 at n=35–51 mapped). All CHOL analyses are exploratory. This is itself an informative result about label scarcity in the smallest rare cohorts.
- **UVM is the strongest cohort** (8 primary biomarkers including chr3 loss and immune subtype, balanced classes, well-characterised drivers).
- **Remaining supplementary labels not yet sourced:** ACC CIMP status (Zheng 2016), MESO molecular subtypes (Hmeljak 2018). These could add 1–2 more exploratory cells if sourced.
