# ONCO-PROBE: Multimodal Adversarial Sensitivity Maps as Biological Probes for Triple Negative Breast Cancer

**Author:** Koushik Chowdhury, M.Sc. — Universität des Saarlandes, Germany

---

## Overview

This repository contains the analysis pipeline for using adversarial attacks (FGSM and PGD) as biological probes for Triple Negative Breast Cancer (TNBC). Rather than using adversarial attacks only for robustness evaluation, the framework directs them toward the TNBC class for every patient and treats the resulting gradient-magnitude maps as patient-level measurements of TNBC-directed sensitivity.

The pipeline trains histopathology image classifiers and gene-expression classifiers, computes adversarial sensitivity maps, performs multimodal fusion, evaluates established explainability methods, conducts statistical and power analyses, evaluates probability calibration, performs survival analysis and unsupervised clustering, and provides external validation using independent cohorts.

## Key Results

- Cross-model adversarial consistency: Spearman r = 0.9636, p = 1.85 × 10^-6
- TNBC vs non-TNBC FGSM sensitivity: Mann-Whitney p = 1.45 × 10^-6
- TNBC vs non-TNBC effect size: |Cohen's d| = 2.56; rank-biserial effect size = 0.896
- Minimum detectable effect at 80% power for the available A2 class sizes: Cohen's d = 0.91
- Primary multimodal adversarial fusion model (EfficientNet-derived features + Random Forest): LOO-CV AUC = 0.9693, 95% CI 0.9313-0.9947
- Alternative EfficientNet fusion baselines: Logistic Regression AUC = 0.9937, MLP AUC = 0.9873, gated multimodal neural fusion AUC = 0.9757
- Primary EfficientNet + Random Forest fusion calibration: Brier score = 0.0478, 10-bin ECE = 0.0573
- EfficientNet + Logistic Regression achieved the lowest Brier score: 0.0269
- EfficientNet + MLP achieved the lowest 10-bin ECE: 0.0303
- FGSM spatial agreement was strongest with SmoothGrad (mean Spearman r = 0.637) and Integrated Gradients (r = 0.543)
- 11 known TNBC-associated genes were recovered among the XGBoost top 20 features
- Estrogen response pathway enrichment: adjusted p = 2.57 × 10^-5
- External CPTAC BRCA validation included 26 confirmed TNBC patients
- Macenko normalization increased EfficientNet mean TNBC probability from 0.070 to 0.203
- Progression-free interval log-rank p = 0.0072
- Adjusted overall-survival HR = 0.737, 95% CI 0.579-0.937, p = 0.0129
- Adjusted progression-free-interval HR = 0.599, 95% CI 0.474-0.759, p < 0.001

---

## Repository Structure

```text
01_data_loading.ipynb
    Data preparation, TNBC label definition, ComBat batch correction,
    gene-expression processing, and patient-level data splits.

02_models.ipynb
    Gene classifier training and METABRIC external validation.

03_adversarial_sensitivity.ipynb
    Main adversarial sensitivity analyses, multimodal fusion,
    pathway enrichment, survival analysis, and UMAP clustering.

train_models.py
    Trains the five histopathology image classifiers.

compute_sensitivity_v3.py
    Computes FGSM and PGD sensitivity maps for the TCGA-A2 cohort.

compute_sensitivity_e2.py
    Computes FGSM and PGD sensitivity maps for the independent
    TCGA-E2 cohort.

reviewer_revision/
├── 01_cptac_inventory.py
│   CPTAC BRCA DICOM/whole-slide-image inventory and dataset inspection.
│
├── 02_cptac_wsidicom_test.py
│   Tests reading CPTAC DICOM whole-slide images with wsidicom.
│
├── 03_cptac_extract_tissue_tiles.py
│   Generates tissue masks and extracts tissue tiles from CPTAC slides.
│
├── 04_cptac_adversarial_validation.py
│   Runs ResNet50-TS and EfficientNet-B0-TS inference and adversarial
│   sensitivity analysis on native CPTAC images.
│
├── 05_cptac_summarize_results.py
│   Summarizes native CPTAC prediction and sensitivity results.
│
├── 06_verify_class_mapping.py
│   Verifies TNBC/non-TNBC class-index mapping used by the trained models.
│
├── 07_macenko_normalize_cptac_tiles.py
│   Applies Macenko stain normalization to CPTAC tissue tiles.
│
├── 08_cptac_adversarial_validation_macenko.py
│   Repeats CPTAC model inference and adversarial analysis using
│   Macenko-normalized tiles.
│
├── 09_cptac_summarize_macenko.py
│   Summarizes prediction and sensitivity results after stain normalization.
│
├── 10_compare_native_macenko.py
│   Statistically compares native and Macenko-normalized CPTAC results.
│
├── 11_multivariable_cox.py
│   Performs penalized multivariable Cox regression adjusted for age,
│   tumour stage, and TNBC status.
│
├── 12_fusion_bootstrap_ci.py
│   Computes patient-level bootstrap confidence intervals for multimodal
│   fusion LOO-CV AUC estimates and saves out-of-fold predictions.
│
├── 13_remaining_analyses.py
│   Performs additional reviewer-requested analyses associated with the
│   multimodal/adversarial framework.
│
├── 14_remaining_analyses_part2.py
│   Performs explainability comparisons, alternative fusion baselines,
│   sample-size/power analysis, multiple-testing audit, and
│   reproducibility audit.
│
├── 15_multimodal_gated_fusion.py
│   Implements a compact gated multimodal neural fusion architecture and
│   compares it with Random Forest, XGBoost, Logistic Regression, and MLP
│   fusion models under patient-level LOO-CV.
│
└── 16_calibration_analysis.py
    Evaluates calibration of multimodal fusion models using out-of-fold
    probabilities, including Brier score, 10-bin expected calibration
    error (ECE), calibration intercept/slope, and calibration curves.
```
---

## Processed Outputs

Trained models, sensitivity maps, figures, and processed results are **not stored in this repository**. Download from Harvard Dataverse:

**Chowdhury, Koushik, 2026, "Output Dataset: Processed Adversarial Sensitivity Maps and Trained Models Derived from TCGA-BRCA for Triple Negative Breast Cancer Analysis", https://doi.org/10.7910/DVN/VWT2W8, Harvard Dataverse**

The download includes:
- `models_v2/` — 5 image classifiers (.pth) and 2 gene classifiers (.pkl)
- `sensitivity_maps_v2/` — FGSM and PGD maps for all patients (.npy)
- `results_v2/figures/` — all figures
- `results_v2/results/` — pathway enrichment, survival data, summary JSON
- `results_v2/processed/` — patient splits, gene expression matrices, clinical data

---

## Loading the Processed Files

After downloading from Harvard Dataverse:

```import numpy as np
import joblib

# Image sensitivity maps
# Each .npy file contains a dictionary:
# {patient_id: 224x224 numpy array}

# TCGA-A2 — ResNet50-TS
fgsm_resnet_tnbc = np.load(
    'fgsm_resnet_tnbc_v3.npy', allow_pickle=True
).item()

pgd_resnet_tnbc = np.load(
    'pgd_resnet_tnbc_v3.npy', allow_pickle=True
).item()

fgsm_resnet_non_tnbc = np.load(
    'fgsm_resnet_non_tnbc_v3.npy', allow_pickle=True
).item()

pgd_resnet_non_tnbc = np.load(
    'pgd_resnet_non_tnbc_v3.npy', allow_pickle=True
).item()

# TCGA-A2 — EfficientNet-TS
fgsm_eff_tnbc = np.load(
    'fgsm_eff_tnbc_v3.npy', allow_pickle=True
).item()

pgd_eff_tnbc = np.load(
    'pgd_eff_tnbc_v3.npy', allow_pickle=True
).item()

# TCGA-E2 external cohort
fgsm_resnet_tnbc_e2 = np.load(
    'fgsm_resnet_tnbc_e2.npy', allow_pickle=True
).item()

pgd_resnet_tnbc_e2 = np.load(
    'pgd_resnet_tnbc_e2.npy', allow_pickle=True
).item()

fgsm_resnet_non_tnbc_e2 = np.load(
    'fgsm_resnet_non_tnbc_e2.npy', allow_pickle=True
).item()

pgd_resnet_non_tnbc_e2 = np.load(
    'pgd_resnet_non_tnbc_e2.npy', allow_pickle=True
).item()

fgsm_eff_tnbc_e2 = np.load(
    'fgsm_eff_tnbc_e2.npy', allow_pickle=True
).item()

pgd_eff_tnbc_e2 = np.load(
    'pgd_eff_tnbc_e2.npy', allow_pickle=True
).item()

# Access one patient map
patient_id = list(fgsm_resnet_tnbc.keys())[0]
map_224x224 = fgsm_resnet_tnbc[patient_id]

# Gene sensitivity
rf_gene_sensitivity = np.load(
    'rf_gene_sensitivity.npy', allow_pickle=True
)

xgb_gene_sensitivity = np.load(
    'xgb_gene_sensitivity.npy', allow_pickle=True
)

gene_names = np.load(
    'top_gene_names.npy', allow_pickle=True
)

# Gene classifiers
rf = joblib.load('random_forest_best.pkl')
xgb = joblib.load('xgboost_best.pkl')
scaler = joblib.load('gene_scaler.pkl')
```

---

## Raw Data

Raw data is not included. Download from the original sources:

| Dataset | Source |
|---|---|
| TCGA-BRCA image tiles and clinical files | zenodo.org/records/5337009 |
| TCGA HiSeqV2 gene expression | UCSC Xena Browser |
| GSE76124 | ncbi.nlm.nih.gov/geo accession GSE76124 |
| GSE58812 | ncbi.nlm.nih.gov/geo accession GSE58812 |
| GSE103091 | ncbi.nlm.nih.gov/geo accession GSE103091 |
| BreakHis | kaggle.com/datasets/ambarish/breakhis |
| METABRIC | cBioPortal study brca_metabric |
| CPTAC BRCA DICOM whole slide images | NCI Imaging Data Commons at portal.imaging.datacommons.cancer.gov/explore/ |

---

## Requirements

```
- Python 3.12
- Package dependencies are listed in `requirements.txt`.

Install the required packages with:

```bash
pip install -r requirements.txt
```

---

## Additional Revision Analyses

The manuscript revision added an independent CPTAC BRCA histopathology validation and an adjusted survival analysis.

The CPTAC workflow includes:

- DICOM whole slide image inventory
- WSI reading with `wsidicom`
- Tissue mask generation
- Extraction of 100 tiles per patient
- Validation with ResNet50-TS and EfficientNet-B0-TS
- Native versus Macenko normalized image comparison
- Patient and tile level statistical summaries

The survival workflow includes penalized multivariable Cox regression for overall survival and progression free interval. The models include standardized FGSM sensitivity, standardized age at diagnosis, tumour stage, and TNBC status.

## Citation

If you use this code or data, please cite the associated paper (citation to be added upon publication).

---

## License

Code is released under MIT License. Processed outputs on Harvard Dataverse are released under CC BY 4.0.
