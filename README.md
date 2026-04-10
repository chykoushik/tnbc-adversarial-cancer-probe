# ONCO-PROBE: Multimodal Adversarial Sensitivity Maps as Biological Probes for Triple Negative Breast Cancer

**Author:** Koushik Chowdhury, M.Sc. — Universität des Saarlandes, Germany

---

## Overview

This repository contains the full analysis pipeline for using adversarial attacks (FGSM and PGD) as biological discovery tools for Triple Negative Breast Cancer (TNBC). Rather than using adversarial attacks for robustness evaluation, this project directs them at the TNBC class for every patient and treats the resulting gradient magnitude maps as per-patient biological measurements.

The pipeline trains five histopathology image classifiers and two gene expression classifiers, computes adversarial sensitivity maps for all patients, and performs multimodal fusion, pathway enrichment, survival analysis, and unsupervised clustering.

**Key results:**
- Cross-model adversarial consistency: Spearman r = 0.9636, p = 1.85×10⁻⁶
- TNBC vs non-TNBC separation: Mann-Whitney p = 1.45×10⁻⁶
- Best fusion model (EfficientNet-TS + RF): LOO-CV AUC = 0.9693
- 11 known TNBC genes recovered in XGBoost top 20 without supervision
- Estrogen response pathway enrichment: adjusted p = 2.57×10⁻⁵
- Survival PFI log-rank p = 0.0072, Cox HR = 0.54

---

## Repository Structure

```
01_data_loading.ipynb         — data preparation, TNBC labels, ComBat batch correction, patient splits
02_models.ipynb               — gene classifier training, METABRIC external validation
03_adversarial_sensitivity.ipynb  — all analysis: sensitivity, fusion, survival, UMAP, pathways
train_models.py               — trains all 5 image classifiers locally on GPU (~13 hours)
compute_sensitivity_v3.py     — computes FGSM+PGD maps for A2 cohort (~40 minutes)
compute_sensitivity_e2.py     — computes FGSM+PGD maps for E2 external cohort (~20 minutes)
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

```python
import numpy as np
import joblib

# ── image sensitivity maps ────────────────────────────────────────────────
# each .npy file is a dict: {patient_id: 224x224 numpy array}

# A2 cohort — ResNet50-TS (used for fusion model and all main analysis)
fgsm_resnet_tnbc     = np.load('fgsm_resnet_tnbc_v3.npy',     allow_pickle=True).item()
pgd_resnet_tnbc      = np.load('pgd_resnet_tnbc_v3.npy',      allow_pickle=True).item()
fgsm_resnet_non_tnbc = np.load('fgsm_resnet_non_tnbc_v3.npy', allow_pickle=True).item()
pgd_resnet_non_tnbc  = np.load('pgd_resnet_non_tnbc_v3.npy',  allow_pickle=True).item()

# A2 cohort — EfficientNet-TS (TNBC only, used for cross-model consistency)
fgsm_eff_tnbc = np.load('fgsm_eff_tnbc_v3.npy', allow_pickle=True).item()
pgd_eff_tnbc  = np.load('pgd_eff_tnbc_v3.npy',  allow_pickle=True).item()

# E2 external cohort
fgsm_resnet_tnbc_e2     = np.load('fgsm_resnet_tnbc_e2.npy',     allow_pickle=True).item()
pgd_resnet_tnbc_e2      = np.load('pgd_resnet_tnbc_e2.npy',      allow_pickle=True).item()
fgsm_resnet_non_tnbc_e2 = np.load('fgsm_resnet_non_tnbc_e2.npy', allow_pickle=True).item()
pgd_resnet_non_tnbc_e2  = np.load('pgd_resnet_non_tnbc_e2.npy',  allow_pickle=True).item()
fgsm_eff_tnbc_e2        = np.load('fgsm_eff_tnbc_e2.npy',        allow_pickle=True).item()
pgd_eff_tnbc_e2         = np.load('pgd_eff_tnbc_e2.npy',         allow_pickle=True).item()

# access one patient map
patient_id  = list(fgsm_resnet_tnbc.keys())[0]
map_224x224 = fgsm_resnet_tnbc[patient_id]  # shape (224, 224)

# ── gene sensitivity maps ─────────────────────────────────────────────────
rf_gene_sensitivity  = np.load('rf_gene_sensitivity.npy',  allow_pickle=True)
xgb_gene_sensitivity = np.load('xgb_gene_sensitivity.npy', allow_pickle=True)
gene_names           = np.load('top_gene_names.npy',        allow_pickle=True)

# ── image classifiers ─────────────────────────────────────────────────────
import torch
from torchvision import models
import timm

# best model: EfficientNet-TS (AUC 0.9259, used in best fusion combination)
efficientnet_ts = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
efficientnet_ts.load_state_dict(torch.load('efficientnet_ts_best.pth', map_location='cpu'))
efficientnet_ts.eval()

# ResNet50-TS (AUC 0.9259, used for all sensitivity map computation)
resnet50_ts = models.resnet50(pretrained=False)
resnet50_ts.fc = torch.nn.Linear(resnet50_ts.fc.in_features, 2)
resnet50_ts.load_state_dict(torch.load('resnet50_ts_best.pth', map_location='cpu'))
resnet50_ts.eval()

# ── gene classifiers ──────────────────────────────────────────────────────
# best fusion combination: EfficientNet-TS + Random Forest (LOO-CV AUC 0.9693)
rf     = joblib.load('random_forest_best.pkl')
xgb    = joblib.load('xgboost_best.pkl')
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

---

## Requirements

```
python 3.12
pytorch
torchvision
timm
scikit-learn
xgboost
pandas
numpy
scipy
matplotlib
lifelines
umap-learn
inmoose
```

---

## Citation

If you use this code or data, please cite the associated paper (citation to be added upon publication).

---

## License

Code is released under MIT License. Processed outputs on Harvard Dataverse are released under CC BY 4.0.
