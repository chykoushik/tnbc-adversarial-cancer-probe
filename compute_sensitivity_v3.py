import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import transforms, models
from PIL import Image
import timm
import pickle
import joblib
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# paths
EXTRACT_A2 = r"E:\apply\journal publication\onco-probe\dataset\tcga_images_a2"
EXTRACT_E2 = r"E:\apply\journal publication\onco-probe\dataset\TCGA-BRCA-E2-DEEPMED-TILES.zip"
e2_extract_dir = r"E:\apply\journal publication\onco-probe\dataset\tcga_images_e2"
CLINI_A2 = r"E:\apply\journal publication\onco-probe\dataset\TCGA-BRCA-A2-CLINI.xlsx"
CLINI_E2 = r"E:\apply\journal publication\onco-probe\dataset\TCGA-BRCA-E2-CLINI.xlsx"
MODELS_DIR = r"E:\apply\journal publication\onco-probe\models"
OUTPUT_DIR = r"E:\apply\journal publication\onco-probe\sensitivity_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device", device)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

# transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# fix 4 — fgsm targets class 1 tnbc for all images — biological probe design
# we measure tnbc-directed sensitivity for every image regardless of true label
# this enables direct comparison between tnbc and non-tnbc patients
def compute_fgsm_sensitivity(model, image_tensor):
    model.eval()
    img = image_tensor.unsqueeze(0).to(device)
    img.requires_grad = True
    output = model(img)
    loss = nn.CrossEntropyLoss()(output, torch.tensor([1]).to(device))
    model.zero_grad()
    loss.backward()
    sensitivity = img.grad.data.abs().squeeze().cpu()
    return sensitivity.mean(dim=0).numpy()

# fix 4 — pgd targets class 1 tnbc for all images — biological probe design
def compute_pgd_sensitivity(model, image_tensor, epsilon=0.03, alpha=0.007, steps=10):
    model.eval()
    img = image_tensor.unsqueeze(0).to(device)
    original = img.clone()
    perturbed = img.clone()
    accumulator = torch.zeros(224, 224)
    for step in range(steps):
        perturbed.requires_grad = True
        output = model(perturbed)
        loss = nn.CrossEntropyLoss()(output, torch.tensor([1]).to(device))
        model.zero_grad()
        loss.backward()
        gradient = perturbed.grad.data
        accumulator += gradient.abs().squeeze().cpu().mean(dim=0)
        perturbed = perturbed + alpha * gradient.sign()
        delta = torch.clamp(perturbed - original, min=-epsilon, max=epsilon)
        perturbed = torch.clamp(original + delta, 0, 1).detach()
    return (accumulator / steps).numpy()

# get patient id from filename
def get_patient_id(fname):
    parts = fname.split('-')
    if len(parts) >= 4:
        pid = '-'.join(parts[:4])
        pid = pid.replace('Z', '').replace('z', '')
        return pid
    return None

# find images per patient
def find_patient_images(extract_dir, patient_ids, max_per_patient=100):
    patient_images = {}
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if f.endswith('.jpg') or f.endswith('.png'):
                pid = get_patient_id(f)
                if pid in patient_ids:
                    if pid not in patient_images:
                        patient_images[pid] = []
                    if len(patient_images[pid]) < max_per_patient:
                        patient_images[pid].append(os.path.join(root, f))
    return patient_images

# get slide level predictions
def get_slide_probs(model, patient_images):
    model.eval()
    patient_probs = {}
    for pid, img_paths in patient_images.items():
        probs = []
        for path in img_paths:
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(img_tensor)
                    prob = torch.softmax(output, dim=1)[:, 1].item()
                probs.append(prob)
            except:
                continue
        if probs:
            patient_probs[pid] = np.mean(probs)
    return patient_probs

# load a2 clinical labels
clini_a2 = pd.read_excel(CLINI_A2)
er = clini_a2['ER Status By IHC'].str.strip() == 'Negative'
pr = clini_a2['PR status by ihc'].str.strip() == 'Negative'
her2 = clini_a2['IHC-HER2'].str.strip() == 'Negative'
clini_a2['TNBC'] = (er & pr & her2).astype(int)
tnbc_ids_a2 = clini_a2[clini_a2['TNBC'] == 1]['Sample ID'].tolist()
non_tnbc_ids_a2 = clini_a2[clini_a2['TNBC'] == 0]['Sample ID'].tolist()
print("a2 tnbc patients", len(tnbc_ids_a2))
print("a2 non-tnbc patients all", len(non_tnbc_ids_a2))

# load e2 clinical labels
clini_e2 = pd.read_excel(CLINI_E2)
er2 = clini_e2['ER Status By IHC'].str.strip() == 'Negative'
pr2 = clini_e2['PR status by ihc'].str.strip() == 'Negative'
her2_2 = clini_e2['IHC-HER2'].str.strip() == 'Negative'
clini_e2['TNBC'] = (er2 & pr2 & her2_2).astype(int)
all_e2_ids = clini_e2['Sample ID'].tolist()
print("e2 total patients", len(all_e2_ids))
print("e2 tnbc patients", clini_e2['TNBC'].sum())

# load resnet50-ts
resnet = models.resnet50(pretrained=False)
resnet.fc = nn.Linear(resnet.fc.in_features, 2)
resnet.load_state_dict(torch.load(
    os.path.join(MODELS_DIR, 'resnet50_bh_best.pth'),
    map_location=device))
resnet = resnet.to(device)
resnet.eval()
print("resnet50-ts loaded")

# load efficientnet-ts
efficientnet = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
efficientnet.load_state_dict(torch.load(
    os.path.join(MODELS_DIR, 'efficientnet_bh_best.pth'),
    map_location=device))
efficientnet = efficientnet.to(device)
efficientnet.eval()
print("efficientnet-ts loaded")

# find a2 images
print("finding a2 tnbc images")
tnbc_images_a2 = find_patient_images(EXTRACT_A2, tnbc_ids_a2, max_per_patient=100)
print("a2 tnbc patients with images", len(tnbc_images_a2))

# fix 1 — equal tile sampling 100 tiles for non-tnbc same as tnbc
print("finding a2 non-tnbc images all 90 — 100 tiles each")
non_tnbc_images_a2 = find_patient_images(EXTRACT_A2, non_tnbc_ids_a2, max_per_patient=100)
print("a2 non-tnbc patients with images", len(non_tnbc_images_a2))

# compute fgsm and pgd for tnbc using resnet50-ts
print("computing resnet50-ts fgsm and pgd for a2 tnbc")
fgsm_resnet_tnbc = {}
pgd_resnet_tnbc = {}

for pid, img_paths in tnbc_images_a2.items():
    fgsm_list = []
    pgd_list = []
    for path in img_paths:
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img)
            fgsm_list.append(compute_fgsm_sensitivity(resnet, img_tensor))
            pgd_list.append(compute_pgd_sensitivity(resnet, img_tensor))
        except:
            continue
    if fgsm_list:
        fgsm_resnet_tnbc[pid] = np.mean(fgsm_list, axis=0)
        pgd_resnet_tnbc[pid] = np.mean(pgd_list, axis=0)
    print(f"resnet tnbc done {pid} tiles {len(fgsm_list)}")

# compute fgsm and pgd for non-tnbc using resnet50-ts — all 90 patients 100 tiles each
print("computing resnet50-ts fgsm and pgd for a2 non-tnbc all 90")
fgsm_resnet_non_tnbc = {}
pgd_resnet_non_tnbc = {}

for pid, img_paths in non_tnbc_images_a2.items():
    fgsm_list = []
    pgd_list = []
    for path in img_paths:
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img)
            fgsm_list.append(compute_fgsm_sensitivity(resnet, img_tensor))
            pgd_list.append(compute_pgd_sensitivity(resnet, img_tensor))
        except:
            continue
    if fgsm_list:
        fgsm_resnet_non_tnbc[pid] = np.mean(fgsm_list, axis=0)
        pgd_resnet_non_tnbc[pid] = np.mean(pgd_list, axis=0)
    print(f"resnet non-tnbc done {pid} tiles {len(fgsm_list)}")

# compute fgsm and pgd for tnbc using efficientnet-ts — cross model consistency
print("computing efficientnet-ts fgsm and pgd for a2 tnbc")
fgsm_eff_tnbc = {}
pgd_eff_tnbc = {}

for pid, img_paths in tnbc_images_a2.items():
    fgsm_list = []
    pgd_list = []
    for path in img_paths:
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img)
            fgsm_list.append(compute_fgsm_sensitivity(efficientnet, img_tensor))
            pgd_list.append(compute_pgd_sensitivity(efficientnet, img_tensor))
        except:
            continue
    if fgsm_list:
        fgsm_eff_tnbc[pid] = np.mean(fgsm_list, axis=0)
        pgd_eff_tnbc[pid] = np.mean(pgd_list, axis=0)
    print(f"efficientnet tnbc done {pid} tiles {len(fgsm_list)}")

# save all maps as v3
np.save(os.path.join(OUTPUT_DIR, 'fgsm_resnet_tnbc_v3.npy'), fgsm_resnet_tnbc)
np.save(os.path.join(OUTPUT_DIR, 'pgd_resnet_tnbc_v3.npy'), pgd_resnet_tnbc)
np.save(os.path.join(OUTPUT_DIR, 'fgsm_resnet_non_tnbc_v3.npy'), fgsm_resnet_non_tnbc)
np.save(os.path.join(OUTPUT_DIR, 'pgd_resnet_non_tnbc_v3.npy'), pgd_resnet_non_tnbc)
np.save(os.path.join(OUTPUT_DIR, 'fgsm_eff_tnbc_v3.npy'), fgsm_eff_tnbc)
np.save(os.path.join(OUTPUT_DIR, 'pgd_eff_tnbc_v3.npy'), pgd_eff_tnbc)
print("sensitivity maps saved")

# save image sensitivity summary v3
rows = []
for pid in fgsm_resnet_tnbc:
    rows.append({
        'patient_id': pid, 'label': 1,
        'fgsm_mean': fgsm_resnet_tnbc[pid].mean(),
        'fgsm_max': fgsm_resnet_tnbc[pid].max(),
        'fgsm_std': fgsm_resnet_tnbc[pid].std(),
        'fgsm_p75': np.percentile(fgsm_resnet_tnbc[pid], 75),
        'fgsm_p90': np.percentile(fgsm_resnet_tnbc[pid], 90),
        'pgd_mean': pgd_resnet_tnbc[pid].mean(),
        'pgd_max': pgd_resnet_tnbc[pid].max(),
        'pgd_std': pgd_resnet_tnbc[pid].std(),
        'pgd_p75': np.percentile(pgd_resnet_tnbc[pid], 75),
        'pgd_p90': np.percentile(pgd_resnet_tnbc[pid], 90),
    })
for pid in fgsm_resnet_non_tnbc:
    rows.append({
        'patient_id': pid, 'label': 0,
        'fgsm_mean': fgsm_resnet_non_tnbc[pid].mean(),
        'fgsm_max': fgsm_resnet_non_tnbc[pid].max(),
        'fgsm_std': fgsm_resnet_non_tnbc[pid].std(),
        'fgsm_p75': np.percentile(fgsm_resnet_non_tnbc[pid], 75),
        'fgsm_p90': np.percentile(fgsm_resnet_non_tnbc[pid], 90),
        'pgd_mean': pgd_resnet_non_tnbc[pid].mean(),
        'pgd_max': pgd_resnet_non_tnbc[pid].max(),
        'pgd_std': pgd_resnet_non_tnbc[pid].std(),
        'pgd_p75': np.percentile(pgd_resnet_non_tnbc[pid], 75),
        'pgd_p90': np.percentile(pgd_resnet_non_tnbc[pid], 90),
    })

pd.DataFrame(rows).to_csv(
    os.path.join(OUTPUT_DIR, 'image_sensitivity_summary_v3.csv'), index=False)
print("image sensitivity summary v3 saved")

# e2 external validation
print()
print("e2 external validation")

if os.path.exists(e2_extract_dir):
    total_e2 = sum(len(f) for _, _, f in os.walk(e2_extract_dir))
    print(f"e2 images found {total_e2}")
else:
    print("e2 not extracted - extracting now")
    import zipfile
    os.makedirs(e2_extract_dir, exist_ok=True)
    with zipfile.ZipFile(EXTRACT_E2, 'r') as z:
        z.extractall(e2_extract_dir)
    print("e2 extracted")
    total_e2 = sum(len(f) for _, _, f in os.walk(e2_extract_dir))
    print(f"e2 images {total_e2}")

print("finding e2 patient images")
e2_images = find_patient_images(e2_extract_dir, all_e2_ids, max_per_patient=200)
print("e2 patients with images", len(e2_images))

print("computing resnet50-ts slide predictions for e2")
e2_probs_resnet = get_slide_probs(resnet, e2_images)

print("computing efficientnet-ts slide predictions for e2")
e2_probs_eff = get_slide_probs(efficientnet, e2_images)

e2_label_map = dict(zip(clini_e2['Sample ID'], clini_e2['TNBC']))

e2_rows = []
for pid in e2_probs_resnet:
    if pid in e2_label_map:
        e2_rows.append({
            'patient_id': pid,
            'label': e2_label_map[pid],
            'resnet_ts_prob': e2_probs_resnet[pid],
            'eff_ts_prob': e2_probs_eff.get(pid, np.nan),
        })

e2_df = pd.DataFrame(e2_rows)
e2_df.to_csv(os.path.join(OUTPUT_DIR, 'e2_validation_results_v3.csv'), index=False)

if len(e2_df) > 0 and e2_df['label'].nunique() > 1:
    e2_auc_resnet = roc_auc_score(e2_df['label'], e2_df['resnet_ts_prob'])
    e2_auc_eff = roc_auc_score(e2_df['label'], e2_df['eff_ts_prob'].fillna(0))

    print()
    print("e2 external validation results")
    print(f"e2 patients evaluated {len(e2_df)}")
    print(f"e2 tnbc patients {e2_df['label'].sum()}")
    print(f"e2 non-tnbc patients {(e2_df['label']==0).sum()}")
    print(f"resnet50-ts e2 slide auc {round(e2_auc_resnet, 4)}")
    print(f"efficientnet-ts e2 slide auc {round(e2_auc_eff, 4)}")

    preds_resnet = (e2_df['resnet_ts_prob'] >= 0.5).astype(int)
    print()
    print("resnet50-ts e2 classification report")
    print(classification_report(e2_df['label'], preds_resnet,
                                target_names=['non-tnbc', 'tnbc'],
                                zero_division=0))
else:
    print("not enough e2 patients for auc calculation")

print()
print("all done")
print("upload sensitivity_v3 folder to drive at ONCO-PROBE/sensitivity_maps_v2/image_maps/")
print("files saved")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  {f}")