import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import transforms, models
from PIL import Image
import timm
import warnings
warnings.filterwarnings('ignore')

# paths — update if different on your machine
EXTRACT_E2  = r"E:\apply\journal publication\onco-probe\dataset\tcga_images_e2"
CLINI_E2    = r"E:\apply\journal publication\onco-probe\dataset\TCGA-BRCA-E2-CLINI.xlsx"
MODELS_DIR  = r"E:\apply\journal publication\onco-probe\models"
OUTPUT_DIR  = r"E:\apply\journal publication\onco-probe\sensitivity_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

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

def compute_pgd_sensitivity(model, image_tensor, epsilon=0.03, alpha=0.007, steps=10):
    model.eval()
    img = image_tensor.unsqueeze(0).to(device)
    original    = img.clone()
    perturbed   = img.clone()
    accumulator = torch.zeros(224, 224)
    for step in range(steps):
        perturbed.requires_grad = True
        output = model(perturbed)
        loss   = nn.CrossEntropyLoss()(output, torch.tensor([1]).to(device))
        model.zero_grad()
        loss.backward()
        gradient    = perturbed.grad.data
        accumulator += gradient.abs().squeeze().cpu().mean(dim=0)
        perturbed   = perturbed + alpha * gradient.sign()
        delta        = torch.clamp(perturbed - original, min=-epsilon, max=epsilon)
        perturbed    = torch.clamp(original + delta, 0, 1).detach()
    return (accumulator / steps).numpy()

def get_patient_id(fname):
    parts = fname.split('-')
    if len(parts) >= 4:
        pid = '-'.join(parts[:4])
        pid = pid.replace('Z', '').replace('z', '')
        return pid
    return None

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

# load e2 clinical labels
clini_e2  = pd.read_excel(CLINI_E2)
er2       = clini_e2['ER Status By IHC'].str.strip() == 'Negative'
pr2       = clini_e2['PR status by ihc'].str.strip() == 'Negative'
her2_2    = clini_e2['IHC-HER2'].str.strip() == 'Negative'
clini_e2['TNBC'] = (er2 & pr2 & her2_2).astype(int)

tnbc_ids_e2     = clini_e2[clini_e2['TNBC'] == 1]['Sample ID'].tolist()
non_tnbc_ids_e2 = clini_e2[clini_e2['TNBC'] == 0]['Sample ID'].tolist()

print(f"e2 tnbc patients:     {len(tnbc_ids_e2)}")
print(f"e2 non-tnbc patients: {len(non_tnbc_ids_e2)}")

# match non-tnbc count to tnbc for balance
np.random.seed(42)
non_tnbc_ids_e2 = np.random.choice(
    non_tnbc_ids_e2, size=len(tnbc_ids_e2), replace=False).tolist()
print(f"e2 non-tnbc sampled:  {len(non_tnbc_ids_e2)}")

# load models
resnet = models.resnet50(pretrained=False)
resnet.fc = nn.Linear(resnet.fc.in_features, 2)
resnet.load_state_dict(torch.load(
    os.path.join(MODELS_DIR, 'resnet50_bh_best.pth'), map_location=device))
resnet = resnet.to(device)
resnet.eval()
print("resnet50-ts loaded")

efficientnet = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
efficientnet.load_state_dict(torch.load(
    os.path.join(MODELS_DIR, 'efficientnet_bh_best.pth'), map_location=device))
efficientnet = efficientnet.to(device)
efficientnet.eval()
print("efficientnet-ts loaded")

# find images
print("\nfinding e2 tnbc images...")
tnbc_images_e2 = find_patient_images(EXTRACT_E2, tnbc_ids_e2, max_per_patient=100)
print(f"e2 tnbc patients with images: {len(tnbc_images_e2)}")

print("finding e2 non-tnbc images...")
non_tnbc_images_e2 = find_patient_images(EXTRACT_E2, non_tnbc_ids_e2, max_per_patient=100)
print(f"e2 non-tnbc patients with images: {len(non_tnbc_images_e2)}")

# compute maps — resnet tnbc
print("\ncomputing resnet50-ts fgsm and pgd for e2 tnbc...")
fgsm_resnet_tnbc_e2 = {}
pgd_resnet_tnbc_e2  = {}
for pid, img_paths in tnbc_images_e2.items():
    fgsm_list, pgd_list = [], []
    for path in img_paths:
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img)
            fgsm_list.append(compute_fgsm_sensitivity(resnet, img_tensor))
            pgd_list.append(compute_pgd_sensitivity(resnet, img_tensor))
        except:
            continue
    if fgsm_list:
        fgsm_resnet_tnbc_e2[pid] = np.mean(fgsm_list, axis=0)
        pgd_resnet_tnbc_e2[pid]  = np.mean(pgd_list,  axis=0)
    print(f"  resnet tnbc e2 done {pid} tiles {len(fgsm_list)}")

# compute maps — resnet non-tnbc
print("\ncomputing resnet50-ts fgsm and pgd for e2 non-tnbc...")
fgsm_resnet_non_tnbc_e2 = {}
pgd_resnet_non_tnbc_e2  = {}
for pid, img_paths in non_tnbc_images_e2.items():
    fgsm_list, pgd_list = [], []
    for path in img_paths:
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img)
            fgsm_list.append(compute_fgsm_sensitivity(resnet, img_tensor))
            pgd_list.append(compute_pgd_sensitivity(resnet, img_tensor))
        except:
            continue
    if fgsm_list:
        fgsm_resnet_non_tnbc_e2[pid] = np.mean(fgsm_list, axis=0)
        pgd_resnet_non_tnbc_e2[pid]  = np.mean(pgd_list,  axis=0)
    print(f"  resnet non-tnbc e2 done {pid} tiles {len(fgsm_list)}")

# compute maps — efficientnet tnbc
print("\ncomputing efficientnet-ts fgsm and pgd for e2 tnbc...")
fgsm_eff_tnbc_e2 = {}
pgd_eff_tnbc_e2  = {}
for pid, img_paths in tnbc_images_e2.items():
    fgsm_list, pgd_list = [], []
    for path in img_paths:
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img)
            fgsm_list.append(compute_fgsm_sensitivity(efficientnet, img_tensor))
            pgd_list.append(compute_pgd_sensitivity(efficientnet, img_tensor))
        except:
            continue
    if fgsm_list:
        fgsm_eff_tnbc_e2[pid] = np.mean(fgsm_list, axis=0)
        pgd_eff_tnbc_e2[pid]  = np.mean(pgd_list,  axis=0)
    print(f"  efficientnet tnbc e2 done {pid} tiles {len(fgsm_list)}")

# save
np.save(os.path.join(OUTPUT_DIR, 'fgsm_resnet_tnbc_e2.npy'),     fgsm_resnet_tnbc_e2)
np.save(os.path.join(OUTPUT_DIR, 'pgd_resnet_tnbc_e2.npy'),      pgd_resnet_tnbc_e2)
np.save(os.path.join(OUTPUT_DIR, 'fgsm_resnet_non_tnbc_e2.npy'), fgsm_resnet_non_tnbc_e2)
np.save(os.path.join(OUTPUT_DIR, 'pgd_resnet_non_tnbc_e2.npy'),  pgd_resnet_non_tnbc_e2)
np.save(os.path.join(OUTPUT_DIR, 'fgsm_eff_tnbc_e2.npy'),        fgsm_eff_tnbc_e2)
np.save(os.path.join(OUTPUT_DIR, 'pgd_eff_tnbc_e2.npy'),         pgd_eff_tnbc_e2)
print("\nall e2 sensitivity maps saved")
print("upload to Drive at ONCO-PROBE/sensitivity_maps_v2/image_maps/")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if 'e2' in f:
        print(f"  {f}")