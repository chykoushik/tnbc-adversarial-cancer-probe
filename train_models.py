import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                             precision_recall_curve, confusion_matrix,
                             f1_score, accuracy_score)
import timm
import json
import warnings
warnings.filterwarnings('ignore')

# paths
ZIP_A2 = r"E:\apply\journal publication\onco-probe\dataset\TCGA-BRCA-A2-DEEPMED-TILES.zip"
ZIP_BREAKHIS = r"E:\apply\journal publication\onco-probe\dataset\BreakHis.zip"
CLINI_A2 = r"E:\apply\journal publication\onco-probe\dataset\TCGA-BRCA-A2-CLINI.xlsx"
EXTRACT_A2 = r"E:\apply\journal publication\onco-probe\dataset\tcga_images_a2"
EXTRACT_BREAKHIS = r"E:\apply\journal publication\onco-probe\dataset\breakhis"
MODELS_DIR = r"E:\apply\journal publication\onco-probe\models"
os.makedirs(MODELS_DIR, exist_ok=True)


# focal loss for imbalanced data
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(
            targets == 1,
            torch.tensor(self.alpha, device=inputs.device),
            torch.tensor(1 - self.alpha, device=inputs.device)
        )
        return (alpha_t * (1 - pt) ** self.gamma * ce_loss).mean()


# dataset
class CancerDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx, 'path']
        label = self.df.loc[idx, 'label']
        try:
            img = Image.open(path).convert('RGB')
        except:
            img = Image.new('RGB', (224, 224), color=0)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


# weighted sampler for balanced batches
def make_sampler(df):
    labels = df['label'].values
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]
    return WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.float),
        num_samples=len(weights),
        replacement=True
    )


# find best threshold using f1
def find_optimal_threshold(labels, probs):
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5


# get patient id from filename
def get_patient_id(fname):
    parts = fname.split('-')
    if len(parts) >= 4:
        pid = '-'.join(parts[:4])
        pid = pid.replace('Z', '').replace('z', '')
        return pid
    return None


# train one epoch
def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    running_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    return running_loss / len(loader)


# evaluate one epoch
def eval_epoch(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            val_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    v_auc = roc_auc_score(all_labels, all_probs)
    return val_loss / len(loader), v_auc


# full training loop saves best model by val auc
def train_full(model, train_loader, val_loader, optimizer, scheduler,
               criterion, epochs, model_name, save_dir, scaler, device):
    best_val_auc = 0
    patience = 5
    patience_counter = 0
    print("training", model_name)
    for epoch in range(epochs):
        t_loss = train_epoch(model, train_loader, optimizer,
                             criterion, scaler, device)
        v_loss, v_auc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        if v_auc > best_val_auc:
            best_val_auc = v_auc
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f'{model_name}_best.pth'))
        else:
            patience_counter += 1
        torch.save(model.state_dict(),
                   os.path.join(save_dir, f'{model_name}_last.pth'))
        print(f"epoch {epoch+1}/{epochs} loss {round(t_loss,4)} val_loss {round(v_loss,4)} val_auc {round(v_auc,4)}")
        if patience_counter >= patience:
            print(f"early stopping at epoch {epoch+1}")
            break
    print(f"best val auc {round(best_val_auc,4)}")
    return best_val_auc


# slide level evaluation averages tile predictions per patient
def evaluate_slide_level(model, test_df, val_transform, device, model_name):
    print(f"slide level results for {model_name}")
    model.eval()
    dataset = CancerDataset(test_df, transform=val_transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    all_probs = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())

    test_df = test_df.copy()
    test_df['prob'] = all_probs

    patient_df = test_df.groupby('sample_id').agg(
        prob=('prob', 'mean'),
        label=('label', 'first')
    ).reset_index()

    slide_labels = patient_df['label'].values
    slide_probs = patient_df['prob'].values

    tile_auc = roc_auc_score(test_df['label'].values, test_df['prob'].values)
    slide_auc = roc_auc_score(slide_labels, slide_probs)
    optimal_threshold = find_optimal_threshold(slide_labels, slide_probs)

    preds_default = (slide_probs >= 0.5).astype(int)
    preds_optimal = (slide_probs >= optimal_threshold).astype(int)

    print(f"test patients {len(patient_df)}")
    print(f"tnbc patients {int(slide_labels.sum())}")
    print(f"non-tnbc patients {int((slide_labels==0).sum())}")
    print(f"tile auc {round(tile_auc, 4)}")
    print(f"slide auc {round(slide_auc, 4)}")
    print(f"optimal threshold {round(optimal_threshold, 4)}")
    print("results at default threshold 0.5")
    print(classification_report(slide_labels, preds_default,
                                target_names=['Non-TNBC', 'TNBC'],
                                zero_division=0))
    print(f"results at optimal threshold {round(optimal_threshold,4)}")
    print(classification_report(slide_labels, preds_optimal,
                                target_names=['Non-TNBC', 'TNBC'],
                                zero_division=0))
    print("confusion matrix at optimal threshold")
    print(confusion_matrix(slide_labels, preds_optimal))

    results = {
        'model': model_name,
        'tile_auc': round(tile_auc, 4),
        'slide_auc': round(slide_auc, 4),
        'optimal_threshold': round(optimal_threshold, 4),
        'n_test_patients': len(patient_df),
        'n_tnbc_test': int(slide_labels.sum()),
        'accuracy_default': round(accuracy_score(slide_labels, preds_default), 4),
        'accuracy_optimal': round(accuracy_score(slide_labels, preds_optimal), 4),
        'f1_tnbc_default': round(f1_score(slide_labels, preds_default, zero_division=0), 4),
        'f1_tnbc_optimal': round(f1_score(slide_labels, preds_optimal, zero_division=0), 4),
    }
    return slide_auc, results


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device", device)
    print("gpu", torch.cuda.get_device_name(0))

    # extract a2
    if not os.path.exists(EXTRACT_A2) or len(os.listdir(EXTRACT_A2)) == 0:
        os.makedirs(EXTRACT_A2, exist_ok=True)
        print("extracting a2")
        with zipfile.ZipFile(ZIP_A2, 'r') as z:
            z.extractall(EXTRACT_A2)
        print("a2 extracted")
    else:
        total = sum(len(f) for _, _, f in os.walk(EXTRACT_A2))
        print(f"a2 already extracted {total} files")

    # extract breakhis
    if not os.path.exists(EXTRACT_BREAKHIS) or len(os.listdir(EXTRACT_BREAKHIS)) == 0:
        os.makedirs(EXTRACT_BREAKHIS, exist_ok=True)
        print("extracting breakhis")
        with zipfile.ZipFile(ZIP_BREAKHIS, 'r') as z:
            z.extractall(EXTRACT_BREAKHIS)
        print("breakhis extracted")
    else:
        print("breakhis already extracted")

    # transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    criterion = FocalLoss(alpha=0.75, gamma=2.0)

    # build breakhis dataframe
    print("building breakhis dataframe")
    breakhis_records = []
    for root, dirs, files in os.walk(EXTRACT_BREAKHIS):
        for f in files:
            if f.endswith('.jpg') or f.endswith('.png'):
                path = os.path.join(root, f)
                parts = path.split(os.sep)
                label = None
                for part in parts:
                    if part.lower() == 'malignant':
                        label = 1
                        break
                    elif part.lower() == 'benign':
                        label = 0
                        break
                if label is not None:
                    breakhis_records.append({'path': path, 'label': label,
                                             'sample_id': 'breakhis'})

    breakhis_df = pd.DataFrame(breakhis_records)
    print(f"breakhis total {len(breakhis_df)} malignant {(breakhis_df['label']==1).sum()} benign {(breakhis_df['label']==0).sum()}")

    bh_train, bh_val = train_test_split(breakhis_df, test_size=0.2,
                                         random_state=42,
                                         stratify=breakhis_df['label'])
    bh_sampler = make_sampler(bh_train)
    bh_train_ds = CancerDataset(bh_train, transform=train_transform)
    bh_val_ds = CancerDataset(bh_val, transform=val_transform)
    bh_train_loader = DataLoader(bh_train_ds, batch_size=64,
                                  sampler=bh_sampler, num_workers=2)
    bh_val_loader = DataLoader(bh_val_ds, batch_size=64,
                                shuffle=False, num_workers=2)

    # build a2 image dataframe using all images
    print("building a2 image dataframe")
    clini = pd.read_excel(CLINI_A2)
    er = clini['ER Status By IHC'].str.strip() == 'Negative'
    pr = clini['PR status by ihc'].str.strip() == 'Negative'
    her2 = clini['IHC-HER2'].str.strip() == 'Negative'
    clini['label'] = (er & pr & her2).astype(int)
    print(f"tnbc patients {clini['label'].sum()} non-tnbc patients {(clini['label']==0).sum()}")

    records = []
    for root, dirs, files in os.walk(EXTRACT_A2):
        for f in files:
            if f.endswith('.jpg') or f.endswith('.png'):
                path = os.path.join(root, f)
                pid = get_patient_id(f)
                records.append({'path': path, 'sample_id': pid})

    img_df = pd.DataFrame(records)
    img_df = img_df.merge(
        clini[['Sample ID', 'label']],
        left_on='sample_id',
        right_on='Sample ID',
        how='inner'
    )
    print(f"total images {len(img_df)} tnbc {(img_df['label']==1).sum()} non-tnbc {(img_df['label']==0).sum()}")

    # patient level split guaranteeing tnbc in each set
    tnbc_patients = img_df[img_df['label']==1]['sample_id'].unique()
    non_tnbc_patients = img_df[img_df['label']==0]['sample_id'].unique()

    tnbc_train_p, tnbc_temp = train_test_split(tnbc_patients, test_size=0.4, random_state=42)
    tnbc_val_p, tnbc_test_p = train_test_split(tnbc_temp, test_size=0.5, random_state=42)
    non_train_p, non_temp = train_test_split(non_tnbc_patients, test_size=0.4, random_state=42)
    non_val_p, non_test_p = train_test_split(non_temp, test_size=0.5, random_state=42)

    train_patients = list(tnbc_train_p) + list(non_train_p)
    val_patients = list(tnbc_val_p) + list(non_val_p)
    test_patients = list(tnbc_test_p) + list(non_test_p)

    train_df = img_df[img_df['sample_id'].isin(train_patients)].copy()
    val_df = img_df[img_df['sample_id'].isin(val_patients)].copy()
    test_df = img_df[img_df['sample_id'].isin(test_patients)].copy()

    print(f"train {len(train_df)} tiles tnbc patients {train_df[train_df['label']==1]['sample_id'].nunique()}")
    print(f"val {len(val_df)} tiles tnbc patients {val_df[val_df['label']==1]['sample_id'].nunique()}")
    print(f"test {len(test_df)} tiles tnbc patients {test_df[test_df['label']==1]['sample_id'].nunique()}")

    train_df.to_csv(os.path.join(MODELS_DIR, 'train_df.csv'), index=False)
    val_df.to_csv(os.path.join(MODELS_DIR, 'val_df.csv'), index=False)
    test_df.to_csv(os.path.join(MODELS_DIR, 'test_df.csv'), index=False)

    # dataloaders for a2
    train_sampler = make_sampler(train_df)
    train_dataset = CancerDataset(train_df, transform=train_transform)
    val_dataset = CancerDataset(val_df, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=64,
                              sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64,
                            shuffle=False, num_workers=2)
    print(f"train batches {len(train_loader)} val batches {len(val_loader)}")

    all_results = {}

    # model 1 resnet50 imagenet only then a2 fine-tune
    print("model 1 resnet50 imagenet")
    resnet_imagenet = models.resnet50(pretrained=True)
    resnet_imagenet.fc = nn.Linear(resnet_imagenet.fc.in_features, 2)
    resnet_imagenet = resnet_imagenet.to(device)
    opt_r1 = optim.Adam(resnet_imagenet.parameters(), lr=0.0001)
    sch_r1 = optim.lr_scheduler.CosineAnnealingLR(opt_r1, T_max=15)
    scaler_r1 = torch.cuda.amp.GradScaler()
    train_full(resnet_imagenet, train_loader, val_loader,
               opt_r1, sch_r1, criterion,
               epochs=15, model_name='resnet50_imagenet',
               save_dir=MODELS_DIR, scaler=scaler_r1, device=device)
    resnet_imagenet.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, 'resnet50_imagenet_best.pth')))
    auc_r1, res_r1 = evaluate_slide_level(
        resnet_imagenet, test_df, val_transform, device, 'resnet50_imagenet')
    all_results['resnet50_imagenet'] = res_r1

    # model 2 resnet50 breakhis pretrain then a2 fine-tune
    print("model 2 resnet50 breakhis pretrain")
    resnet_bh = models.resnet50(pretrained=True)
    resnet_bh.fc = nn.Linear(resnet_bh.fc.in_features, 2)
    resnet_bh = resnet_bh.to(device)
    bh_opt_r = optim.Adam(resnet_bh.parameters(), lr=0.0001)
    bh_sch_r = optim.lr_scheduler.CosineAnnealingLR(bh_opt_r, T_max=5)
    bh_scaler_r = torch.cuda.amp.GradScaler()
    train_full(resnet_bh, bh_train_loader, bh_val_loader,
               bh_opt_r, bh_sch_r, criterion,
               epochs=5, model_name='resnet50_bh_phase1',
               save_dir=MODELS_DIR, scaler=bh_scaler_r, device=device)
    resnet_bh.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, 'resnet50_bh_phase1_best.pth')))
    resnet_bh.fc = nn.Linear(resnet_bh.fc.in_features, 2)
    resnet_bh = resnet_bh.to(device)
    opt_r2 = optim.Adam(resnet_bh.parameters(), lr=0.0001)
    sch_r2 = optim.lr_scheduler.CosineAnnealingLR(opt_r2, T_max=15)
    scaler_r2 = torch.cuda.amp.GradScaler()
    train_full(resnet_bh, train_loader, val_loader,
               opt_r2, sch_r2, criterion,
               epochs=15, model_name='resnet50_bh',
               save_dir=MODELS_DIR, scaler=scaler_r2, device=device)
    resnet_bh.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, 'resnet50_bh_best.pth')))
    auc_r2, res_r2 = evaluate_slide_level(
        resnet_bh, test_df, val_transform, device, 'resnet50_breakhis_pretrain')
    all_results['resnet50_breakhis_pretrain'] = res_r2

    # model 3 resnet50 tcga-brca simclr pretrain then a2 fine-tune
    print("model 3 resnet50 tcga-brca simclr pretrain")
    resnet_simclr = timm.create_model(
        'hf-hub:1aurent/resnet50.tcga_brca_simclr',
        pretrained=True,
        num_classes=2
    )
    resnet_simclr = resnet_simclr.to(device)
    opt_r3 = optim.Adam(resnet_simclr.parameters(), lr=0.0001)
    sch_r3 = optim.lr_scheduler.CosineAnnealingLR(opt_r3, T_max=15)
    scaler_r3 = torch.cuda.amp.GradScaler()
    train_full(resnet_simclr, train_loader, val_loader,
               opt_r3, sch_r3, criterion,
               epochs=15, model_name='resnet50_simclr',
               save_dir=MODELS_DIR, scaler=scaler_r3, device=device)
    resnet_simclr.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, 'resnet50_simclr_best.pth')))
    auc_r3, res_r3 = evaluate_slide_level(
        resnet_simclr, test_df, val_transform, device, 'resnet50_simclr')
    all_results['resnet50_simclr'] = res_r3

    # model 4 efficientnet imagenet only then a2 fine-tune
    print("model 4 efficientnet imagenet")
    eff_imagenet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
    eff_imagenet = eff_imagenet.to(device)
    opt_e1 = optim.Adam(eff_imagenet.parameters(), lr=0.0001)
    sch_e1 = optim.lr_scheduler.CosineAnnealingLR(opt_e1, T_max=15)
    scaler_e1 = torch.cuda.amp.GradScaler()
    train_full(eff_imagenet, train_loader, val_loader,
               opt_e1, sch_e1, criterion,
               epochs=15, model_name='efficientnet_imagenet',
               save_dir=MODELS_DIR, scaler=scaler_e1, device=device)
    eff_imagenet.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, 'efficientnet_imagenet_best.pth')))
    auc_e1, res_e1 = evaluate_slide_level(
        eff_imagenet, test_df, val_transform, device, 'efficientnet_imagenet')
    all_results['efficientnet_imagenet'] = res_e1

    # model 5 efficientnet breakhis pretrain then a2 fine-tune
    print("model 5 efficientnet breakhis pretrain")
    eff_bh = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
    eff_bh = eff_bh.to(device)
    bh_opt_e = optim.Adam(eff_bh.parameters(), lr=0.0001)
    bh_sch_e = optim.lr_scheduler.CosineAnnealingLR(bh_opt_e, T_max=5)
    bh_scaler_e = torch.cuda.amp.GradScaler()
    train_full(eff_bh, bh_train_loader, bh_val_loader,
               bh_opt_e, bh_sch_e, criterion,
               epochs=5, model_name='efficientnet_bh_phase1',
               save_dir=MODELS_DIR, scaler=bh_scaler_e, device=device)
    eff_bh.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, 'efficientnet_bh_phase1_best.pth')))
    eff_bh.classifier = nn.Linear(eff_bh.classifier.in_features, 2)
    eff_bh = eff_bh.to(device)
    opt_e2 = optim.Adam(eff_bh.parameters(), lr=0.0001)
    sch_e2 = optim.lr_scheduler.CosineAnnealingLR(opt_e2, T_max=15)
    scaler_e2 = torch.cuda.amp.GradScaler()
    train_full(eff_bh, train_loader, val_loader,
               opt_e2, sch_e2, criterion,
               epochs=15, model_name='efficientnet_bh',
               save_dir=MODELS_DIR, scaler=scaler_e2, device=device)
    eff_bh.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, 'efficientnet_bh_best.pth')))
    auc_e2, res_e2 = evaluate_slide_level(
        eff_bh, test_df, val_transform, device, 'efficientnet_breakhis_pretrain')
    all_results['efficientnet_breakhis_pretrain'] = res_e2

    # save all results
    with open(os.path.join(MODELS_DIR, 'training_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print("all models done")
    print(f"resnet50 imagenet slide auc {round(auc_r1, 4)}")
    print(f"resnet50 breakhis pretrain slide auc {round(auc_r2, 4)}")
    print(f"resnet50 simclr pretrain slide auc {round(auc_r3, 4)}")
    print(f"efficientnet imagenet slide auc {round(auc_e1, 4)}")
    print(f"efficientnet breakhis pretrain slide auc {round(auc_e2, 4)}")
    print("results saved in " + MODELS_DIR)