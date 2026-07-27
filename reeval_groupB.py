import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve
import warnings
warnings.filterwarnings('ignore')
TRAIN_DATA_DIR = ''
TRAIN_CSV = ''
TEST_DATA_DIR = ''
TEST_CSV = ''
DIR_MEDICALNET = ''
DIR_DINOV2 = ''
OUTPUT_ALL = ''
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_FOLDS = 5
SEED = 42
BATCH_SIZE = 16
ALL_MODALITIES = ['T2W', 'DWI', 'PET', 'CT']
FUSIONS = ['inter', 'early', 'late']
IMG_SIZE = 224

def volume_to_2d(vol_3d):
    z_c = vol_3d.shape[3] // 2
    slices = vol_3d[0, :, :, z_c - 1:z_c + 2].permute(2, 0, 1).unsqueeze(0)
    slices = torch.nn.functional.interpolate(slices.float(), size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False).squeeze(0)
    for c in range(3):
        ch = slices[c]
        mn, mx = (ch.min(), ch.max())
        if mx - mn > 1e-08:
            slices[c] = (ch - mn) / (mx - mn)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (slices - mean) / std

class ProstateDataset(Dataset):

    def __init__(self, data_dir, csv_path, indices=None, mode='3d'):
        df = pd.read_csv(csv_path)
        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)
        self.labels = df['label'].values.astype(np.int64)
        self.mode = mode
        self.data = {}
        for mod in ALL_MODALITIES:
            arr = np.load(os.path.join(data_dir, f'{mod}.npy'))
            if indices is not None:
                arr = arr[indices]
            self.data[mod] = arr.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        vols = {}
        for mod in ALL_MODALITIES:
            vol = torch.tensor(self.data[mod][idx]).unsqueeze(0)
            if self.mode == '2d':
                vol = volume_to_2d(vol)
            vols[mod] = vol
        return (vols, torch.tensor(self.labels[idx], dtype=torch.float32))

def make_loader(ds):
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

def compute_optimal(labels, probs):
    auc = roc_auc_score(labels, probs)
    fpr, tpr, thresh = roc_curve(labels, probs)
    best_idx = np.argmax(tpr - fpr)
    best_thresh = float(thresh[best_idx])
    preds = (probs >= best_thresh).astype(int)
    f1 = f1_score(labels, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn + 1e-08)
    spec = tn / (tn + fp + 1e-08)
    ppv = tp / (tp + fp + 1e-08)
    npv = tn / (tn + fn + 1e-08)
    return {'AUC': round(auc, 4), 'Sens': round(sens, 4), 'Spec': round(spec, 4), 'F1': round(f1, 4), 'PPV': round(ppv, 4), 'NPV': round(npv, 4), 'thresh': round(best_thresh, 4)}

@torch.no_grad()
def get_probs(model, loader):
    model.eval()
    all_probs, all_labels = ([], [])
    for vols, labels in loader:
        vols = {m: v.to(DEVICE) for m, v in vols.items()}
        probs = torch.sigmoid(model(vols)).cpu().numpy().flatten()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())
    return (np.array(all_labels), np.array(all_probs))

def load_pt(path, key_remap=None):
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    state = {k: v.float() if v.dtype == torch.float16 else v for k, v in state.items()}
    if key_remap is not None:
        old, new = key_remap
        state = {new + k[len(old):] if k.startswith(old) else k: v for k, v in state.items()}
    return state

class MLPHead(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)

class MedBasicBlock(nn.Module):

    def __init__(self, in_c, out_c, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_c)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)

class MedResNet10(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 64, 7, stride=(2, 2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)
        self.layer1 = self._make(64, 64, 1)
        self.layer2 = self._make(64, 128, 2)
        self.layer3 = self._make(128, 256, 2)
        self.layer4 = self._make(256, 512, 2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)

    def _make(self, in_c, out_c, stride):
        ds = None
        if stride != 1 or in_c != out_c:
            ds = nn.Sequential(nn.Conv3d(in_c, out_c, 1, stride=stride, bias=False), nn.BatchNorm3d(out_c))
        return nn.Sequential(MedBasicBlock(in_c, out_c, stride, ds))

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.avgpool(x).flatten(1)

class MedInter(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([MedResNet10() for _ in ALL_MODALITIES])
        self.mlp = MLPHead(512 * 4)

    def forward(self, vols):
        feats = [enc(vols[m]) for enc, m in zip(self.encoders, ALL_MODALITIES)]
        return self.mlp(torch.cat(feats, dim=1))

class MedLate(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([MedResNet10() for _ in ALL_MODALITIES])
        self.heads = nn.ModuleList([MLPHead(512) for _ in ALL_MODALITIES])

    def forward(self, vols):
        logits = [head(enc(vols[m])) for enc, head, m in zip(self.encoders, self.heads, ALL_MODALITIES)]
        return torch.stack(logits, dim=0).mean(dim=0)

class MedEarly(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = MedResNet10()
        old = self.encoder.conv1
        new = nn.Conv3d(4, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight = nn.Parameter(old.weight.repeat(1, 4, 1, 1, 1) / 4)
        self.encoder.conv1 = new
        self.mlp = MLPHead(512)

    def forward(self, vols):
        x = torch.cat([vols[m] for m in ALL_MODALITIES], dim=1)
        return self.mlp(self.encoder(x))
MED_MODELS = {'inter': MedInter, 'late': MedLate, 'early': MedEarly}

class DinoEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        self.dino = AutoModel.from_pretrained('facebook/dinov2-small', cache_dir=None)
        for p in self.dino.parameters():
            p.requires_grad = False

    def encode(self, x):
        return self.dino(pixel_values=x).last_hidden_state[:, 0, :]

class DinoInter(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = DinoEncoder()
        self.mlp = MLPHead(384 * 4)

    def forward(self, vols):
        feats = [self.encoder.encode(vols[m]) for m in ALL_MODALITIES]
        return self.mlp(torch.cat(feats, dim=1))

class DinoLate(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = DinoEncoder()
        self.heads = nn.ModuleList([MLPHead(384) for _ in ALL_MODALITIES])

    def forward(self, vols):
        logits = [head(self.encoder.encode(vols[m])) for head, m in zip(self.heads, ALL_MODALITIES)]
        return torch.stack(logits, dim=0).mean(dim=0)

class DinoEarly(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = DinoEncoder()
        self.proj = nn.Conv2d(12, 3, 1, bias=False)
        self.mlp = MLPHead(384)

    def forward(self, vols):
        x = torch.cat([vols[m] for m in ALL_MODALITIES], dim=1)
        x = self.proj(x)
        return self.mlp(self.encoder.encode(x))
DINO_MODELS = {'inter': DinoInter, 'late': DinoLate, 'early': DinoEarly}
DINO_KEY_REMAP = ('encoder.model.', 'encoder.dino.')

def eval_group(group_label, model_dir, model_factory, input_mode, key_remap=None, fusions=FUSIONS):
    out_dir = os.path.join(model_dir, 'reeval')
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(TRAIN_CSV)
    labels = df['label'].values
    N = len(labels)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.arange(N), labels))
    all_cv = []
    all_ext = []
    for fusion in fusions:
        print(f'\n  [{group_label}] fusion={fusion}')
        fold_metrics = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            fold = fold_idx + 1
            path = os.path.join(model_dir, f'fold{fold}_{fusion}_best.pt')
            if not os.path.exists(path):
                print(f'    Fold {fold}: [SKIP]')
                continue
            model = model_factory[fusion]().to(DEVICE)
            state = load_pt(path, key_remap=key_remap)
            model.load_state_dict(state, strict=True)
            val_ds = ProstateDataset(TRAIN_DATA_DIR, TRAIN_CSV, indices=val_idx, mode=input_mode)
            lbl, probs = get_probs(model, make_loader(val_ds))
            m = compute_optimal(lbl, probs)
            print(f"    Fold {fold}: AUC={m['AUC']:.4f}  Sens={m['Sens']:.4f}  Spec={m['Spec']:.4f}  thresh={m['thresh']:.3f}")
            fold_metrics.append(m)
            all_cv.append({'group': group_label, 'fusion': fusion, 'fold': fold, **m})
        if not fold_metrics:
            continue
        print(f'  {fusion} 5-fold (optimal):')
        for metric in ['AUC', 'Sens', 'Spec', 'F1']:
            vals = [x[metric] for x in fold_metrics]
            print(f'    {metric}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}')
        test_ds = ProstateDataset(TEST_DATA_DIR, TEST_CSV, indices=None, mode=input_mode)
        all_probs = []
        for fold_idx in range(N_FOLDS):
            fold = fold_idx + 1
            path = os.path.join(model_dir, f'fold{fold}_{fusion}_best.pt')
            if not os.path.exists(path):
                continue
            model = model_factory[fusion]().to(DEVICE)
            model.load_state_dict(load_pt(path, key_remap=key_remap), strict=True)
            _, probs = get_probs(model, make_loader(test_ds))
            all_probs.append(probs)
        if all_probs:
            ens = np.mean(all_probs, axis=0)
            ext_m = compute_optimal(test_ds.labels, ens)
            print(f"  {fusion} external: AUC={ext_m['AUC']:.4f}  Sens={ext_m['Sens']:.4f}  Spec={ext_m['Spec']:.4f}")
            all_ext.append({'group': group_label, 'fusion': fusion, **ext_m})
    if all_cv:
        cv_df = pd.DataFrame(all_cv)
        rows = []
        for fusion in fusions:
            sub = cv_df[cv_df['fusion'] == fusion]
            if sub.empty:
                continue
            row = {'group': group_label, 'fusion': fusion}
            for metric in ['AUC', 'Sens', 'Spec', 'F1', 'PPV', 'NPV']:
                if metric in sub.columns:
                    row[metric] = f'{sub[metric].mean():.4f}+/-{sub[metric].std():.4f}'
            rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'cv_summary_optimal.csv'), index=False)
    if all_ext:
        pd.DataFrame(all_ext).to_csv(os.path.join(out_dir, 'external_optimal.csv'), index=False)
    return (all_cv, all_ext)

def run_reeval():
    os.makedirs(OUTPUT_ALL, exist_ok=True)
    torch.manual_seed(SEED)
    print(f"\n{'=' * 60}")
    print(f'Group B re-evaluation - optimal threshold (Youden Index)')
    print(f'Device: {DEVICE}')
    print(f"{'=' * 60}")
    all_cv = []
    all_ext = []
    print(f"\n{'-' * 60}")
    print('groupB_MedicalNet')
    print(f"{'-' * 60}")
    cv, ext = eval_group(group_label='GroupB_MedicalNet', model_dir=DIR_MEDICALNET, model_factory=MED_MODELS, input_mode='3d', key_remap=None)
    all_cv.extend(cv)
    all_ext.extend(ext)
    print(f"\n{'-' * 60}")
    print('groupB_DINOv2')
    print(f"{'-' * 60}")
    sample_path = os.path.join(DIR_DINOV2, 'fold1_inter_best.pt')
    if os.path.exists(sample_path):
        sample = torch.load(sample_path, map_location='cpu', weights_only=False)
        sample_keys = [k for k in list(sample.keys())[:3]]
        print(f'  Checkpoint key sample: {sample_keys}')
        if any((k.startswith('encoder.model.') for k in sample_keys)):
            key_remap = ('encoder.model.', 'encoder.dino.')
            print(f'  Key remap: encoder.model.* -> encoder.dino.*')
        else:
            key_remap = None
            print(f'  No key remap needed')
    else:
        key_remap = ('encoder.model.', 'encoder.dino.')
    try:
        cv, ext = eval_group(group_label='GroupB_DINOv2', model_dir=DIR_DINOV2, model_factory=DINO_MODELS, input_mode='2d', key_remap=key_remap)
        all_cv.extend(cv)
        all_ext.extend(ext)
    except Exception as e:
        print(f'  [ERROR] DINOv2 failed: {e}')
    summary_rows = []
    if all_cv:
        cv_df = pd.DataFrame(all_cv)
        for (group, fusion), sub in cv_df.groupby(['group', 'fusion']):
            row = {'group': group, 'fusion': fusion}
            for metric in ['AUC', 'Sens', 'Spec', 'F1', 'PPV', 'NPV']:
                if metric in sub.columns:
                    row[metric] = f'{sub[metric].mean():.4f}+/-{sub[metric].std():.4f}'
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_ALL, 'all_cv_summary.csv'), index=False)
    if all_ext:
        pd.DataFrame(all_ext).to_csv(os.path.join(OUTPUT_ALL, 'all_external.csv'), index=False)
    print(f"\n{'=' * 60}")
    print('FINAL SUMMARY - Group B optimal threshold')
    print(f"{'-' * 60}")
    print(f"{'Group':<22} {'Fusion':<8} {'CV AUC':>12} {'CV Sens':>10} {'CV Spec':>10} {'Ext AUC':>10}")
    print(f"{'-' * 60}")
    ext_lookup = {(r['group'], r['fusion']): r for r in all_ext}
    for row in summary_rows:
        ext = ext_lookup.get((row['group'], row['fusion']), {})
        print(f"{row['group']:<22} {row['fusion']:<8} {row['AUC']:>12} {row['Sens']:>10} {row['Spec']:>10} {ext.get('AUC', 'N/A'):>10}")
    print(f'\nOutputs saved to: {OUTPUT_ALL}')
    print(f"{'=' * 60}\n")
if __name__ == '__main__':
    run_reeval()
