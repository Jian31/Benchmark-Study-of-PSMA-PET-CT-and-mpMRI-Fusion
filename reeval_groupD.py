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
MODEL_DIR = ''
OUTPUT_DIR = ''
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_FOLDS = 5
SEED = 42
BATCH_SIZE = 8
ABLATION_VARIANTS = {'full': ['T2W', 'DWI', 'PET', 'CT'], 'no_pet': ['T2W', 'DWI', 'CT'], 'no_ct': ['T2W', 'DWI', 'PET'], 'no_dwi': ['T2W', 'PET', 'CT'], 'no_t2w': ['DWI', 'PET', 'CT'], 'mri_only': ['T2W', 'DWI'], 'petct_only': ['PET', 'CT']}

class ProstateDataset(Dataset):

    def __init__(self, data_dir, csv_path, modalities, indices=None):
        df = pd.read_csv(csv_path)
        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)
        self.labels = df['label'].values.astype(np.int64)
        self.modalities = modalities
        self.data = {}
        for mod in modalities:
            arr = np.load(os.path.join(data_dir, f'{mod}.npy'))
            if indices is not None:
                arr = arr[indices]
            self.data[mod] = arr.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        vols = {mod: torch.tensor(self.data[mod][idx]).unsqueeze(0) for mod in self.modalities}
        return (vols, torch.tensor(self.labels[idx], dtype=torch.float32))

def make_loader(dataset):
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

class BasicBlock3D(nn.Module):

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
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

class ResNet18_3D(nn.Module):

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, 64, 7, stride=2, padding=3, bias=False), nn.BatchNorm3d(64), nn.ReLU(inplace=True), nn.MaxPool3d(3, stride=2, padding=1))
        self.layer1 = self._make(64, 64, 1)
        self.layer2 = self._make(64, 128, 2)
        self.layer3 = self._make(128, 256, 2)
        self.layer4 = self._make(256, 512, 2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)

    def _make(self, in_c, out_c, stride):
        ds = None
        if stride != 1 or in_c != out_c:
            ds = nn.Sequential(nn.Conv3d(in_c, out_c, 1, stride=stride, bias=False), nn.BatchNorm3d(out_c))
        return nn.Sequential(BasicBlock3D(in_c, out_c, stride, ds))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.avgpool(x).flatten(1)

class CrossModalTransformer(nn.Module):
    EMBED_DIM = 256
    N_HEADS = 4
    N_LAYERS = 4
    DROPOUT = 0.1

    def __init__(self, modalities):
        super().__init__()
        self.modalities = modalities
        n_mod = len(modalities)
        D = self.EMBED_DIM
        self.encoders = nn.ModuleDict({m: ResNet18_3D() for m in modalities})
        self.proj = nn.ModuleDict({m: nn.Sequential(nn.Linear(512, D), nn.LayerNorm(D)) for m in modalities})
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_mod + 1, D))
        enc_layer = nn.TransformerEncoderLayer(d_model=D, nhead=self.N_HEADS, dim_feedforward=D * 4, dropout=self.DROPOUT, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=self.N_LAYERS, norm=nn.LayerNorm(D))
        self.head = nn.Sequential(nn.LayerNorm(D), nn.Dropout(0.2), nn.Linear(D, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, vols):
        B = next(iter(vols.values())).shape[0]
        tokens = [self.proj[m](self.encoders[m](vols[m])).unsqueeze(1) for m in self.modalities]
        x = torch.cat([self.cls_token.expand(B, -1, -1)] + tokens, dim=1)
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.transformer(x)
        return self.head(x[:, 0, :])

def compute_metrics_fixed(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    auc = roc_auc_score(labels, probs)
    f1 = f1_score(labels, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {'AUC': round(auc, 4), 'F1': round(f1, 4), 'Sens': round(tp / (tp + fn + 1e-08), 4), 'Spec': round(tn / (tn + fp + 1e-08), 4), 'threshold': 0.5}

def compute_metrics_optimal(labels, probs):
    auc = roc_auc_score(labels, probs)
    fpr, tpr, thresh = roc_curve(labels, probs)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_thresh = float(thresh[best_idx])
    preds = (probs >= best_thresh).astype(int)
    f1 = f1_score(labels, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn + 1e-08)
    spec = tn / (tn + fp + 1e-08)
    ppv = tp / (tp + fp + 1e-08)
    npv = tn / (tn + fn + 1e-08)
    return {'AUC': round(auc, 4), 'F1': round(f1, 4), 'Sens': round(sens, 4), 'Spec': round(spec, 4), 'PPV': round(ppv, 4), 'NPV': round(npv, 4), 'threshold': round(best_thresh, 4)}

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

def load_model(variant, modalities, fold):
    path = os.path.join(MODEL_DIR, f'fold{fold}_{variant}_best.pt')
    if not os.path.exists(path):
        return (None, path)
    model = CrossModalTransformer(modalities).to(DEVICE)
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    state = {k: v.float() if v.dtype == torch.float16 else v for k, v in state.items()}
    model.load_state_dict(state)
    return (model, path)

def run_reeval():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    print(f"\n{'=' * 60}")
    print(f'Group D re-evaluation - optimal threshold (Youden Index)')
    print(f'Device : {DEVICE}')
    print(f'Models : {MODEL_DIR}')
    print(f'Output : {OUTPUT_DIR}')
    print(f"{'=' * 60}\n")
    df = pd.read_csv(TRAIN_CSV)
    labels = df['label'].values
    N = len(labels)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.arange(N), labels))
    all_cv_rows = []
    all_ext_rows = []
    for variant, modalities in ABLATION_VARIANTS.items():
        print(f"\n{'-' * 50}")
        print(f"Variant: {variant}  [{'+'.join(modalities)}]")
        print(f"{'-' * 50}")
        fold_metrics_opt = []
        fold_metrics_fixed = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            fold = fold_idx + 1
            model, path = load_model(variant, modalities, fold)
            if model is None:
                print(f'  Fold {fold}: [SKIP] not found - {path}')
                continue
            val_ds = ProstateDataset(TRAIN_DATA_DIR, TRAIN_CSV, modalities, indices=val_idx)
            loader = make_loader(val_ds)
            val_labels, val_probs = get_probs(model, loader)
            m_opt = compute_metrics_optimal(val_labels, val_probs)
            m_fixed = compute_metrics_fixed(val_labels, val_probs)
            print(f"  Fold {fold}  AUC={m_opt['AUC']:.4f}  Sens={m_opt['Sens']:.4f}  Spec={m_opt['Spec']:.4f}  F1={m_opt['F1']:.4f}  thresh={m_opt['threshold']:.3f}  [fixed0.5: Sens={m_fixed['Sens']:.4f} Spec={m_fixed['Spec']:.4f}]")
            fold_metrics_opt.append(m_opt)
            fold_metrics_fixed.append(m_fixed)
            all_cv_rows.append({'variant': variant, 'modalities': '+'.join(modalities), 'fold': fold, **{f'{k}_opt': v for k, v in m_opt.items()}, **{f'{k}_fixed': v for k, v in m_fixed.items()}})
        if not fold_metrics_opt:
            print(f'  No folds found for {variant}, skipping.')
            continue
        print(f'\n  {variant} 5-fold (optimal threshold):')
        for metric in ['AUC', 'Sens', 'Spec', 'F1', 'PPV', 'NPV']:
            vals = [m[metric] for m in fold_metrics_opt]
            print(f'    {metric:4s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}')
        print(f'\n  {variant} 5-fold (fixed threshold=0.5):')
        for metric in ['Sens', 'Spec', 'F1']:
            vals = [m[metric] for m in fold_metrics_fixed]
            print(f'    {metric:4s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}')
        test_ds = ProstateDataset(TEST_DATA_DIR, TEST_CSV, modalities, indices=None)
        test_loader = make_loader(test_ds)
        all_fold_probs = []
        for fold_idx in range(N_FOLDS):
            fold = fold_idx + 1
            model, path = load_model(variant, modalities, fold)
            if model is None:
                continue
            _, probs = get_probs(model, test_loader)
            all_fold_probs.append(probs)
        if all_fold_probs:
            ens_probs = np.mean(all_fold_probs, axis=0)
            test_labels = test_ds.labels
            ext_opt = compute_metrics_optimal(test_labels, ens_probs)
            ext_fixed = compute_metrics_fixed(test_labels, ens_probs)
            print(f"\n  {variant} external (optimal):  AUC={ext_opt['AUC']:.4f}  Sens={ext_opt['Sens']:.4f}  Spec={ext_opt['Spec']:.4f}  F1={ext_opt['F1']:.4f}  thresh={ext_opt['threshold']:.3f}")
            print(f"  {variant} external (fixed 0.5): AUC={ext_fixed['AUC']:.4f}  Sens={ext_fixed['Sens']:.4f}  Spec={ext_fixed['Spec']:.4f}  F1={ext_fixed['F1']:.4f}")
            all_ext_rows.append({'variant': variant, 'modalities': '+'.join(modalities), **{f'{k}_opt': v for k, v in ext_opt.items()}, **{f'{k}_fixed': v for k, v in ext_fixed.items()}})
    cv_df = pd.DataFrame(all_cv_rows)
    cv_df.to_csv(os.path.join(OUTPUT_DIR, 'cv_optimal.csv'), index=False)
    summary_rows = []
    for variant in ABLATION_VARIANTS:
        sub = cv_df[cv_df['variant'] == variant]
        if sub.empty:
            continue
        row = {'variant': variant, 'modalities': sub['modalities'].iloc[0]}
        for metric in ['AUC', 'Sens', 'Spec', 'F1', 'PPV', 'NPV']:
            col = f'{metric}_opt'
            if col in sub.columns:
                m, s = (sub[col].mean(), sub[col].std())
                row[f'{metric}'] = f'{m:.4f}+/-{s:.4f}'
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'cv_summary_optimal.csv'), index=False)
    ext_df = pd.DataFrame(all_ext_rows)
    ext_df.to_csv(os.path.join(OUTPUT_DIR, 'external_optimal.csv'), index=False)
    print(f"\n{'=' * 60}")
    print('FINAL SUMMARY - optimal threshold (Youden Index)')
    print(f"{'-' * 60}")
    print(f"{'Variant':<12} {'Modalities':<22} {'CV AUC':>10} {'CV Sens':>9} {'CV Spec':>9} {'Ext AUC':>9} {'Ext Sens':>9} {'Ext Spec':>9}")
    print(f"{'-' * 60}")
    for _, row in summary_df.iterrows():
        ext_sub = ext_df[ext_df['variant'] == row['variant']]
        if ext_sub.empty:
            continue
        e = ext_sub.iloc[0]
        print(f"{row['variant']:<12} {row['modalities']:<22} {row['AUC']:>10} {row['Sens']:>9} {row['Spec']:>9} {e['AUC_opt']:>9.4f} {e['Sens_opt']:>9.4f} {e['Spec_opt']:>9.4f}")
    print(f'\nFiles saved to: {OUTPUT_DIR}')
    print(f'  cv_optimal.csv          - per-fold detail')
    print(f'  cv_summary_optimal.csv  - mean+/-std, use for paper Table 4')
    print(f'  external_optimal.csv    - external test, use for paper Table 3')
    print(f"{'=' * 60}\n")
if __name__ == '__main__':
    run_reeval()
