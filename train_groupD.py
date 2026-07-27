import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
DATA_DIR = ''
CSV_PATH = ''
TEST_DIR = ''
TEST_CSV_PATH = ''
OUTPUT_DIR = ''
N_FOLDS = 5
BATCH_SIZE = 8
LR = 0.0001
EPOCHS = 100
PATIENCE = 20
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ALL_MODALITIES = ['T2W', 'DWI', 'PET', 'CT']
EMBED_DIM = 256
N_HEADS = 4
N_LAYERS = 4
DROPOUT = 0.1
ABLATION_VARIANTS = {'full': ['T2W', 'DWI', 'PET', 'CT'], 'no_pet': ['T2W', 'DWI', 'CT'], 'no_ct': ['T2W', 'DWI', 'PET'], 'no_dwi': ['T2W', 'PET', 'CT'], 'no_t2w': ['DWI', 'PET', 'CT'], 'mri_only': ['T2W', 'DWI'], 'petct_only': ['PET', 'CT']}

class ProstateDataset(Dataset):

    def __init__(self, data_dir, csv_path, modalities, indices=None, augment=False):
        df = pd.read_csv(csv_path)
        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)
        self.labels = df['label'].values.astype(np.int64)
        self.modalities = modalities
        self.augment = augment
        self.data = {}
        for mod in modalities:
            arr = np.load(os.path.join(data_dir, f'{mod}.npy'))
            if indices is not None:
                arr = arr[indices]
            self.data[mod] = arr.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        vols = {}
        for mod in self.modalities:
            vol = torch.tensor(self.data[mod][idx]).unsqueeze(0)
            if self.augment:
                for dim in [1, 2, 3]:
                    if torch.rand(1) > 0.5:
                        vol = torch.flip(vol, dims=[dim])
            vols[mod] = vol
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return (vols, label)

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

    def __init__(self, modalities):
        super().__init__()
        self.modalities = modalities
        n_mod = len(modalities)
        self.encoders = nn.ModuleDict({mod: ResNet18_3D() for mod in modalities})
        self.proj = nn.ModuleDict({mod: nn.Sequential(nn.Linear(512, EMBED_DIM), nn.LayerNorm(EMBED_DIM)) for mod in modalities})
        self.cls_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_mod + 1, EMBED_DIM))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=N_HEADS, dim_feedforward=EMBED_DIM * 4, dropout=DROPOUT, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS, norm=nn.LayerNorm(EMBED_DIM))
        self.head = nn.Sequential(nn.LayerNorm(EMBED_DIM), nn.Dropout(0.2), nn.Linear(EMBED_DIM, 64), nn.GELU(), nn.Linear(64, 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm3d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, vols):
        B = next(iter(vols.values())).shape[0]
        tokens = []
        for mod in self.modalities:
            feat = self.encoders[mod](vols[mod])
            token = self.proj[mod](feat)
            tokens.append(token.unsqueeze(1))
        mod_tokens = torch.cat(tokens, dim=1)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, mod_tokens], dim=1)
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.transformer(x)
        cls_out = x[:, 0, :]
        return self.head(cls_out)

    def get_attention_weights(self, vols):
        B = next(iter(vols.values())).shape[0]
        tokens = []
        for mod in self.modalities:
            feat = self.encoders[mod](vols[mod])
            token = self.proj[mod](feat)
            tokens.append(token.unsqueeze(1))
        mod_tokens = torch.cat(tokens, dim=1)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, mod_tokens], dim=1)
        x = x + self.pos_embed[:, :x.shape[1], :]
        attn_weights = []

        def hook(module, input, output):
            pass
        layer = self.transformer.layers[0]
        with torch.no_grad():
            attn_out, attn_w = layer.self_attn(x, x, x, need_weights=True, average_attn_weights=False)
        return attn_w

def make_loader(dataset, shuffle=True, balance=True):
    if balance and shuffle:
        labels = dataset.labels
        counts = np.bincount(labels)
        weights = 1.0 / counts[labels]
        sampler = WeightedRandomSampler(weights=torch.tensor(weights, dtype=torch.float), num_samples=len(labels), replacement=True)
        return DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, pin_memory=True)

def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    auc = roc_auc_score(labels, probs)
    f1 = f1_score(labels, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn + 1e-08)
    spec = tn / (tn + fp + 1e-08)
    return {'AUC': auc, 'F1': f1, 'Sens': sens, 'Spec': spec}

def train_one_epoch(model, loader, optimizer, criterion, scheduler=None):
    model.train()
    total_loss = 0.0
    for vols, labels in loader:
        vols = {m: v.to(DEVICE) for m, v in vols.items()}
        labels = labels.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(model(vols), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = ([], [])
    for vols, labels in loader:
        vols = {m: v.to(DEVICE) for m, v in vols.items()}
        probs = torch.sigmoid(model(vols)).cpu().numpy().flatten()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())
    return (np.array(all_labels), np.array(all_probs))

def run_cv(variant_name, modalities, df, output_dir):
    labels = df['label'].values
    N = len(labels)
    n_pos = int(labels.sum())
    n_neg = N - n_pos
    POS_WEIGHT_SCALE = 1.5
    pos_weight = torch.tensor([n_neg / n_pos * POS_WEIGHT_SCALE]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(N), labels)):
        print(f'\n  Fold {fold + 1}/{N_FOLDS}  train={len(train_idx)} val={len(val_idx)}')
        train_ds = ProstateDataset(DATA_DIR, CSV_PATH, modalities, indices=train_idx, augment=True)
        val_ds = ProstateDataset(DATA_DIR, CSV_PATH, modalities, indices=val_idx, augment=False)
        train_loader = make_loader(train_ds, shuffle=True, balance=True)
        val_loader = make_loader(val_ds, shuffle=False, balance=False)
        model = CrossModalTransformer(modalities).to(DEVICE)
        n_params = sum((p.numel() for p in model.parameters()))
        print(f'    Params: {n_params:,}')
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0001)
        total_steps = EPOCHS * len(train_loader)
        warmup_steps = total_steps // 10
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps), optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=LR * 0.01)], milestones=[warmup_steps])
        best_auc = 0.0
        patience_cnt = 0
        best_path = os.path.join(output_dir, f'fold{fold + 1}_{variant_name}_best.pt')
        for epoch in range(EPOCHS):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, scheduler)
            val_labels, val_probs = evaluate(model, val_loader)
            metrics = compute_metrics(val_labels, val_probs)
            if metrics['AUC'] > best_auc:
                best_auc = metrics['AUC']
                patience_cnt = 0
                torch.save(model.state_dict(), best_path)
            else:
                patience_cnt += 1
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch + 1:3d}  loss={loss:.4f}  AUC={metrics['AUC']:.4f}  F1={metrics['F1']:.4f}  [best={best_auc:.4f}]")
            if patience_cnt >= PATIENCE:
                print(f'    Early stop at epoch {epoch + 1}')
                break
        model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=False))
        val_labels, val_probs = evaluate(model, val_loader)
        metrics = compute_metrics(val_labels, val_probs)
        print(f"  Fold {fold + 1} best -> AUC={metrics['AUC']:.4f}  F1={metrics['F1']:.4f}  Sens={metrics['Sens']:.4f}  Spec={metrics['Spec']:.4f}")
        fold_metrics.append({'variant': variant_name, 'modalities': '+'.join(modalities), 'fold': fold + 1, **metrics})
    return fold_metrics

def run_external_test(variant_name, modalities, output_dir):
    test_ds = ProstateDataset(TEST_DIR, TEST_CSV_PATH, modalities, indices=None, augment=False)
    test_loader = make_loader(test_ds, shuffle=False, balance=False)
    all_probs = []
    for fold in range(1, N_FOLDS + 1):
        mp = os.path.join(output_dir, f'fold{fold}_{variant_name}_best.pt')
        m = CrossModalTransformer(modalities).to(DEVICE)
        m.load_state_dict(torch.load(mp, map_location=DEVICE, weights_only=False))
        _, probs = evaluate(m, test_loader)
        all_probs.append(probs)
    ens = np.mean(all_probs, axis=0)
    metrics = compute_metrics(test_ds.labels, ens)
    return metrics

def run_training():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"\n{'=' * 55}")
    print(f'Group D: Cross-modal Transformer')
    print(f'  Encoder   : ResNet-18 3D (per modality)')
    print(f'  Embed dim : {EMBED_DIM}   Heads: {N_HEADS}   Layers: {N_LAYERS}')
    print(f'Device  : {DEVICE}')
    print(f'Folds   : {N_FOLDS}   Epochs: {EPOCHS}   Patience: {PATIENCE}')
    print(f'Batch   : {BATCH_SIZE}')
    print(f'Variants: {list(ABLATION_VARIANTS.keys())}')
    print(f"{'=' * 55}\n")
    df = pd.read_csv(CSV_PATH)
    all_cv_results = []
    all_ext_results = []
    for variant_name, modalities in ABLATION_VARIANTS.items():
        print(f"\n{'-' * 50}")
        print(f'Variant: {variant_name.upper()}  modalities: {modalities}')
        print(f"{'-' * 50}")
        fold_metrics = run_cv(variant_name, modalities, df, OUTPUT_DIR)
        all_cv_results.extend(fold_metrics)
        print(f'\n  {variant_name} 5-fold summary:')
        for metric in ['AUC', 'F1', 'Sens', 'Spec']:
            vals = [m[metric] for m in fold_metrics]
            print(f'    {metric:4s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}')
        ext = run_external_test(variant_name, modalities, OUTPUT_DIR)
        print(f"  {variant_name} external test: AUC={ext['AUC']:.4f}  F1={ext['F1']:.4f}  Sens={ext['Sens']:.4f}  Spec={ext['Spec']:.4f}")
        all_ext_results.append({'variant': variant_name, 'modalities': '+'.join(modalities), **ext})
    cv_df = pd.DataFrame(all_cv_results)
    cv_df.to_csv(os.path.join(OUTPUT_DIR, 'cv_results.csv'), index=False)
    summary_rows = []
    for variant_name in ABLATION_VARIANTS:
        sub = cv_df[cv_df['variant'] == variant_name]
        row = {'variant': variant_name, 'modalities': sub['modalities'].iloc[0]}
        for metric in ['AUC', 'F1', 'Sens', 'Spec']:
            m, s = (sub[metric].mean(), sub[metric].std())
            row[f'{metric}_cv'] = f'{m:.4f}+/-{s:.4f}'
        summary_rows.append(row)
    ext_df = pd.DataFrame(all_ext_results)
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.merge(ext_df[['variant', 'AUC', 'F1', 'Sens', 'Spec']].rename(columns={'AUC': 'AUC_ext', 'F1': 'F1_ext', 'Sens': 'Sens_ext', 'Spec': 'Spec_ext'}), on='variant')
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'summary.csv'), index=False)
    ext_df.to_csv(os.path.join(OUTPUT_DIR, 'external_test_results.csv'), index=False)
    print(f"\n{'=' * 55}")
    print('Group D final summary')
    print(f"{'=' * 55}")
    print(f"{'Variant':<12} {'Modalities':<20} {'CV AUC':>12} {'Ext AUC':>10}")
    print(f"{'-' * 55}")
    for _, row in summary_df.iterrows():
        print(f"{row['variant']:<12} {row['modalities']:<20} {row['AUC_cv']:>12} {row['AUC_ext']:>10.4f}")
    print(f'\nAll results saved to: {OUTPUT_DIR}')
    print(f"{'=' * 55}\n")
if __name__ == '__main__':
    run_training()
