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
MEDICALNET_WEIGHTS = ''
OUTPUT_DIR = ''
N_FOLDS = 5
BATCH_SIZE = 16
LR = 0.001
EPOCHS = 60
PATIENCE = 12
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODALITIES = ['T2W', 'DWI', 'PET', 'CT']
FUSIONS = ['inter', 'early', 'late']

class ProstateDataset(Dataset):

    def __init__(self, data_dir, csv_path, indices=None, augment=False):
        df = pd.read_csv(csv_path)
        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)
        self.labels = df['label'].values.astype(np.int64)
        self.augment = augment
        self.data = {}
        for mod in MODALITIES:
            arr = np.load(os.path.join(data_dir, f'{mod}.npy'))
            if indices is not None:
                arr = arr[indices]
            self.data[mod] = arr.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        vols = {}
        for mod in MODALITIES:
            vol = torch.tensor(self.data[mod][idx]).unsqueeze(0)
            if self.augment:
                for dim in [1, 2, 3]:
                    if torch.rand(1) > 0.5:
                        vol = torch.flip(vol, dims=[dim])
            vols[mod] = vol
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return (vols, label)

class BasicBlock(nn.Module):

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

class ResNet10_3D(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _make_layer(self, in_c, out_c, stride):
        downsample = None
        if stride != 1 or in_c != out_c:
            downsample = nn.Sequential(nn.Conv3d(in_c, out_c, 1, stride=stride, bias=False), nn.BatchNorm3d(out_c))
        return nn.Sequential(BasicBlock(in_c, out_c, stride, downsample))

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.avgpool(x).flatten(1)

def load_medicalnet_weights(model, weights_path):
    if not os.path.exists(weights_path):
        print(f'  [WARNING] Weights not found: {weights_path}')
        print(f'  Running with random initialization.')
        return model
    ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    missing, _ = model.load_state_dict(state_dict, strict=False)
    loaded = len(state_dict) - len(missing)
    print(f'  MedicalNet weights loaded: {loaded}/{len(state_dict)} layers')
    if missing:
        print(f'  Missing keys: {len(missing)}')
    return model

def freeze_all(encoder):
    for p in encoder.parameters():
        p.requires_grad = False

class MLPHead(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)

class IntermediateFusionModel(nn.Module):

    def __init__(self, base_encoder):
        super().__init__()
        self.encoders = nn.ModuleList([ResNet10_3D() for _ in MODALITIES])
        for enc in self.encoders:
            enc.load_state_dict(base_encoder.state_dict())
            freeze_all(enc)
        self.mlp = MLPHead(512 * len(MODALITIES))

    def forward(self, vols):
        feats = [enc(vols[mod]) for enc, mod in zip(self.encoders, MODALITIES)]
        return self.mlp(torch.cat(feats, dim=1))

class EarlyFusionModel(nn.Module):

    def __init__(self, base_encoder):
        super().__init__()
        self.encoder = ResNet10_3D()
        self.encoder.load_state_dict(base_encoder.state_dict())
        old = self.encoder.conv1
        new_conv = nn.Conv3d(len(MODALITIES), old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding, bias=False)
        with torch.no_grad():
            new_conv.weight = nn.Parameter(old.weight.repeat(1, len(MODALITIES), 1, 1, 1) / len(MODALITIES))
        self.encoder.conv1 = new_conv
        freeze_all(self.encoder)
        self.mlp = MLPHead(512)

    def forward(self, vols):
        x = torch.cat([vols[mod] for mod in MODALITIES], dim=1)
        return self.mlp(self.encoder(x))

class LateFusionModel(nn.Module):

    def __init__(self, base_encoder):
        super().__init__()
        self.encoders = nn.ModuleList([ResNet10_3D() for _ in MODALITIES])
        for enc in self.encoders:
            enc.load_state_dict(base_encoder.state_dict())
            freeze_all(enc)
        self.heads = nn.ModuleList([MLPHead(512) for _ in MODALITIES])

    def forward(self, vols):
        logits = [head(enc(vols[mod])) for enc, head, mod in zip(self.encoders, self.heads, MODALITIES)]
        return torch.stack(logits, dim=0).mean(dim=0)

def build_model(fusion, base_encoder):
    if fusion == 'inter':
        return IntermediateFusionModel(base_encoder).to(DEVICE)
    elif fusion == 'early':
        return EarlyFusionModel(base_encoder).to(DEVICE)
    elif fusion == 'late':
        return LateFusionModel(base_encoder).to(DEVICE)
    else:
        raise ValueError(f'Unknown fusion: {fusion}')

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

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for vols, labels in loader:
        vols = {m: v.to(DEVICE) for m, v in vols.items()}
        labels = labels.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(model(vols), labels)
        loss.backward()
        optimizer.step()
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

def run_training():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"\n{'=' * 55}")
    print(f'Group B: MedicalNet ResNet-10 (fully frozen) + MLP')
    print(f'  Backbone : ALL frozen - no gradient')
    print(f'  Trained  : MLP head only')
    print(f'Device  : {DEVICE}')
    print(f'Fusions : {FUSIONS}')
    print(f'Folds   : {N_FOLDS}   Epochs: {EPOCHS}   Patience: {PATIENCE}')
    print(f"{'=' * 55}\n")
    df = pd.read_csv(CSV_PATH)
    labels = df['label'].values
    N = len(labels)
    print('Loading MedicalNet weights...')
    base_encoder = ResNet10_3D()
    base_encoder = load_medicalnet_weights(base_encoder, MEDICALNET_WEIGHTS)
    print()
    n_pos = int(labels.sum())
    n_neg = N - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f'Class balance: pos={n_pos} neg={n_neg} pos_weight={pos_weight.item():.3f}\n')
    all_results = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fusion in FUSIONS:
        print(f"\n{'-' * 50}")
        print(f'Fusion: {fusion.upper()}')
        print(f"{'-' * 50}")
        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(N), labels)):
            print(f'\n  Fold {fold + 1}/{N_FOLDS}  train={len(train_idx)} val={len(val_idx)}')
            train_ds = ProstateDataset(DATA_DIR, CSV_PATH, indices=train_idx, augment=True)
            val_ds = ProstateDataset(DATA_DIR, CSV_PATH, indices=val_idx, augment=False)
            train_loader = make_loader(train_ds, shuffle=True, balance=True)
            val_loader = make_loader(val_ds, shuffle=False, balance=False)
            model = build_model(fusion, base_encoder)
            trainable = [p for p in model.parameters() if p.requires_grad]
            print(f'    Trainable params: {sum((p.numel() for p in trainable)):,}')
            optimizer = optim.Adam(trainable, lr=LR, weight_decay=0.0001)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
            best_auc = 0.0
            patience_cnt = 0
            best_path = os.path.join(OUTPUT_DIR, f'fold{fold + 1}_{fusion}_best.pt')
            for epoch in range(EPOCHS):
                loss = train_one_epoch(model, train_loader, optimizer, criterion)
                scheduler.step()
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
            fold_metrics.append(metrics)
            all_results.append({'fusion': fusion, 'fold': fold + 1, **metrics})
        print(f'\n  {fusion.upper()} 5-fold summary:')
        for metric in ['AUC', 'F1', 'Sens', 'Spec']:
            vals = [m[metric] for m in fold_metrics]
            print(f'    {metric:4s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}')
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'cv_results.csv'), index=False)
    summary_rows = []
    for fusion in FUSIONS:
        sub = results_df[results_df['fusion'] == fusion]
        row = {'fusion': fusion}
        for metric in ['AUC', 'F1', 'Sens', 'Spec']:
            m, s = (sub[metric].mean(), sub[metric].std())
            row[metric] = f'{m:.4f}+/-{s:.4f}'
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_DIR, 'cv_summary.csv'), index=False)
    print(f'\nCV results saved to {OUTPUT_DIR}/cv_results.csv')
    print(f"\n{'=' * 55}")
    print('External test set evaluation')
    print(f"{'=' * 55}")
    test_ds = ProstateDataset(TEST_DIR, TEST_CSV_PATH, indices=None, augment=False)
    test_loader = make_loader(test_ds, shuffle=False, balance=False)
    ext_results = []
    for fusion in FUSIONS:
        all_probs = []
        for fold in range(1, N_FOLDS + 1):
            model_path = os.path.join(OUTPUT_DIR, f'fold{fold}_{fusion}_best.pt')
            model = build_model(fusion, base_encoder)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
            _, probs = evaluate(model, test_loader)
            all_probs.append(probs)
        ensemble_probs = np.mean(all_probs, axis=0)
        metrics = compute_metrics(test_ds.labels, ensemble_probs)
        print(f"  {fusion.upper()} (5-fold ensemble):  AUC={metrics['AUC']:.4f}  F1={metrics['F1']:.4f}  Sens={metrics['Sens']:.4f}  Spec={metrics['Spec']:.4f}")
        ext_results.append({'fusion': fusion, **metrics})
    pd.DataFrame(ext_results).to_csv(os.path.join(OUTPUT_DIR, 'external_test_results.csv'), index=False)
    print(f'\nAll results saved to: {OUTPUT_DIR}')
    print(f"{'=' * 55}\n")
if __name__ == '__main__':
    run_training()
