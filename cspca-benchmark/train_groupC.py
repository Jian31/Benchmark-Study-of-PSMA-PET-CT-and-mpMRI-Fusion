import os
import argparse
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
OUTPUT_BASE = ''
DEFAULT_ARCH = 'resnet18'
DEFAULT_MODE = '2d'
N_FOLDS = 5
BATCH_SIZE = 16
LR_HEAD = 0.001
LR_FINETUNE = 0.0001
LR_FROZEN = 0.0
EPOCHS = 60
PATIENCE = 12
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODALITIES = ['T2W', 'DWI', 'PET', 'CT']
FUSIONS = ['inter', 'early', 'late']
IMG_SIZE = 224

def volume_to_2d(vol_3d):
    z_c = vol_3d.shape[3] // 2
    slices = vol_3d[0, :, :, z_c - 1:z_c + 2]
    slices = slices.permute(2, 0, 1).unsqueeze(0)
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

    def __init__(self, data_dir, csv_path, input_mode, indices=None, augment=False):
        df = pd.read_csv(csv_path)
        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)
        self.labels = df['label'].values.astype(np.int64)
        self.augment = augment
        self.input_mode = input_mode
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
            if self.input_mode == '2d':
                vol = volume_to_2d(vol)
                if self.augment:
                    if torch.rand(1) > 0.5:
                        vol = torch.flip(vol, dims=[2])
                    if torch.rand(1) > 0.5:
                        vol = torch.flip(vol, dims=[1])
            elif self.augment:
                for dim in [1, 2, 3]:
                    if torch.rand(1) > 0.5:
                        vol = torch.flip(vol, dims=[dim])
            vols[mod] = vol
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return (vols, label)

def build_efficientnet_2d():
    import torchvision.models as tvm
    m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT)
    for p in m.parameters():
        p.requires_grad = False
    for p in m.features[6].parameters():
        p.requires_grad = True
    for p in m.features[7].parameters():
        p.requires_grad = True
    feat_dim = 1280
    encoder = nn.Sequential(m.features, m.avgpool, nn.Flatten())
    return (encoder, feat_dim)

def build_resnet18_2d():
    import torchvision.models as tvm
    m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
    for p in m.parameters():
        p.requires_grad = False
    for p in m.layer3.parameters():
        p.requires_grad = True
    for p in m.layer4.parameters():
        p.requires_grad = True
    feat_dim = 512
    encoder = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4, m.avgpool, nn.Flatten())
    return (encoder, feat_dim)

def build_vit_2d():
    try:
        import timm
    except ImportError:
        raise ImportError('timm not installed. Run: pip install timm')
    m = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
    for p in m.parameters():
        p.requires_grad = False
    n_blocks = len(m.blocks)
    for p in m.blocks[n_blocks - 2].parameters():
        p.requires_grad = True
    for p in m.blocks[n_blocks - 1].parameters():
        p.requires_grad = True
    for p in m.norm.parameters():
        p.requires_grad = True
    feat_dim = 384

    class ViTEncoder(nn.Module):

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x)
    return (ViTEncoder(m), feat_dim)

def build_resnet18_3d():

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
            if self.downsample:
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
    return (ResNet18_3D(), 512)

def build_efficientnet_3d():

    class MBConv3D(nn.Module):

        def __init__(self, in_c, out_c, stride=1, expand=4):
            super().__init__()
            mid = in_c * expand
            self.net = nn.Sequential(nn.Conv3d(in_c, mid, 1, bias=False), nn.BatchNorm3d(mid), nn.SiLU(), nn.Conv3d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False), nn.BatchNorm3d(mid), nn.SiLU(), nn.Conv3d(mid, out_c, 1, bias=False), nn.BatchNorm3d(out_c))
            self.skip = stride == 1 and in_c == out_c

        def forward(self, x):
            return x + self.net(x) if self.skip else self.net(x)

    class EfficientNet3D(nn.Module):

        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv3d(1, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm3d(32), nn.SiLU())
            self.blocks = nn.Sequential(MBConv3D(32, 16, 1), MBConv3D(16, 24, 2), MBConv3D(24, 24, 1), MBConv3D(24, 40, 2), MBConv3D(40, 40, 1), MBConv3D(40, 80, 2), MBConv3D(80, 80, 1), MBConv3D(80, 112, 1), MBConv3D(112, 112, 1), MBConv3D(112, 192, 2), MBConv3D(192, 192, 1), MBConv3D(192, 320, 1))
            self.head = nn.Sequential(nn.Conv3d(320, 1280, 1, bias=False), nn.BatchNorm3d(1280), nn.SiLU())
            self.avgpool = nn.AdaptiveAvgPool3d(1)

        def forward(self, x):
            x = self.stem(x)
            x = self.blocks(x)
            x = self.head(x)
            return self.avgpool(x).flatten(1)
    return (EfficientNet3D(), 1280)

def build_vit_3d():

    class PatchEmbed3D(nn.Module):

        def __init__(self, patch=8, in_c=1, embed_dim=256):
            super().__init__()
            self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=patch, stride=patch)

        def forward(self, x):
            x = self.proj(x)
            B, D, H, W, Z = x.shape
            return x.flatten(2).transpose(1, 2)

    class ViT3D(nn.Module):

        def __init__(self, embed_dim=256, depth=6, n_heads=8):
            super().__init__()
            self.patch_embed = PatchEmbed3D(patch=8, embed_dim=embed_dim)
            n_patches = 64 // 8 * (64 // 8) * (32 // 8)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4, dropout=0.1, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
            self.norm = nn.LayerNorm(embed_dim)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        def forward(self, x):
            B = x.shape[0]
            x = self.patch_embed(x)
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = x + self.pos_embed
            x = self.transformer(x)
            return self.norm(x[:, 0])
    return (ViT3D(), 256)

def get_encoder(arch, mode):
    if mode == '2d':
        if arch == 'efficientnet':
            return build_efficientnet_2d()
        elif arch == 'resnet18':
            return build_resnet18_2d()
        elif arch == 'vit':
            return build_vit_2d()
    elif arch == 'efficientnet':
        return build_efficientnet_3d()
    elif arch == 'resnet18':
        return build_resnet18_3d()
    elif arch == 'vit':
        return build_vit_3d()
    raise ValueError(f'Unknown arch={arch} mode={mode}')

class MLPHead(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)

class IntermediateFusion(nn.Module):

    def __init__(self, arch, mode):
        super().__init__()
        self.encoders = nn.ModuleList()
        feat_dim = None
        for _ in MODALITIES:
            enc, fd = get_encoder(arch, mode)
            self.encoders.append(enc)
            feat_dim = fd
        self.mlp = MLPHead(feat_dim * len(MODALITIES))

    def forward(self, vols):
        feats = [enc(vols[mod]) for enc, mod in zip(self.encoders, MODALITIES)]
        return self.mlp(torch.cat(feats, dim=1))

class EarlyFusion(nn.Module):

    def __init__(self, arch, mode):
        super().__init__()
        self.mode = mode
        enc, feat_dim = get_encoder(arch, mode)
        if mode == '2d':
            self.proj = nn.Conv2d(len(MODALITIES) * 3, 3, kernel_size=1, bias=False)
            nn.init.xavier_uniform_(self.proj.weight)
            self.encoder = enc
        else:
            self.encoder = enc
            first_conv = self._find_first_conv(enc)
            if first_conv is not None:
                old = first_conv
                new_conv = nn.Conv3d(len(MODALITIES), old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding, bias=False)
                nn.init.kaiming_normal_(new_conv.weight)
                self._replace_first_conv(enc, new_conv)
        self.mlp = MLPHead(feat_dim)

    def _find_first_conv(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv3d):
                return m
        return None

    def _replace_first_conv(self, module, new_conv):
        for name, m in module.named_modules():
            if isinstance(m, nn.Conv3d):
                parts = name.split('.')
                parent = module
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], new_conv)
                return

    def forward(self, vols):
        if self.mode == '2d':
            x = torch.cat([vols[mod] for mod in MODALITIES], dim=1)
            x = self.proj(x)
        else:
            x = torch.cat([vols[mod] for mod in MODALITIES], dim=1)
        return self.mlp(self.encoder(x))

class LateFusion(nn.Module):

    def __init__(self, arch, mode):
        super().__init__()
        self.encoders = nn.ModuleList()
        feat_dim = None
        for _ in MODALITIES:
            enc, fd = get_encoder(arch, mode)
            self.encoders.append(enc)
            feat_dim = fd
        self.heads = nn.ModuleList([MLPHead(feat_dim) for _ in MODALITIES])

    def forward(self, vols):
        logits = [head(enc(vols[mod])) for enc, head, mod in zip(self.encoders, self.heads, MODALITIES)]
        return torch.stack(logits, dim=0).mean(dim=0)

def build_model(fusion, arch, mode):
    if fusion == 'inter':
        return IntermediateFusion(arch, mode).to(DEVICE)
    elif fusion == 'early':
        return EarlyFusion(arch, mode).to(DEVICE)
    elif fusion == 'late':
        return LateFusion(arch, mode).to(DEVICE)
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

def get_optimizer(model):
    head_params = []
    finetune_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'mlp' in name or 'heads' in name or 'proj' in name:
            head_params.append(p)
        else:
            finetune_params.append(p)
    param_groups = []
    if finetune_params:
        param_groups.append({'params': finetune_params, 'lr': LR_FINETUNE})
    if head_params:
        param_groups.append({'params': head_params, 'lr': LR_HEAD})
    return optim.AdamW(param_groups, weight_decay=0.0001)

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

def run_training(arch, mode):
    output_dir = os.path.join(OUTPUT_BASE, f'groupC_{arch}_{mode}')
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"\n{'=' * 55}")
    print(f'Group C: {arch.upper()} + finetune  [{mode.upper()} input]')
    print(f'Device  : {DEVICE}')
    print(f'Fusions : {FUSIONS}')
    print(f'Folds   : {N_FOLDS}   Epochs: {EPOCHS}   Patience: {PATIENCE}')
    print(f'Output  : {output_dir}')
    print(f"{'=' * 55}\n")
    df = pd.read_csv(CSV_PATH)
    labels = df['label'].values
    N = len(labels)
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
            train_ds = ProstateDataset(DATA_DIR, CSV_PATH, mode, indices=train_idx, augment=True)
            val_ds = ProstateDataset(DATA_DIR, CSV_PATH, mode, indices=val_idx, augment=False)
            train_loader = make_loader(train_ds, shuffle=True, balance=True)
            val_loader = make_loader(val_ds, shuffle=False, balance=False)
            model = build_model(fusion, arch, mode)
            trainable = sum((p.numel() for p in model.parameters() if p.requires_grad))
            total = sum((p.numel() for p in model.parameters()))
            print(f'    Params: trainable={trainable:,} / total={total:,}')
            optimizer = get_optimizer(model)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
            best_auc = 0.0
            patience_cnt = 0
            best_path = os.path.join(output_dir, f'fold{fold + 1}_{fusion}_best.pt')
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
    results_df.to_csv(os.path.join(output_dir, 'cv_results.csv'), index=False)
    summary_rows = []
    for fusion in FUSIONS:
        sub = results_df[results_df['fusion'] == fusion]
        row = {'arch': arch, 'mode': mode, 'fusion': fusion}
        for metric in ['AUC', 'F1', 'Sens', 'Spec']:
            m, s = (sub[metric].mean(), sub[metric].std())
            row[metric] = f'{m:.4f}+/-{s:.4f}'
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, 'cv_summary.csv'), index=False)
    print(f"\n{'=' * 55}")
    print(f'External test set ({arch} {mode})')
    print(f"{'=' * 55}")
    test_ds = ProstateDataset(TEST_DIR, TEST_CSV_PATH, mode, indices=None, augment=False)
    test_loader = make_loader(test_ds, shuffle=False, balance=False)
    ext_results = []
    for fusion in FUSIONS:
        all_probs = []
        for fold in range(1, N_FOLDS + 1):
            mp = os.path.join(output_dir, f'fold{fold}_{fusion}_best.pt')
            m = build_model(fusion, arch, mode)
            m.load_state_dict(torch.load(mp, map_location=DEVICE, weights_only=False))
            _, probs = evaluate(m, test_loader)
            all_probs.append(probs)
        ens = np.mean(all_probs, axis=0)
        metrics = compute_metrics(test_ds.labels, ens)
        print(f"  {fusion.upper()} ensemble: AUC={metrics['AUC']:.4f}  F1={metrics['F1']:.4f}  Sens={metrics['Sens']:.4f}  Spec={metrics['Spec']:.4f}")
        ext_results.append({'fusion': fusion, **metrics})
    pd.DataFrame(ext_results).to_csv(os.path.join(output_dir, 'external_test_results.csv'), index=False)
    print(f'\nAll results saved to: {output_dir}')
    print(f"{'=' * 55}\n")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default=DEFAULT_ARCH, choices=['efficientnet', 'resnet18', 'vit'], help='Backbone architecture')
    parser.add_argument('--mode', default=DEFAULT_MODE, choices=['2d', '3d'], help='Input mode: 2d (2.5D slices) or 3d (full volume)')
    args = parser.parse_args()
    run_training(args.arch, args.mode)
