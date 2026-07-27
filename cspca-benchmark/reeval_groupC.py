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
GROUPC_BASE = ''
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

def make_loader(dataset):
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

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

class MLPHead(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)

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

class ResNet18_3D_Encoder(nn.Module):

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

class MBConv3D(nn.Module):

    def __init__(self, in_c, out_c, stride=1, expand=4):
        super().__init__()
        mid = in_c * expand
        self.net = nn.Sequential(nn.Conv3d(in_c, mid, 1, bias=False), nn.BatchNorm3d(mid), nn.SiLU(), nn.Conv3d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False), nn.BatchNorm3d(mid), nn.SiLU(), nn.Conv3d(mid, out_c, 1, bias=False), nn.BatchNorm3d(out_c))
        self.skip = stride == 1 and in_c == out_c

    def forward(self, x):
        return x + self.net(x) if self.skip else self.net(x)

class EfficientNet3D_Encoder(nn.Module):

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

class PatchEmbed3D(nn.Module):

    def __init__(self, patch=8, in_c=1, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        x = self.proj(x)
        B, D, H, W, Z = x.shape
        return x.flatten(2).transpose(1, 2)

class ViT3D_Encoder(nn.Module):

    def __init__(self, embed_dim=256, depth=6, n_heads=8):
        super().__init__()
        self.patch_embed = PatchEmbed3D(patch=8, embed_dim=embed_dim)
        n_patches = 64 // 8 * (64 // 8) * (32 // 8)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.transformer(x)
        return self.norm(x[:, 0])

def get_resnet18_2d_encoder():
    import torchvision.models as tvm
    m = tvm.resnet18(weights=None)
    enc = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4, m.avgpool, nn.Flatten())
    return (enc, 512)

def get_efficientnet_2d_encoder():
    import torchvision.models as tvm
    m = tvm.efficientnet_b0(weights=None)
    enc = nn.Sequential(m.features, m.avgpool, nn.Flatten())
    return (enc, 1280)

def get_vit_2d_encoder():
    try:
        import timm
        m = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=0)

        class ViTWrap(nn.Module):

            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                return self.model(x)
        return (ViTWrap(m), 384)
    except ImportError:
        raise ImportError('pip install timm')

def make_inter_model(enc_fn, feat_dim, n_mod=4):

    class M(nn.Module):

        def __init__(self):
            super().__init__()
            self.encoders = nn.ModuleList([enc_fn() for _ in range(n_mod)])
            self.mlp = MLPHead(feat_dim * n_mod)

        def forward(self, vols):
            feats = [enc(vols[mod]) for enc, mod in zip(self.encoders, ALL_MODALITIES)]
            return self.mlp(torch.cat(feats, dim=1))
    return M

def make_early_model_2d(enc_fn, feat_dim, n_mod=4):

    class M(nn.Module):

        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(n_mod * 3, 3, 1, bias=False)
            self.encoder = enc_fn()
            self.mlp = MLPHead(feat_dim)

        def forward(self, vols):
            x = torch.cat([vols[m] for m in ALL_MODALITIES], dim=1)
            x = self.proj(x)
            return self.mlp(self.encoder(x))
    return M

def make_early_model_3d(enc_fn, feat_dim, n_mod=4):

    class M(nn.Module):

        def __init__(self):
            super().__init__()
            self.encoder = enc_fn()
            first_conv = None
            for module in self.encoder.modules():
                if isinstance(module, nn.Conv3d):
                    first_conv = module
                    break
            if first_conv is not None:
                new = nn.Conv3d(n_mod, first_conv.out_channels, first_conv.kernel_size, first_conv.stride, first_conv.padding, bias=False)
                nn.init.kaiming_normal_(new.weight)
                replaced = False
                for name, m in self.encoder.named_modules():
                    if isinstance(m, nn.Conv3d) and (not replaced):
                        parts = name.split('.')
                        parent = self.encoder
                        for p in parts[:-1]:
                            parent = getattr(parent, p)
                        setattr(parent, parts[-1], new)
                        replaced = True
            self.mlp = MLPHead(feat_dim)

        def forward(self, vols):
            x = torch.cat([vols[m] for m in ALL_MODALITIES], dim=1)
            return self.mlp(self.encoder(x))
    return M

def make_late_model(enc_fn, feat_dim, n_mod=4):

    class M(nn.Module):

        def __init__(self):
            super().__init__()
            self.encoders = nn.ModuleList([enc_fn() for _ in range(n_mod)])
            self.heads = nn.ModuleList([MLPHead(feat_dim) for _ in range(n_mod)])

        def forward(self, vols):
            logits = [head(enc(vols[mod])) for enc, head, mod in zip(self.encoders, self.heads, ALL_MODALITIES)]
            return torch.stack(logits, dim=0).mean(dim=0)
    return M

def get_arch_config(arch_name, mode):
    if arch_name == 'resnet18' and mode == '2d':
        enc_fn = lambda: get_resnet18_2d_encoder()[0]
        return (enc_fn, 512, '2d')
    elif arch_name == 'efficientnet' and mode == '2d':
        enc_fn = lambda: get_efficientnet_2d_encoder()[0]
        return (enc_fn, 1280, '2d')
    elif arch_name == 'vit' and mode == '2d':
        enc_fn = lambda: get_vit_2d_encoder()[0]
        return (enc_fn, 384, '2d')
    elif arch_name == 'resnet18' and mode == '3d':
        return (ResNet18_3D_Encoder, 512, '3d')
    elif arch_name == 'efficientnet' and mode == '3d':
        return (EfficientNet3D_Encoder, 1280, '3d')
    elif arch_name == 'vit' and mode == '3d':
        return (ViT3D_Encoder, 256, '3d')
    raise ValueError(f'Unknown arch={arch_name} mode={mode}')

def get_fusion_model_cls(fusion, enc_fn, feat_dim, mode):
    if fusion == 'inter':
        return make_inter_model(enc_fn, feat_dim)
    elif fusion == 'late':
        return make_late_model(enc_fn, feat_dim)
    elif fusion == 'early':
        if mode == '2d':
            return make_early_model_2d(enc_fn, feat_dim)
        else:
            return make_early_model_3d(enc_fn, feat_dim)
    raise ValueError(f'Unknown fusion: {fusion}')

def eval_one(model_dir, arch_name, mode):
    enc_fn, feat_dim, input_mode = get_arch_config(arch_name, mode)
    out_dir = os.path.join(model_dir, 'reeval')
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(TRAIN_CSV)
    labels = df['label'].values
    N = len(labels)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.arange(N), labels))
    all_cv_rows = []
    all_ext_rows = []
    for fusion in FUSIONS:
        print(f'\n  [{arch_name} {mode}] fusion={fusion}')
        fold_metrics = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            fold = fold_idx + 1
            path = os.path.join(model_dir, f'fold{fold}_{fusion}_best.pt')
            if not os.path.exists(path):
                print(f'    Fold {fold}: [SKIP] {path}')
                continue
            ModelCls = get_fusion_model_cls(fusion, enc_fn, feat_dim, mode)
            model = ModelCls().to(DEVICE)
            state = torch.load(path, map_location=DEVICE, weights_only=False)
            state = {k: v.float() if v.dtype == torch.float16 else v for k, v in state.items()}
            model.load_state_dict(state)
            val_ds = ProstateDataset(TRAIN_DATA_DIR, TRAIN_CSV, indices=val_idx, mode=input_mode)
            lbl, probs = get_probs(model, make_loader(val_ds))
            m = compute_optimal(lbl, probs)
            print(f"    Fold {fold}: AUC={m['AUC']:.4f}  Sens={m['Sens']:.4f}  Spec={m['Spec']:.4f}  F1={m['F1']:.4f}  thresh={m['thresh']:.3f}")
            fold_metrics.append(m)
            all_cv_rows.append({'arch': arch_name, 'mode': mode, 'fusion': fusion, 'fold': fold, **m})
        if not fold_metrics:
            continue
        print(f'  {fusion} 5-fold summary (optimal):')
        for metric in ['AUC', 'Sens', 'Spec', 'F1']:
            vals = [x[metric] for x in fold_metrics]
            print(f'    {metric}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}')
        test_ds = ProstateDataset(TEST_DATA_DIR, TEST_CSV, indices=None, mode=input_mode)
        test_loader = make_loader(test_ds)
        all_probs = []
        for fold_idx in range(N_FOLDS):
            fold = fold_idx + 1
            path = os.path.join(model_dir, f'fold{fold}_{fusion}_best.pt')
            if not os.path.exists(path):
                continue
            ModelCls = get_fusion_model_cls(fusion, enc_fn, feat_dim, mode)
            model = ModelCls().to(DEVICE)
            state = torch.load(path, map_location=DEVICE, weights_only=False)
            state = {k: v.float() if v.dtype == torch.float16 else v for k, v in state.items()}
            model.load_state_dict(state)
            _, probs = get_probs(model, test_loader)
            all_probs.append(probs)
        if all_probs:
            ens = np.mean(all_probs, axis=0)
            ext_m = compute_optimal(test_ds.labels, ens)
            print(f"  {fusion} external: AUC={ext_m['AUC']:.4f}  Sens={ext_m['Sens']:.4f}  Spec={ext_m['Spec']:.4f}")
            all_ext_rows.append({'arch': arch_name, 'mode': mode, 'fusion': fusion, **ext_m})
    if all_cv_rows:
        cv_df = pd.DataFrame(all_cv_rows)
        rows = []
        for fusion in FUSIONS:
            sub = cv_df[cv_df['fusion'] == fusion]
            if sub.empty:
                continue
            row = {'arch': arch_name, 'mode': mode, 'fusion': fusion}
            for m in ['AUC', 'Sens', 'Spec', 'F1', 'PPV', 'NPV']:
                if m in sub.columns:
                    row[m] = f'{sub[m].mean():.4f}+/-{sub[m].std():.4f}'
            rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'cv_summary_optimal.csv'), index=False)
    if all_ext_rows:
        pd.DataFrame(all_ext_rows).to_csv(os.path.join(out_dir, 'external_optimal.csv'), index=False)
    return (all_cv_rows, all_ext_rows)

def run_reeval():
    os.makedirs(OUTPUT_ALL, exist_ok=True)
    torch.manual_seed(SEED)
    print(f"\n{'=' * 60}")
    print(f'Group C re-evaluation - optimal threshold (Youden Index)')
    print(f'Device : {DEVICE}')
    print(f"{'=' * 60}")
    configs = [('resnet18', '2d'), ('efficientnet', '2d'), ('vit', '2d'), ('resnet18', '3d'), ('efficientnet', '3d'), ('vit', '3d')]
    all_cv = []
    all_ext = []
    for arch, mode in configs:
        folder = os.path.join(GROUPC_BASE, f'groupC_{arch}_{mode}')
        if not os.path.exists(folder):
            print(f'\n[SKIP] {folder} not found')
            continue
        print(f"\n{'-' * 60}")
        print(f'groupC_{arch}_{mode}')
        print(f"{'-' * 60}")
        cv_rows, ext_rows = eval_one(folder, arch, mode)
        all_cv.extend(cv_rows)
        all_ext.extend(ext_rows)
    if all_cv:
        cv_df = pd.DataFrame(all_cv)
        summary_rows = []
        for (arch, mode, fusion), sub in cv_df.groupby(['arch', 'mode', 'fusion']):
            row = {'arch': arch, 'mode': mode, 'fusion': fusion}
            for m in ['AUC', 'Sens', 'Spec', 'F1', 'PPV', 'NPV']:
                if m in sub.columns:
                    row[m] = f'{sub[m].mean():.4f}+/-{sub[m].std():.4f}'
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_ALL, 'all_cv_summary.csv'), index=False)
        print(f'\nSaved: {OUTPUT_ALL}/all_cv_summary.csv')
    if all_ext:
        pd.DataFrame(all_ext).to_csv(os.path.join(OUTPUT_ALL, 'all_external.csv'), index=False)
        print(f'Saved: {OUTPUT_ALL}/all_external.csv')
    print(f"\n{'=' * 60}")
    print('FINAL SUMMARY - Group C optimal threshold')
    print(f"{'-' * 60}")
    print(f"{'Arch+Mode':<22} {'Fusion':<8} {'CV AUC':>12} {'CV Sens':>10} {'CV Spec':>10} {'Ext AUC':>10}")
    print(f"{'-' * 60}")
    ext_lookup = {(r['arch'], r['mode'], r['fusion']): r for r in all_ext}
    for row in summary_rows:
        key = (row['arch'], row['mode'], row['fusion'])
        ext = ext_lookup.get(key, {})
        print(f"{row['arch']}_{row['mode']:<14} {row['fusion']:<8} {row['AUC']:>12} {row['Sens']:>10} {row['Spec']:>10} {ext.get('AUC', 'N/A'):>10}")
    print(f'\nAll outputs in: {OUTPUT_ALL}')
    print(f"{'=' * 60}\n")
if __name__ == '__main__':
    run_reeval()
