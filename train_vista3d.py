import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
DATA_DIR = ''
LABELS_FILE = ''
WEIGHTS_PATH = ''
OUTPUT_DIR = ''
PROBS_DIR = ''
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROBS_DIR, exist_ok=True)
N_FOLDS = 5
SEED = 42
BATCH_SIZE = 8
LR = 0.001
MAX_EPOCHS = 50
PATIENCE = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResBlock3D(nn.Module):

    def __init__(self, ch):
        super().__init__()
        self.norm1 = nn.InstanceNorm3d(ch, affine=True)
        self.conv1 = nn.Conv3d(ch, ch, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(ch, affine=True)
        self.conv2 = nn.Conv3d(ch, ch, 3, padding=1, bias=False)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        r = x
        x = self.act(self.norm1(x))
        x = self.conv1(x)
        x = self.act(self.norm2(x))
        x = self.conv2(x)
        return x + r

class EncoderStage(nn.Module):

    def __init__(self, in_ch, out_ch, n_blocks, has_downsample=True):
        super().__init__()
        self.blocks = nn.ModuleList([ResBlock3D(in_ch) for _ in range(n_blocks)])
        if has_downsample:
            self.downsample = nn.Conv3d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        else:
            self.downsample = None

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class VISTA3DEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_init = nn.Conv3d(1, 48, 3, padding=1, bias=False)
        self.layers = nn.ModuleList([EncoderStage(48, 96, 1, has_downsample=True), EncoderStage(96, 192, 2, has_downsample=True), EncoderStage(192, 384, 2, has_downsample=True), EncoderStage(384, 768, 4, has_downsample=True), EncoderStage(768, 768, 4, has_downsample=False)])

    def forward(self, x):
        x = self.conv_init(x)
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats

def build_vista3d_encoder(weights_path):
    encoder = VISTA3DEncoder()
    if os.path.exists(weights_path):
        print(f'  Loading VISTA3D weights from {weights_path}')
        ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
        weight = ckpt.get('state_dict', ckpt)
        current = encoder.state_dict()
        new_dict = {k: current[k].clone() for k in current}
        loaded, skipped = (0, 0)
        for wk, wv in weight.items():
            if not wk.startswith('image_encoder.encoder.'):
                continue
            ck = wk[len('image_encoder.encoder.'):]
            if ck in current and current[ck].shape == wv.shape:
                new_dict[ck] = wv.clone()
                loaded += 1
            else:
                if ck in current:
                    print(f'  shape mismatch: {ck} {current[ck].shape} vs {wv.shape}')
                skipped += 1
        encoder.load_state_dict(new_dict, strict=False)
        print(f'  VISTA3D weights: {loaded} matched, {skipped} skipped')
    else:
        print(f'  [WARNING] weights not found: {weights_path}')
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder

class VISTA3DClassifier(nn.Module):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Sequential(nn.Linear(768, 256), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, batch):
        ct = batch['CT'].to(DEVICE)
        with torch.no_grad():
            f = self.pool(self.encoder(ct)[-1]).flatten(1)
        return self.mlp(f)

class ProstateDataset(Dataset):

    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.ct = np.load(os.path.join(DATA_DIR, 'center1_CT.npy'), mmap_mode='r')
        self.labels = df['label'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        idx = self.df.iloc[i]['array_idx']
        return {'CT': torch.tensor(self.ct[idx][None], dtype=torch.float32), 'label': torch.tensor(self.labels[i], dtype=torch.float32), 'idx': idx}

def train_fold(model, train_loader, val_loader, fold):
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.402]).to(DEVICE))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    best_auc, best_epoch, no_improve = (0, 0, 0)
    best_path = os.path.join(OUTPUT_DIR, f'fold{fold}_best.pt')
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch in train_loader:
            loss = criterion(model(batch).squeeze(1), batch['label'].to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        probs, lbls = ([], [])
        with torch.no_grad():
            for batch in val_loader:
                probs.extend(torch.sigmoid(model(batch).squeeze(1)).cpu().numpy())
                lbls.extend(batch['label'].numpy())
        auc = roc_auc_score(lbls, probs)
        if auc > best_auc:
            best_auc, best_epoch, no_improve = (auc, epoch, 0)
            torch.save(model.mlp.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break
        if epoch % 10 == 0:
            print(f'    Fold {fold} Epoch {epoch:3d}: AUC={auc:.4f}  best={best_auc:.4f}')
    print(f'  Fold {fold} done: best AUC={best_auc:.4f} @ epoch {best_epoch}')
    return (best_auc, best_path)

def main():
    print('=' * 55)
    print('VISTA3D (frozen) + MLP - CT -> csPCa')
    print(f'Device: {DEVICE}')
    print('=' * 55)
    labels_df = pd.read_csv(LABELS_FILE)
    labels_df['array_idx'] = range(len(labels_df))
    encoder = build_vista3d_encoder(WEIGHTS_PATH)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_aucs = []
    all_probs, all_labels, all_idxs = ([], [], [])
    for fold, (train_idx, val_idx) in enumerate(skf.split(labels_df, labels_df['label']), 1):
        print(f'\nFold {fold}/{N_FOLDS}')
        train_df = labels_df.iloc[train_idx]
        val_df = labels_df.iloc[val_idx]
        counts = train_df['label'].value_counts()
        weights = train_df['label'].map({0: 1 / counts[0], 1: 1 / counts[1]}).values
        sampler = WeightedRandomSampler(weights, len(weights))
        train_ld = DataLoader(ProstateDataset(train_df), BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
        val_ld = DataLoader(ProstateDataset(val_df), BATCH_SIZE, shuffle=False, num_workers=4)
        model = VISTA3DClassifier(encoder).to(DEVICE)
        auc, best_path = train_fold(model, train_ld, val_ld, fold)
        cv_aucs.append(auc)
        model.mlp.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
        model.eval()
        with torch.no_grad():
            for batch in val_ld:
                all_probs.extend(torch.sigmoid(model(batch).squeeze(1)).cpu().numpy())
                all_labels.extend(batch['label'].numpy())
                all_idxs.extend(batch['idx'].numpy())
    print(f"\n{'=' * 55}")
    print(f'CV AUC: {np.mean(cv_aucs):.4f} +/- {np.std(cv_aucs):.4f}')
    print(f"Per-fold: {[f'{a:.4f}' for a in cv_aucs]}")
    pd.DataFrame({'array_idx': all_idxs, 'label': all_labels, 'prob': all_probs}).sort_values('array_idx').to_csv(os.path.join(PROBS_DIR, 'groupA_vista3d_cv_probs.csv'), index=False)
    print('CV probs saved.')
if __name__ == '__main__':
    main()
