import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from monai.networks.nets.swin_unetr import SwinTransformer
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
IMG_SIZE = (64, 64, 32)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_mars_encoder(weights_path):

    def make_encoder():
        enc = SwinTransformer(in_chans=1, embed_dim=48, window_size=(7, 7, 7), patch_size=(2, 2, 2), depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24), mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, use_checkpoint=False, spatial_dims=3, downsample='merging')
        return enc

    def load_mars_weights(model, weight_file):
        ckpt = torch.load(weight_file, map_location='cpu', weights_only=False)
        weight = ckpt.get('state_dict', ckpt)
        current = model.state_dict()
        new_dict = {k: current[k].clone() for k in current}
        loaded, skipped = (0, 0)
        for wk, wv in weight.items():
            if wk.startswith('swinViT.'):
                ck = wk[len('swinViT.'):]
            else:
                ck = wk
            if ck in current:
                if current[ck].shape == wv.shape:
                    new_dict[ck] = wv.clone()
                    loaded += 1
                else:
                    print(f'    shape mismatch: {ck} {current[ck].shape} vs {wv.shape}')
                    skipped += 1
            else:
                skipped += 1
        model.load_state_dict(new_dict, strict=False)
        print(f'    MARS weights loaded: {loaded} matched, {skipped} skipped')
        return model
    enc_t2w = make_encoder()
    enc_dwi = make_encoder()
    if os.path.exists(weights_path):
        print(f'  Loading MARS weights from {weights_path}')
        enc_t2w = load_mars_weights(enc_t2w, weights_path)
        enc_dwi = load_mars_weights(enc_dwi, weights_path)
    else:
        print(f'  [WARNING] MARS weights not found at {weights_path}')
        print(f'  Download from: https://drive.google.com/file/d/1__lWJfBaCSQqkyPvxpQqK-MWH_d__bWz/view')
        print(f'  Running with random init (results will not be valid)')
    for p in enc_t2w.parameters():
        p.requires_grad = False
    for p in enc_dwi.parameters():
        p.requires_grad = False
    return (enc_t2w, enc_dwi)

class MARSClassifier(nn.Module):

    def __init__(self, enc_t2w, enc_dwi):
        super().__init__()
        self.enc_t2w = enc_t2w
        self.enc_dwi = enc_dwi
        feat_dim = 768 * 2
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Sequential(nn.Linear(feat_dim, 512), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))

    def _extract(self, enc, x):
        hidden = enc(x)
        feat = hidden[-1]
        return self.pool(feat).flatten(1)

    def forward(self, batch):
        t2w = batch['T2W'].to(DEVICE)
        dwi = batch['DWI'].to(DEVICE)
        with torch.no_grad():
            f_t2w = self._extract(self.enc_t2w, t2w)
            f_dwi = self._extract(self.enc_dwi, dwi)
        feat = torch.cat([f_t2w, f_dwi], dim=1)
        return self.mlp(feat)

class ProstateDataset(Dataset):

    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.t2w = np.load(os.path.join(DATA_DIR, 'center1_T2.npy'), mmap_mode='r')
        self.dwi = np.load(os.path.join(DATA_DIR, 'center1_DWI.npy'), mmap_mode='r')
        self.labels = df['label'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        idx = self.df.iloc[i]['array_idx']
        t2w = torch.tensor(self.t2w[idx][None], dtype=torch.float32)
        dwi = torch.tensor(self.dwi[idx][None], dtype=torch.float32)
        return {'T2W': t2w, 'DWI': dwi, 'label': torch.tensor(self.labels[i], dtype=torch.float32), 'idx': idx}

def train_fold(model, train_loader, val_loader, fold):
    pos_weight = torch.tensor([0.402]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    best_auc, best_epoch, no_improve = (0, 0, 0)
    best_path = os.path.join(OUTPUT_DIR, f'fold{fold}_best.pt')
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch in train_loader:
            logits = model(batch).squeeze(1)
            labels = batch['label'].to(DEVICE)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        probs, lbls = ([], [])
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch).squeeze(1)
                probs.extend(torch.sigmoid(logits).cpu().numpy())
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
            print(f'    Fold {fold} Epoch {epoch:3d}: val AUC={auc:.4f}  best={best_auc:.4f}')
    print(f'  Fold {fold} done: best AUC={best_auc:.4f} @ epoch {best_epoch}')
    return (best_auc, best_path)

def main():
    print('=' * 55)
    print('MARS (frozen) + MLP - T2W + DWI -> csPCa')
    print(f'Device: {DEVICE}')
    print('=' * 55)
    labels_df = pd.read_csv(LABELS_FILE)
    labels_df['array_idx'] = range(len(labels_df))
    enc_t2w, enc_dwi = build_mars_encoder(WEIGHTS_PATH)
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
        train_ds = ProstateDataset(train_df)
        val_ds = ProstateDataset(val_df)
        train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
        val_ld = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        model = MARSClassifier(enc_t2w, enc_dwi).to(DEVICE)
        auc, best_path = train_fold(model, train_ld, val_ld, fold)
        cv_aucs.append(auc)
        model.mlp.load_state_dict(torch.load(best_path, map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            for batch in val_ld:
                logits = model(batch).squeeze(1)
                p = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(p)
                all_labels.extend(batch['label'].numpy())
                all_idxs.extend(batch['idx'].numpy())
    cv_auc_mean = np.mean(cv_aucs)
    cv_auc_std = np.std(cv_aucs)
    print(f"\n{'=' * 55}")
    print(f'CV AUC: {cv_auc_mean:.4f} +/- {cv_auc_std:.4f}')
    print(f"Per-fold: {[f'{a:.4f}' for a in cv_aucs]}")
    probs_df = pd.DataFrame({'array_idx': all_idxs, 'label': all_labels, 'prob': all_probs}).sort_values('array_idx')
    out_path = os.path.join(PROBS_DIR, 'groupA_mars_cv_probs.csv')
    probs_df.to_csv(out_path, index=False)
    print(f'CV probs saved: {out_path}')
if __name__ == '__main__':
    main()
