# Multimodal Benchmark for Clinically Significant Prostate Cancer (csPCa)

Code accompanying our study on multimodal (mpMRI + PSMA PET/CT) deep-learning
classification of clinically significant prostate cancer. This repository is
released for **research transparency and reproducibility**. It contains the full
preprocessing, training, and evaluation pipeline used in the paper.

> **Note on data.** The imaging data used in this study cannot be shared publicly
> for ethical and patient-privacy reasons. Only code is released here. See
> [Data availability](#data-availability).
>
> **Note on the code.** For anonymization, the scripts have been stripped of
> comments and docstrings, and **all local paths are left blank** — set them
> before running (see [Configure paths](#configure-paths)). All documentation for
> the pipeline lives in this README.

## Overview

Each patient contributes four co-registered modalities — T2W, DWI, PET, and CT —
cropped to the prostate region and resampled to a fixed `64 x 64 x 32` volume. The
task is binary classification (csPCa vs. non-csPCa). Models are trained with 5-fold
cross-validation on an internal cohort and evaluated on an independent external
cohort (5-fold ensemble).

The experiments are organized into four groups plus a preprocessing step:

| Script | Group | Description |
|--------|-------|-------------|
| `preprocess.py` | — | NIfTI to cropped, resized, z-score-normalized `.npy` volumes |
| `train_mars.py` | A | Frozen MARS (MONAI SwinTransformer) encoder + MLP, T2W+DWI |
| `train_vista3d.py` | A | Frozen VISTA3D encoder + MLP, CT |
| `train_groupB.py` | B | Frozen MedicalNet ResNet-10 (3D) + MLP; inter/early/late fusion |
| `train_groupC.py` | C | ImageNet-pretrained EfficientNet-B0 / ResNet-18 / ViT-S, partial fine-tune; 2.5D and 3D inputs |
| `train_groupD.py` | D | Cross-modal Transformer fusion (main method) + modality-ablation experiments |
| `reeval_groupB.py` | B | Re-score saved checkpoints at the Youden-optimal threshold (adds DINOv2) |
| `reeval_groupC.py` | C | Re-score all six Group C configurations at the optimal threshold |
| `reeval_groupD.py` | D | Re-score Group D + ablations at the optimal threshold |

Group D is the paper's main contribution: per-modality 3D ResNet-18 encoders
produce modality tokens fused by a Transformer encoder with a learnable `[CLS]`
token; the modality-ablation variants (e.g. `no_pet`, `mri_only`, `petct_only`)
quantify the incremental value of PSMA PET.

## Repository layout

```
.
├── preprocess.py        # raw NIfTI -> model-ready .npy tensors
├── train_mars.py        # Group A: MARS (frozen) + MLP
├── train_vista3d.py     # Group A: VISTA3D (frozen) + MLP
├── train_groupB.py      # Group B: MedicalNet + MLP
├── train_groupC.py      # Group C: ImageNet backbones + fine-tune
├── train_groupD.py      # Group D: cross-modal Transformer + ablations
├── reeval_groupB.py     # optimal-threshold re-evaluation
├── reeval_groupC.py
├── reeval_groupD.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

Tested with Python 3.10+ and PyTorch 2.x. A CUDA GPU is recommended for training;
all scripts fall back to CPU automatically.

## Configure paths

Every script defines its data / weight / output path variables at the top, left
**blank** (`""`). Before running a script, fill in the ones it uses:

- **Group B / C / D** (`train_groupB.py`, `train_groupC.py`, `train_groupD.py`,
  and the matching `reeval_*.py`): `DATA_DIR`, `CSV_PATH`, `TEST_DIR`,
  `TEST_CSV_PATH`, output directories, and (Group B) `MEDICALNET_WEIGHTS`.
  Each cohort folder is expected to contain `T2W.npy`, `DWI.npy`, `PET.npy`,
  `CT.npy` and a `labels_clean.csv` with `ID, label` columns.
- **Group A** (`train_mars.py`, `train_vista3d.py`): `DATA_DIR`, `LABELS_FILE`,
  `WEIGHTS_PATH`, `OUTPUT_DIR`, `PROBS_DIR`.
- **Preprocessing** (`preprocess.py`): the image/mask folder variables
  (`DIR_T2W`, `DIR_MASK_T2W`, ...), `CSV_PATH`, and `OUTPUT_DIR`.

## Pretrained weights

The foundation-model experiments require external pretrained checkpoints, whose
paths you set as described above:

- **MedicalNet** ResNet-10 — from the MedicalNet project.
- **MARS** — SwinTransformer weights.
- **VISTA3D** — MONAI VISTA3D encoder weights.
- **DINOv2** (`facebook/dinov2-small`) — downloaded automatically via `transformers`.

Please obtain these from their respective original sources under their own
licenses; they are not redistributed here.

## Usage

```bash
# 1. Preprocess each cohort (edit the path variables first)
python preprocess.py

# 2. Train
python train_mars.py
python train_vista3d.py
python train_groupB.py
python train_groupC.py --arch resnet18 --mode 2d      # Group C is parameterized
python train_groupD.py

# 3. Re-evaluate saved checkpoints at the optimal (Youden) threshold
python reeval_groupB.py
python reeval_groupC.py
python reeval_groupD.py
```

Cross-validation uses a fixed seed (`SEED = 42`) and `StratifiedKFold`, so folds
are reproducible across scripts.

## Data availability

The MRI/PET/CT imaging data and patient labels are **not** included in this
repository and cannot be released publicly due to patient-privacy and
institutional-ethics constraints. Access may be considered on reasonable request,
subject to the appropriate data-use and ethics approvals; please contact the
corresponding author. The code is provided so that the full methodology can be
inspected and re-implemented on comparable data.

## License

Released under the [MIT License](LICENSE). Update the copyright line in `LICENSE`
with your name or institution before publishing.

## Citation

If you use this code, please cite our paper. *(Citation / BibTeX to be added upon
publication.)*
