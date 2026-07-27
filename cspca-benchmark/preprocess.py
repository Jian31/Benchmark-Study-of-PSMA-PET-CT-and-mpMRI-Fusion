import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import zoom
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
DIR_T2W = ''
DIR_DWI = ''
DIR_PET = ''
DIR_CT = ''
DIR_MASK_T2W = ''
DIR_MASK_DWI = ''
DIR_MASK_PETCT = ''
CSV_PATH = ''
OUTPUT_DIR = ''
TARGET_SIZE = (64, 64, 32)
BBOX_MARGIN = 10
BLACKLIST = set()
MODALITY_DIRS = {'T2W': (DIR_T2W, DIR_MASK_T2W), 'DWI': (DIR_DWI, DIR_MASK_DWI), 'PET': (DIR_PET, DIR_MASK_PETCT), 'CT': (DIR_CT, DIR_MASK_PETCT)}

def get_path(folder, patient_id):
    for ext in ['.nii.gz', '.nii']:
        p = os.path.join(folder, f'{patient_id}{ext}')
        if os.path.exists(p):
            return p
    return os.path.join(folder, f'{patient_id}.nii.gz')

def all_files_exist(patient_id):
    missing = []
    for mod, (img_dir, mask_dir) in MODALITY_DIRS.items():
        for folder, tag in [(img_dir, f'{mod} image'), (mask_dir, f'{mod} mask')]:
            p = get_path(folder, patient_id)
            if not os.path.exists(p):
                missing.append(f'{tag}: {p}')
    return (len(missing) == 0, missing)

def load_nifti(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    arr = np.transpose(arr, (2, 1, 0))
    return arr.astype(np.float32)

def get_bounding_box(mask, margin):
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        raise ValueError('Mask is all zeros.')
    x0, y0, z0 = coords.min(axis=0)
    x1, y1, z1 = coords.max(axis=0)
    sx, sy, sz = mask.shape
    x0 = max(0, x0 - margin)
    x1 = min(sx - 1, x1 + margin)
    y0 = max(0, y0 - margin)
    y1 = min(sy - 1, y1 + margin)
    z0 = max(0, z0 - margin)
    z1 = min(sz - 1, z1 + margin)
    return (x0, x1, y0, y1, z0, z1)

def crop_and_resize(image, mask, margin, target_size):
    x0, x1, y0, y1, z0, z1 = get_bounding_box(mask, margin)
    cropped = image[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]
    bbox_shape = cropped.shape
    zoom_factors = np.array(target_size, dtype=float) / np.array(cropped.shape)
    resized = zoom(cropped, zoom_factors, order=1)
    return (resized.astype(np.float32), bbox_shape)

def zscore_normalize(volume):
    nz = volume[volume != 0]
    if len(nz) == 0:
        return volume
    mu = nz.mean()
    sigma = nz.std()
    if sigma < 1e-08:
        sigma = 1e-08
    out = np.where(volume != 0, (volume - mu) / sigma, 0.0)
    return out.astype(np.float32)

def process_one_patient(patient_id):
    volumes = {}
    log_row = {'ID': patient_id}
    for mod, (img_dir, mask_dir) in MODALITY_DIRS.items():
        image = load_nifti(get_path(img_dir, patient_id))
        mask = load_nifti(get_path(mask_dir, patient_id))
        if image.shape != mask.shape:
            raise ValueError(f'{mod} shape mismatch: image={image.shape} mask={mask.shape}')
        vol, bbox_shape = crop_and_resize(image, mask, BBOX_MARGIN, TARGET_SIZE)
        volumes[mod] = zscore_normalize(vol)
        log_row[f'{mod}_bbox'] = str(bbox_shape)
    return (volumes, log_row)

def run_preprocessing():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    assert 'ID' in df.columns, f"CSV missing 'ID' column.    Found: {list(df.columns)}"
    assert 'label' in df.columns, f"CSV missing 'label' column. Found: {list(df.columns)}"
    df['ID'] = df['ID'].astype(str).str.strip()
    total_csv = len(df)
    print(f"\n{'=' * 55}")
    print(f'Patients in CSV  : {total_csv}')
    print(f'  csPCa  (1)     : {(df.label == 1).sum()}')
    print(f'  non-csPCa (0)  : {(df.label == 0).sum()}')
    print(f'Target size      : {TARGET_SIZE}')
    print(f'BBox margin      : {BBOX_MARGIN} voxels')
    print(f'Output dir       : {OUTPUT_DIR}')
    print(f"{'=' * 55}\n")
    print('Step 1/3  Checking file availability...')
    valid_rows, skipped_rows = ([], [])
    for _, row in df.iterrows():
        if row['ID'] in BLACKLIST:
            skipped_rows.append({'ID': row['ID'], 'label': row['label'], 'missing': 'excluded: image/mask shape mismatch'})
            continue
        ok, missing = all_files_exist(row['ID'])
        if ok:
            valid_rows.append(row)
        else:
            skipped_rows.append({'ID': row['ID'], 'label': row['label'], 'missing': ' | '.join(missing)})
    if skipped_rows:
        skip_df = pd.DataFrame(skipped_rows)
        skip_path = os.path.join(OUTPUT_DIR, 'skipped_cases.csv')
        skip_df.to_csv(skip_path, index=False)
        print(f'  Dropped {len(skipped_rows)} case(s) with missing files -> {skip_path}')
        for s in skipped_rows:
            print(f"    x  {s['ID']}  ({s['missing']})")
    else:
        print(f'  All {total_csv} patients have complete files.')
    valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    n_valid = len(valid_df)
    print(f'  Proceeding with {n_valid} patients.\n')
    print('Step 2/3  Processing volumes...')
    modalities = ['T2W', 'DWI', 'PET', 'CT']
    data = {m: np.zeros((n_valid, *TARGET_SIZE), dtype=np.float32) for m in modalities}
    log_rows = []
    error_rows = []
    for i, row in enumerate(tqdm(valid_df.itertuples(index=False), total=n_valid, desc='  Patients')):
        pid = row.ID
        try:
            volumes, log_row = process_one_patient(pid)
            for mod in modalities:
                data[mod][i] = volumes[mod]
            log_row['label'] = row.label
            log_row['status'] = 'ok'
            log_rows.append(log_row)
        except Exception as e:
            tqdm.write(f'  [ERROR] {pid}: {e}')
            error_rows.append({'ID': pid, 'error': str(e)})
            log_rows.append({'ID': pid, 'label': row.label, 'status': f'ERROR: {e}'})
    print('\nStep 3/3  Saving outputs...')
    for mod in modalities:
        path = os.path.join(OUTPUT_DIR, f'{mod}.npy')
        np.save(path, data[mod])
        print(f'  {mod}.npy    shape={data[mod].shape}   {data[mod].nbytes / 1000000.0:.0f} MB')
    clean_csv_path = os.path.join(OUTPUT_DIR, 'labels_clean.csv')
    valid_df.to_csv(clean_csv_path, index=False)
    print(f'  labels_clean.csv   {n_valid} rows   csPCa={int((valid_df.label == 1).sum())}  non-csPCa={int((valid_df.label == 0).sum())}')
    pd.DataFrame(log_rows).to_csv(os.path.join(OUTPUT_DIR, 'preprocess_log.csv'), index=False)
    print(f'  preprocess_log.csv')
    if error_rows:
        pd.DataFrame(error_rows).to_csv(os.path.join(OUTPUT_DIR, 'errors.csv'), index=False)
        print(f'  errors.csv   ({len(error_rows)} unexpected errors)')
    print(f"\n{'=' * 55}")
    print(f'Original CSV     : {total_csv} patients')
    print(f'Dropped          : {len(skipped_rows)}  (missing files)')
    print(f'Errors           : {len(error_rows)}   (corrupt / shape mismatch)')
    print(f'Final dataset    : {n_valid - len(error_rows)} patients')
    print(f'  csPCa  (1)     : {int((valid_df.label == 1).sum())}')
    print(f'  non-csPCa (0)  : {int((valid_df.label == 0).sum())}')
    print(f'\nOutputs saved to : {OUTPUT_DIR}')
    print(f'Original data    : untouched')
    print(f"{'=' * 55}\n")

def sanity_check(patient_id):
    print(f"\n{'-' * 50}")
    print(f'Sanity check: {patient_id}')
    print(f"{'-' * 50}")
    ok, missing = all_files_exist(patient_id)
    if not ok:
        print('MISSING FILES:')
        for m in missing:
            print(f'  x  {m}')
        return
    for mod, (img_dir, mask_dir) in MODALITY_DIRS.items():
        image = load_nifti(get_path(img_dir, patient_id))
        mask = load_nifti(get_path(mask_dir, patient_id))
        bbox = get_bounding_box(mask, BBOX_MARGIN)
        x0, x1, y0, y1, z0, z1 = bbox
        crop_shape = (x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1)
        vol, _ = crop_and_resize(image, mask, BBOX_MARGIN, TARGET_SIZE)
        vol = zscore_normalize(vol)
        nz = vol[vol != 0]
        print(f'\n  [{mod}]')
        print(f'    raw image   : {image.shape}')
        print(f'    mask        : {mask.shape}  non-zero={int((mask > 0).sum())} voxels')
        print(f'    bbox crop   : {crop_shape}')
        print(f'    after resize: {vol.shape}')
        if len(nz):
            print(f'    norm stats  : mean={nz.mean():.3f}  std={nz.std():.3f}  min={nz.min():.3f}  max={nz.max():.3f}')
    print(f'\n  All files OK for {patient_id}')
    print(f"{'-' * 50}\n")
if __name__ == '__main__':
    run_preprocessing()
