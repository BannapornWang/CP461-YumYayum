# Branch 1: feat/data-prep - Detailed Implementation Plan

## Overview
Prepare and organize datasets (DIV2K, Flickr2K, OTS) for training Real-ESRGAN with proper train/val/test splits, metadata generation, and synthetic test set creation.

---

## Task Breakdown

### Task 1.1: Download Script (`project/download_datasets.py`)
**Purpose**: Automated dataset downloading with error handling

**Inputs**:
- DIV2K download link (from CVPR challenge)
- Flickr2K download link
- OTS download link

**Outputs**:
- `datasets/DIV2K/` (800 train + 100 val images)
- `datasets/Flickr2K/` (2,650 images)
- `datasets/OTS/` (hazy + clear pairs)

**Implementation**:
- Use `requests` + `zipfile` for downloads
- Verify checksums after download
- Extract to proper folders
- Log download progress to `_download.log`
- Argparse: `--dataset {div2k|flickr2k|ots|all}`

---

### Task 1.2: OTS Filtering Script (`project/filter_ots_clear.py`)
**Purpose**: Extract only clear (non-hazy) images from OTS dataset

**Inputs**: `datasets/OTS/original/`
**Outputs**: `datasets/OTS_clear/` (only GT clear images)

**Implementation**:
- OTS has paired hazy/clear in subfolders
- Keep only `_GT.png` files (ground truth/clear)
- Discard `_hazy.png` files
- Copy with verification (check dimensions)
- Log removed files
- Argparse: `--input_dir`, `--output_dir`

---

### Task 1.3: Extract Subimages (`project/prepare_subimages.py`)
**Purpose**: Wrapper around repo's `scripts/extract_subimages.py` with logging

**Inputs**:
- DIV2K, Flickr2K, OTS_clear full images
- Config: crop_size=512, step=480

**Outputs**:
- `datasets/DF2K_sub/` folder structure:
  - `DIV2K_train_sub/` (DIV2K train crops)
  - `DIV2K_val_sub/` (DIV2K val crops, 100 images held-out)
  - `Flickr2K_sub/` (Flickr2K crops)
  - `OTS_clear_sub/` (OTS crops)

**Implementation**:
- Call `scripts/extract_subimages.py` for each dataset
- Log extraction stats (# images → # subimages)
- Argparse: `--datasets {div2k|flickr2k|ots|all|custom}`, `--crop_size`, `--step`

---

### Task 1.4: Meta-Info Generation (`project/generate_metainfo.py`)
**Purpose**: Create metadata .txt files for training data

**Outputs**:
- `datasets/meta_info_train.txt` (train: DIV2K(800×10≈8k) + Flickr2K(2650×10≈26k) + OTS_clear(≈5k) = ≈39k subimages)
- `datasets/meta_info_val.txt` (val: DIV2K_val crops)

**Format**:
```
subimages/DIV2K_train_sub/001_s001.png
subimages/Flickr2K_sub/1_s001.png
subimages/OTS_clear_sub/0_s001.png
...
```

**Implementation**:
- Scan folders, validate image existence
- Use `--check` flag from `generate_meta_info.py`
- Shuffle train split before saving
- Log statistics

---

### Task 1.5: Train/Val/Test Split (`project/split_dataset.py`)
**Purpose**: Create fixed splits with reproducible seed

**Outputs**:
- `datasets/meta_info_train.txt` (train subimages)
- `datasets/meta_info_val.txt` (val subimages from DIV2K val)
- `datasets/meta_info_test.txt` (test subimages - separate folder)
- Held-out test set in `datasets/test_div2k_holdout/`
  - 100 original DIV2K validation images (NOT cropped, full resolution)
  - For later reference-based evaluation

**Implementation**:
- DIV2K: split 0-799 (train) | 800-899 (val → test_holdout)
- Flickr2K: 100% to train
- OTS_clear: 100% to train
- Seed: `42` (fixed, hardcoded, or `config.json`)
- Remove earlier cropped DIV2K_val subimages from metadata

---

### Task 1.6: Synthetic Test Set Generator (`project/generate_synthetic_testset.py`)
**Purpose**: Create synthetic degraded versions for evaluation

**Output Structure**:
```
datasets/test_synthetic/
├── easy/          (light degradation)
├── medium/        (normal degradation)
└── hard/          (heavy degradation)
```

Each folder:
```
├── HR/            (original, 512-640px crops)
├── LR_x4/         (degraded LR, ready for x4 SR)
├── meta_info.txt
```

**Degradation Levels**:

| Level  | Parameters | Motivation |
|--------|-----------|-----------|
| Easy   | 1st: blur×0.5, resize 0.5–1.0, noise σ≤10, JPEG Q≥70 | Clean input |
| Medium | 1st: blur×1.0, resize 0.15–1.5, noise σ≤20, JPEG Q 30–95 | Standard repo config |
| Hard   | 1st: blur×1.5, resize 0.15–1.5, noise σ≤30, JPEG Q 30–95 + 2nd pass | Challenging |

**Implementation**:
- Use Real-ESRGAN's degradation code (`realesrgan/data/realesrgan_dataset.py`)
- Sample 50 images from DIV2K holdout test set
- Create ~150 degraded pairs total (50 per level)
- Fixed seed per level: 42, 43, 44
- Log degradation parameters used

---

## File Output Summary

### Created Files in `project/`:
```
project/
├── download_datasets.py              # Task 1.1
├── filter_ots_clear.py              # Task 1.2
├── prepare_subimages.py             # Task 1.3
├── generate_metainfo.py             # Task 1.4
├── split_dataset.py                 # Task 1.5
├── generate_synthetic_testset.py    # Task 1.6
├── README_data_prep.md              # Instructions
└── _data_prep_log.json              # Summary of all steps
```

### Created Directories:
```
datasets/
├── DIV2K/                      (raw, 800 train + 100 val)
├── Flickr2K/                   (raw, 2650)
├── OTS/                        (raw, hazy+clear pairs)
├── OTS_clear/                  (filtered GT only)
├── DF2K_sub/                   (all subimages)
│   ├── DIV2K_train_sub/
│   ├── DIV2K_val_sub/          (will be excluded from train)
│   ├── Flickr2K_sub/
│   └── OTS_clear_sub/
├── test_div2k_holdout/         (100 full-res DIV2K val images)
├── test_synthetic/
│   ├── easy/  {HR/, LR_x4/, meta_info.txt}
│   ├── medium/ {HR/, LR_x4/, meta_info.txt}
│   └── hard/  {HR/, LR_x4/, meta_info.txt}
├── meta_info_train.txt         (~39k subimages)
├── meta_info_val.txt           (~1k subimages)
└── meta_info_test.txt
```

---

## Execution Command Sequence

```bash
# Run all steps in sequence with logging
python project/download_datasets.py --dataset all          # ~3-4 hours (download intensive)
python project/filter_ots_clear.py                         # ~5-10 min
python project/prepare_subimages.py --datasets all         # ~20-30 min
python project/split_dataset.py                             # ~5 min
python project/generate_metainfo.py --mode all             # ~5 min
python project/generate_synthetic_testset.py               # ~15-20 min
```

---

## Code Quality & Logging

Each script:
- ✓ Type hints (Python 3.10+)
- ✓ Docstrings (functions, classes)
- ✓ Argparse with `--help`
- ✓ Fixed seed (42) for reproducibility
- ✓ Structured logging (`logging` module)
- ✓ Progress bars (`tqdm`)
- ✓ Config saved to `_data_prep_log.json` with git commit hash

---

## Git Commit Strategy

After completing all tasks:
```bash
git checkout -b feat/data-prep
git add project/ datasets/meta_info*.txt
git commit -m "feat: implement data preparation pipeline

- Add download_datasets.py for DIV2K/Flickr2K/OTS downloads
- Add filter_ots_clear.py to extract clear images from OTS
- Add prepare_subimages.py wrapper for subimage extraction
- Add split_dataset.py for train/val/test split (seed 42)
- Add generate_metainfo.py for metadata .txt generation
- Add generate_synthetic_testset.py for easy/medium/hard test sets
- Metadata format: one subimage path per line
- Test set: 100 DIV2K holdout + 150 synthetic pairs (3 levels)

Datasets prepared: ~39k train, ~1k val subimages
GPU memory check: batch_size=4, patch_size=192 ✓"

git push origin feat/data-prep
```

---

## Known Constraints & Assumptions

1. **GPU Memory (RTX 3060 12GB)**:
   - Batch size: 4 (not 12 from reference config)
   - Patch size: 192×192 (not 256×256)
   - Will adjust in Branch 2

2. **Dataset Availability**:
   - DIV2K: Official CVPR challenge link
   - Flickr2K: Official RealSR project link
   - OTS: Official OneDrive link
   - Assumed ✓ accessible from Windows network

3. **Degradation Model**:
   - Use repo's built-in `realesrgan_dataset.py` degradation
   - Second-order blur (2-stage) as per Real-ESRGAN paper
   - Sinc filters enabled

4. **Metadata Format**:
   - Relative paths from `datasets/` root
   - One path per line (no headers)
   - Used directly in `train_realesrgan_x4.yml`

---

## Questions for You (Before Proceeding)

1. ✅ Okay to use official download links? (DIV2K CVPR, Flickr2K RealSR, OTS OneDrive)
2. ✅ Datasets location: `datasets/` at project root? (Not changed)
3. ✅ Batch size/patch size for RTX 3060: 4/192? (Will confirm in Branch 2)
4. ✅ Fixed seed `42` for all splits OK?
5. ✅ Keep 100 DIV2K images as untouched test? (Full resolution, not cropped)
