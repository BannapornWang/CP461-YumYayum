# Data Preparation for Real-ESRGAN x4 Training

This guide walks through preparing the datasets for Real-ESRGAN x4 training using the provided scripts.

## Overview

**Goal**: Prepare DIV2K, Flickr2K, and OTS datasets for training with proper train/val/test splits and synthetic test sets.

**Dataset Split**:
- **Train**: DIV2K (800) + Flickr2K (2,650) + OTS clear (~5k) → ~39k subimages after cropping
- **Val**: DIV2K validation subimages (~1k)
- **Test**: DIV2K validation full-resolution (100 images, held-out for reference evaluation)
- **Synthetic Test**: 3 degradation levels (easy, medium, hard) with 50 images each

**Output Structure**:
```
datasets/
├── DIV2K/                      # Raw images
├── Flickr2K/                   # Raw images
├── OTS_clear/                  # Filtered GT only
├── DF2K_sub/                   # Cropped subimages
│   ├── DIV2K_train_sub/
│   ├── DIV2K_val_sub/
│   ├── Flickr2K_sub/
│   └── OTS_clear_sub/
├── test_div2k_holdout/         # Full-resolution DIV2K val (100 images)
├── test_synthetic/             # Synthetic test sets
│   ├── easy/    {HR/, LR_x4/, meta_info.txt}
│   ├── medium/  {HR/, LR_x4/, meta_info.txt}
│   └── hard/    {HR/, LR_x4/, meta_info.txt}
├── meta_info_train.txt         # Training subimages list
├── meta_info_val.txt           # Validation subimages list
└── meta_info_test.txt          # Test subimages list
```

---

## Step-by-Step Execution

### Prerequisites

```bash
# Install dependencies
pip install torch torchvision
pip install basicsr>=1.4.2
pip install opencv-python pillow tqdm
pip install realesrgan
```

### Step 1: Download Datasets

**Purpose**: Download DIV2K, Flickr2K, and OTS datasets.

```bash
python project/download_datasets.py --dataset all --base-dir datasets/
```

**Arguments**:
- `--dataset` {div2k, flickr2k, ots, all}: Which dataset to download
- `--base-dir`: Base directory for storing datasets (default: `datasets/`)
- `--skip-verify`: Skip verification check after download

**Output**:
- `datasets/DIV2K/DIV2K_train_HR/` (800 images)
- `datasets/DIV2K/DIV2K_valid_HR/` (100 images)
- `datasets/Flickr2K/Flickr2K_HR/` (2,650 images)
- `datasets/OTS/original/` (hazy + GT pairs)
- `datasets/_download_log.txt` and `_download_log.json`

**Notes**:
- This step is network-intensive (~45+ GB total)
- You may need to configure download URLs manually based on your access:
  - **DIV2K**: https://data.vision.ee.ethz.ch/cvl/DIV2K/ (CVPR challenge)
  - **Flickr2K**: Contact authors or use RealSR project links
  - **OTS**: OneDrive or official sources
- Consider using `wget` or `curl` for large files if urllib times out

---

### Step 2: Filter OTS Clear Images

**Purpose**: Extract only clear (GT) ground truth images from OTS.

OTS contains paired hazy/clear images. We keep only the clear ones for training data augmentation.

```bash
python project/filter_ots_clear.py \
  --input-dir datasets/OTS/original \
  --output-dir datasets/OTS_clear \
  --verify-dims
```

**Arguments**:
- `--input-dir`: OTS/original directory with hazy/GT pairs
- `--output-dir`: Output directory for GT images
- `--verify-dims`: Validate image dimensions during copy

**Output**:
- `datasets/OTS_clear/` (all clear images from OTS)
- `datasets/_filter_ots_log.txt` and `_filter_ots_log.json`

**Expected**:
- OTS structure: `OTS/original/{0,1,...,N}/{name}_hazy.png` and `{name}_GT.png`
- Output: One folder with only `*_GT.png` files

---

### Step 3: Extract Subimages

**Purpose**: Crop large images into smaller patches (512×512 with 480-pixel step).

This uses the repo's `scripts/extract_subimages.py` for efficiency.

```bash
python project/prepare_subimages.py \
  --base-dir datasets \
  --output-dir datasets/DF2K_sub \
  --datasets all \
  --crop-size 512 \
  --step 480
```

**Arguments**:
- `--base-dir`: Base datasets directory
- `--output-dir`: Output directory for subimages
- `--datasets` {div2k_train, div2k_val, flickr2k, ots, all}: Which to extract
- `--crop-size`: Crop size in pixels (default: 512)
- `--step`: Step size between crops (default: 480, overlap ~6%)

**Output**:
- `datasets/DF2K_sub/DIV2K_train_sub/` (~8k subimages)
- `datasets/DF2K_sub/DIV2K_val_sub/` (~1k subimages)
- `datasets/DF2K_sub/Flickr2K_sub/` (~26k subimages)
- `datasets/DF2K_sub/OTS_clear_sub/` (~5k subimages)
- `datasets/DF2K_sub/_extraction_log.txt` and `_extraction_log.json`

**Time**:
- ~20-30 minutes (multi-threaded, 20 threads default)

---

### Step 4: Create Train/Val/Test Splits

**Purpose**: Organize subimages into train/val splits with fixed seed for reproducibility.

DIV2K validation images are held-out (not used for training) and saved as full-resolution for reference evaluation.

```bash
python project/split_dataset.py \
  --base-dir datasets \
  --subimages-dir datasets/DF2K_sub \
  --seed 42
```

**Arguments**:
- `--base-dir`: Base datasets directory
- `--subimages-dir`: Subimages directory from Step 3
- `--seed`: Random seed (default: 42, fixed for reproducibility)
- `--skip-holdout-copy`: Skip copying full-res DIV2K val (for large datasets)

**Output**:
- `datasets/meta_info_train.txt` (~39k paths, shuffled)
- `datasets/meta_info_val.txt` (~1k paths, DIV2K_val subimages)
- `datasets/test_div2k_holdout/` (100 full-resolution DIV2K validation images)
- `datasets/_split_dataset.txt` and `_split_log.json`

**Format** (meta_info files):
```
DIV2K_train_sub/001_s001.png
DIV2K_train_sub/001_s002.png
Flickr2K_sub/1_s001.png
OTS_clear_sub/0_s001.png
...
```

**Split Details**:
- **Train**: DIV2K 0-799 + Flickr2K (all) + OTS clear (all) = ~39k subimages
- **Val**: DIV2K validation subimages (~1k)
- **Test**: DIV2K validation full-res images (100, held-out)

---

### Step 5: Generate Metadata Files

**Purpose**: Create metadata .txt files for training dataset loader.

This step validates and formalizes the train/val metadata from Step 4.

```bash
python project/generate_metainfo.py \
  --base-dir datasets \
  --mode all \
  --check
```

**Arguments**:
- `--base-dir`: Base datasets directory
- `--mode` {train, val, test, all}: Which metadata to process
- `--check`: Validate image file existence

**Output**:
- `datasets/meta_info_train.txt` (validated)
- `datasets/meta_info_val.txt` (validated)
- `datasets/_metainfo_log.json`

**Note**: This step expects `meta_info_{train,val}.txt` to already exist from Step 4.

---

### Step 6: Generate Synthetic Test Sets

**Purpose**: Create three degradation-level test sets for comprehensive evaluation.

Uses Real-ESRGAN's degradation pipeline to create:
- **Easy**: Light degradation (clean input)
- **Medium**: Standard Real-ESRGAN degradation
- **Hard**: Heavy degradation (challenging cases)

```bash
python project/generate_synthetic_testset.py \
  --input-dir datasets/test_div2k_holdout \
  --output-dir datasets/test_synthetic \
  --num-images-per-level 50 \
  --levels all \
  --seed 42
```

**Arguments**:
- `--input-dir`: Input images (DIV2K holdout, full-res)
- `--output-dir`: Output base directory
- `--num-images-per-level`: Images per degradation level (default: 50)
- `--levels` {easy, medium, hard, all}: Which levels to generate
- `--seed`: Random seed for reproducibility

**Output**:
```
datasets/test_synthetic/
├── easy/
│   ├── HR/          (50 original images)
│   ├── LR_x4/       (50 degraded LR images)
│   └── meta_info.txt
├── medium/
│   ├── HR/          (50 original images)
│   ├── LR_x4/       (50 degraded LR images)
│   └── meta_info.txt
└── hard/
    ├── HR/          (50 original images)
    ├── LR_x4/       (50 degraded LR images)
    └── meta_info.txt
```

**Degradation Levels**:
| Level  | Blur Sigma | Resize Range | Noise σ | JPEG Q | 2nd Pass |
|--------|-----------|--------------|---------|--------|----------|
| Easy   | 0.2-1.0   | 0.7-1.3      | 1-10    | 70-95  | No       |
| Medium | 0.2-3.0   | 0.15-1.5     | 1-30    | 30-95  | Yes      |
| Hard   | 0.5-4.0   | 0.1-1.5      | 5-30    | 20-90  | Yes      |

**Time**: ~15-20 minutes (samples from 100 holdout images)

---

## Complete Execution Script

Run all steps automatically:

```bash
#!/bin/bash
# Run all data prep steps in sequence

echo "Step 1: Download datasets..."
python project/download_datasets.py --dataset all --base-dir datasets/

echo "Step 2: Filter OTS..."
python project/filter_ots_clear.py \
  --input-dir datasets/OTS/original \
  --output-dir datasets/OTS_clear

echo "Step 3: Extract subimages..."
python project/prepare_subimages.py \
  --base-dir datasets \
  --output-dir datasets/DF2K_sub \
  --datasets all

echo "Step 4: Create splits..."
python project/split_dataset.py \
  --base-dir datasets \
  --subimages-dir datasets/DF2K_sub \
  --seed 42

echo "Step 5: Generate metadata..."
python project/generate_metainfo.py \
  --base-dir datasets \
  --mode all

echo "Step 6: Generate synthetic test sets..."
python project/generate_synthetic_testset.py \
  --input-dir datasets/test_div2k_holdout \
  --output-dir datasets/test_synthetic \
  --levels all

echo "✓ Data preparation complete!"
ls -lh datasets/meta_info*.txt
ls -lh datasets/test_synthetic/*/meta_info.txt
```

---

## Troubleshooting

### 1. "Download failed" - Network Issues
- Check internet connection
- Verify URLs are accessible (some may require VPN or manual download)
- Use `wget` or `curl` for large files if urllib times out
- Download links may be region-restricted

### 2. "No images found in..."
- Verify extraction/download completed successfully
- Check file paths are correct
- Look at log files for details: `_download_log.txt`, `_extraction_log.txt`

### 3. Memory errors during extraction
- Reduce `--crop-size` or `--step`
- Reduce `--num-thread` in subset command
- Process datasets separately if needed

### 4. PyTorch import errors in generate_synthetic_testset.py
- Install basicsr: `pip install basicsr`
- Script includes PIL fallback if PyTorch unavailable
- Check log file for degradation backend used

### 5. Git commit issues
- Ensure `.gitignore` excludes datasets/
- Don't commit large dataset files
- Only commit scripts and metadata files

---

## Verification Checklist

After completing all steps:

- [ ] `datasets/meta_info_train.txt` exists and has ~39k lines
- [ ] `datasets/meta_info_val.txt` exists and has ~1k lines
- [ ] `datasets/test_div2k_holdout/` has 100 PNG files
- [ ] `datasets/test_synthetic/easy/HR/` has 50 images
- [ ] `datasets/test_synthetic/medium/HR/` has 50 images
- [ ] `datasets/test_synthetic/hard/HR/` has 50 images
- [ ] All `*_log.json` files are readable and non-empty
- [ ] No dataset folders committed to git

---

## Configuration for Training

Use these metadata files in your training config (`options/custom/train_realesrgan_x4.yml`):

```yaml
datasets:
  train:
    dataroot_gt: datasets/DF2K_sub          # Subimages root
    meta_info: datasets/meta_info_train.txt  # Training list
    
  val:
    dataroot_gt: datasets/DF2K_sub           # Subimages root
    meta_info: datasets/meta_info_val.txt    # Val list
```

For inference/evaluation, use:
- **Reference-based**: `datasets/test_div2k_holdout/` (full-res) + synthetic test sets
- **No-reference**: Your own real-world images

---

## GPU Memory Considerations

For **RTX 3060 12GB**:
- Batch size: 4 (not 12 from reference config)
- Patch size: 192×192 (not 256×256)
- Mixed precision: Recommended

Adjust `--crop-size` if needed:
- 256×256: For 24GB+ GPUs
- 192×192: For 12GB GPUs (current)
- 128×128: For 6GB GPUs

---

## Next Steps

After data preparation:
1. **Branch 2 (feat/train-esrnet)**: Create training config and train Real-ESRNet
2. **Branch 3 (feat/train-esrgan)**: Train Real-ESRGAN on top of Real-ESRNet
3. **Branch 4 (feat/eval)**: Evaluate all models
4. **Branch 5 (feat/ablation)**: Ablation study

---

## References

- Real-ESRGAN Paper: https://arxiv.org/abs/2107.10833
- DIV2K Dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/
- Flickr2K: From RealSR project
- OTS Dataset: Published in dehazing literature

---

## Log Files

All scripts generate logs:
- `_download_log.txt` / `_download_log.json`
- `_filter_ots.txt` / `_filter_ots_log.json`
- `_prepare_subimages.txt` / `_extraction_log.json`
- `_split_dataset.txt` / `_split_log.json`
- `_generate_metainfo.txt` / `_metainfo_log.json`
- `_generate_synthetic_testset.txt` / `_synthetic_testset_log.json`

Check these for detailed execution info and debugging.
