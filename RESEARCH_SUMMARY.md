# Real-ESRGAN Project Structure & Data Handling Analysis

## 1. DATA HANDLING SCRIPTS

### Location: `scripts/`

| Script | Purpose | Key Features |
|--------|---------|--------------|
| **generate_meta_info.py** | Creates meta_info txt files for GT images | Supports multiple folders, image validation with `--check` flag, generates relative paths |
| **generate_meta_info_pairdata.py** | Creates paired GT/LQ meta_info files | Requires matching GT and LQ folder pairs, outputs comma-separated paths |
| **extract_subimages.py** | Crops large images into subimages | Multi-threaded, configurable crop size & step, compression level control |
| **generate_multiscale_DF2K.py** | Creates multiple downscaled versions | Generates 4 scales (0.75x, 0.5x, 0.33x, shortest-edge=400), LANCZOS resampling |
| **pytorch2onnx.py** | Model format conversion | Exports model to ONNX format |

---

## 2. CONFIG FILE ANALYSIS: `options/train_realesrnet_x4plus.yml`

### Basic Settings
```yaml
name: train_RealESRNetx4plus_1000k_B12G4
model_type: RealESRNetModel
scale: 4
num_gpu: auto
manual_seed: 0
gt_usm: True  # Unsharp Mask on ground truth
```

### Dataset Configuration
```yaml
datasets:
  train:
    name: DF2K+OST
    type: RealESRGANDataset
    dataroot_gt: datasets/DF2K
    meta_info: datasets/DF2K/meta_info/meta_info_DF2Kmultiscale+OST_sub.txt
    io_backend:
      type: disk  # or 'lmdb' for LMDB database
    gt_size: 256
    use_hflip: True
    use_rot: False
    batch_size_per_gpu: 12
    num_worker_per_gpu: 5
    dataset_enlarge_ratio: 1
```

---

## 3. DEGRADATION PIPELINE (SYNTHETIC)

### Two-Stage Degradation Process (RealESRGANModel)

**First Degradation:**
```yaml
resize_prob: [0.2, 0.7, 0.1]        # [up, down, keep]
resize_range: [0.15, 1.5]           # Up/down scaling range
gaussian_noise_prob: 0.5
noise_range: [1, 30]                # Gaussian noise sigma
poisson_scale_range: [0.05, 3]      # Poisson noise
gray_noise_prob: 0.4                # Probability of grayscale noise
jpeg_range: [30, 95]                # JPEG quality (Q values)
```

**Second Degradation:**
```yaml
second_blur_prob: 0.8               # Apply second blur
resize_prob2: [0.3, 0.4, 0.3]       # More conservative
resize_range2: [0.3, 1.2]
gaussian_noise_prob2: 0.5
noise_range2: [1, 25]               # Smaller range
poisson_scale_range2: [0.05, 2.5]
gray_noise_prob2: 0.4
jpeg_range2: [30, 95]
```

**Blur Kernels:**
```yaml
blur_kernel_size: 21                # Fixed 21x21 kernel
kernel_list: ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
kernel_prob: [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
sinc_prob: 0.1                      # Sinc filter probability
blur_sigma: [0.2, 3]                # Blur strength range
betag_range: [0.5, 4]               # Generalized Gaussian shape
betap_range: [1, 2]                 # Plateau kernel shape
final_sinc_prob: 0.8                # Final sinc filter probability
```

---

## 4. DIV2K/FLICKR2K DATASET STRUCTURE

### Folder Organization
```
datasets/
└── DF2K/  (Combined DIV2K + Flickr2K)
    ├── DF2K_HR/                           # Original high-res images
    ├── DF2K_multiscale/                  # Multi-scale versions (0.75x, 0.5x, 0.33x, shortest=400)
    ├── DF2K_HR_sub/                      # Cropped subimages (GT)
    ├── DIV2K_train_HR_sub/               # DIV2K subimages
    ├── DIV2K_train_LR_bicubic_X4_sub/    # Bicubic downsampled LR pairs
    └── meta_info/
        ├── meta_info_DF2Kmultiscale+OST_sub.txt      # Training meta file
        ├── meta_info_DF2Kmultiscale.txt
        └── meta_info_DIV2K_sub_pair.txt              # Paired data meta file
```

### Meta Info Format

**Single Image Format (for RealESRGANDataset):**
```
DF2K_HR_sub/000001_s001.png
DF2K_HR_sub/000001_s002.png
DF2K_HR_sub/000001_s003.png
```

**Paired Format (for RealESRGANPairedDataset):**
```
DIV2K_train_HR_sub/000001.png, DIV2K_train_LR_bicubic_X4_sub/000001.png
DIV2K_train_HR_sub/000002.png, DIV2K_train_LR_bicubic_X4_sub/000002.png
```

---

## 5. DEGRADATION PIPELINE CODE SNIPPETS

### Stage 1: Blur + Resize + Noise (from realesrgan_model.py:75-110)

```python
# Blur with generated kernel
out = filter2D(self.gt_usm, self.kernel1)

# Random resize (up/down/keep)
updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
if updown_type == 'up':
    scale = np.random.uniform(1, self.opt['resize_range'][1])
elif updown_type == 'down':
    scale = np.random.uniform(self.opt['resize_range'][0], 1)
else:
    scale = 1
mode = random.choice(['area', 'bilinear', 'bicubic'])
out = F.interpolate(out, scale_factor=scale, mode=mode)

# Add noise
if np.random.uniform() < self.opt['gaussian_noise_prob']:
    out = random_add_gaussian_noise_pt(
        out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_prob)
else:
    out = random_add_poisson_noise_pt(
        out, scale_range=self.opt['poisson_scale_range'], gray_prob=gray_prob, clip=True, rounds=False)

# JPEG compression
jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
out = torch.clamp(out, 0, 1)
out = self.jpeger(out, quality=jpeg_p)
```

### Stage 2: Second Blur + Resize + Noise (from realesrgan_model.py:111-140)

```python
# Optional second blur
if np.random.uniform() < self.opt['second_blur_prob']:
    out = filter2D(out, self.kernel2)

# Second resize (more conservative)
updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
if updown_type == 'up':
    scale = np.random.uniform(1, self.opt['resize_range2'][1])
elif updown_type == 'down':
    scale = np.random.uniform(self.opt['resize_range2'][0], 1)
else:
    scale = 1
out = F.interpolate(
    out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)

# Second noise pass
if np.random.uniform() < self.opt['gaussian_noise_prob2']:
    out = random_add_gaussian_noise_pt(
        out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_prob)
else:
    out = random_add_poisson_noise_pt(
        out, scale_range=self.opt['poisson_scale_range2'], gray_prob=gray_prob, clip=True, rounds=False)

# Final JPEG compression
jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
out = torch.clamp(out, 0, 1)
out = self.jpeger(out, quality=jpeg_p)
```

### Stage 3: Sinc Filter & Resize to Target (from realesrgan_model.py:141-160)

```python
# Two possible orderings (empirically avoid twisted lines)
if np.random.uniform() < 0.5:
    # Order 1: Resize + Sinc + JPEG
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
    out = filter2D(out, self.sinc_kernel)  # Final sinc filter
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
    out = torch.clamp(out, 0, 1)
    out = self.jpeger(out, quality=jpeg_p)
else:
    # Order 2: JPEG + Resize + Sinc
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
    out = torch.clamp(out, 0, 1)
    out = self.jpeger(out, quality=jpeg_p)
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
    out = filter2D(out, self.sinc_kernel)
```

### Kernel Generation (from realesrgan_dataset.py:140-180)

```python
# Generates blur kernels for degradation
kernel_size = random.choice(self.kernel_range)  # 7 to 21
if np.random.uniform() < self.opt['sinc_prob']:
    # Sinc filter (circular lowpass)
    if kernel_size < 13:
        omega_c = np.random.uniform(np.pi / 3, np.pi)
    else:
        omega_c = np.random.uniform(np.pi / 5, np.pi)
    kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
else:
    # Mixed blur kernels (iso, aniso, generalized, plateau)
    kernel = random_mixed_kernels(
        self.kernel_list,
        self.kernel_prob,
        kernel_size,
        self.blur_sigma,
        self.blur_sigma, [-math.pi, math.pi],
        self.betag_range,
        self.betap_range,
        noise_range=None)
# Pad to 21x21
pad_size = (21 - kernel_size) // 2
kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
```

---

## 6. DATA LOADING MODES

### Mode 1: LMDB Backend (Disk -> LMDB)
- Requires `.lmdb` folders with `data.mdb`, `lock.mdb`, and `meta_info.txt`
- Faster random access for large datasets
- Path format in meta_info: filename without extension

### Mode 2: Meta Info (Disk with Meta File)
- Reads relative image paths from txt file
- Single images: one path per line
- Paired images: two paths per line (comma-separated)
- Optional validation with `--check` flag

### Mode 3: Folder Scan (Auto Discovery)
- Scans directories directly without meta file
- Slower but no pre-processing needed
- Used when `meta_info` is None

---

## 7. KEY CODE LOCATIONS

| Component | File Path |
|-----------|-----------|
| RealESRGAN Model (synthesis) | [realesrgan/models/realesrgan_model.py](realesrgan/models/realesrgan_model.py#L70) |
| RealESRGAN Dataset | [realesrgan/data/realesrgan_dataset.py](realesrgan/data/realesrgan_dataset.py#L36) |
| Paired Dataset | [realesrgan/data/realesrgan_paired_dataset.py](realesrgan/data/realesrgan_paired_dataset.py#L12) |
| RealESRNet Model (no discriminator) | [realesrgan/models/realesrnet_model.py](realesrgan/models/realesrnet_model.py) |
| Meta Info Generation | [scripts/generate_meta_info.py](scripts/generate_meta_info.py#L1) |
| Paired Meta Info | [scripts/generate_meta_info_pairdata.py](scripts/generate_meta_info_pairdata.py#L1) |
| Subimage Extraction | [scripts/extract_subimages.py](scripts/extract_subimages.py#L1) |
| Multiscale Generation | [scripts/generate_multiscale_DF2K.py](scripts/generate_multiscale_DF2K.py#L1) |

---

## 8. DATASET NAMING CONVENTION

- **DF2K**: Combined DIV2K + Flickr2K (800 + 2650 images)
- **DIV2K_train_HR**: Original DIV2K training set (high-resolution)
- **DIV2K_train_LR_bicubic_X4**: 4x bicubic downsampled version
- **_sub**: Cropped subimages (typically 512x512 crops with overlap)
- **multiscale**: Multiple scaled versions (0.75x, 0.5x, 0.33x, min-shorter-520)

---

## 9. TRAINING PIPELINE SUMMARY

```
GT Image → USM Sharpening
    ↓
[STAGE 1 DEGRADATION]
    ├─ Blur (Kernel1)
    ├─ Resize (0.15-1.5x)
    ├─ Noise (Gaussian/Poisson, σ:1-30)
    └─ JPEG Compression (Q:30-95)
    ↓
[STAGE 2 DEGRADATION]
    ├─ Optional Blur (Kernel2, 80% prob)
    ├─ Resize (0.3-1.2x)
    ├─ Noise (Gaussian/Poisson, σ:1-25)
    └─ JPEG Compression (Q:30-95)
    ↓
[STAGE 3 FINALIZATION]
    ├─ Option A: Resize + Sinc Filter + JPEG (50% prob)
    └─ Option B: JPEG + Resize + Sinc Filter (50% prob)
    ↓
LQ Output (Downsampled 4x by default)
```

**Key Feature**: Training pair pool (queue) increases degradation diversity across batches
