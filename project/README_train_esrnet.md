# Real-ESRNet x4 Training Guide

This guide covers training Real-ESRNet x4 super-resolution model on prepared datasets with RTX 3060 12GB GPU.

## Overview

**Real-ESRNet**: Lightweight super-resolution network using **only L1 loss** (no GAN, no discriminator).

**Purpose**: Provides baseline RestorerNet for the subsequent Real-ESRGAN training (Branch 3).

**Training Pipeline**:
1. Load subimage patches from DF2K_sub folder
2. Apply synthesized degradation (2-stage: blur + resize + noise + JPEG)
3. Compute L1 pixel loss between degraded LR and reconstructed output vs ground truth HR
4. Update weights with Adam optimizer
5. Validate every 5,000 iterations on DIV2K validation set

---

## Prerequisites

### Dependencies

```bash
# Install PyTorch with CUDA support (adjust for your GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install BasicSR (required for training infrastructure)
pip install basicsr>=1.4.2

# Install Real-ESRGAN package
pip install realesrgan

# Additional utilities
pip install tensorboard  # For training monitoring
pip install wandb        # Optional: Weights & Biases logging
```

### Dataset Preparation

Before training, complete **Branch 1 (feat/data-prep)**:

```bash
# Verify these files exist:
$ ls -lh datasets/
    meta_info_train.txt     # ~39k subimages
    meta_info_val.txt       # ~1k subimages
    DF2K_sub/               # Directory with subimages
```

If missing, run:
```bash
cd datasets
python ../project/split_dataset.py --seed 42
python ../project/generate_metainfo.py --mode all
```

### GPU Memory Check

```python
# Quick check
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

---

## Configuration

### Training Config: `options/custom/train_realesrnet_x4.yml`

Key parameters adjusted for **RTX 3060 12GB**:

| Parameter | Value | Reason |
|-----------|-------|--------|
| **Patch size (gt_size)** | 192×192 | Reduced from 256 to save memory |
| **Batch size** | 4 | Reduced from 12 (fits ~10-11GB) |
| **Workers** | 4 | Windows stability |
| **Total iterations** | 1,000,000 | ~48-72 hours on RTX 3060 |
| **Learning rate** | 2e-4 | Adam optimizer |
| **Validation freq** | Every 5k iters | Checkpoint + metrics |

### GPU Memory Breakdown

**RTX 3060 12GB with batch=4, patch=192×192, fp32**:
- Model weights (RRDBNet): ~170 MB
- Forward pass (4 samples, 192×192×3): ~2.5 GB
- Backward pass (gradients): ~2.5 GB
- Optimizer states (Adam): ~5 GB
- **Total**: ~10-11 GB (safe margin: 1 GB free)

**If you get out-of-memory errors**:
1. Reduce batch size: `batch_size_per_gpu: 2` (in config)
2. Reduce patch size: `gt_size: 128` (in config)
3. Reduce workers: `num_worker_per_gpu: 2` (in config)
4. Enable mixed precision (requires BasicSR config)

---

## Training Execution

### Quick Start

```bash
# Simple training command
python realesrgan/train.py -opt options/custom/train_realesrnet_x4.yml
```

### Recommended: Using Wrapper Script

The wrapper script logs git commit hash, checks GPU, and validates datasets:

```bash
# Train with git commit tracking
python project/train_realesrnet.py -opt options/custom/train_realesrnet_x4.yml
```

**Output**:
- Checkpoints: `experiments/train_RealESRNetx4_CP461_team_*/models/`
- Logs: `experiments/train_RealESRNetx4_CP461_team_*/log/`
- TensorBoard: `experiments/train_RealESRNetx4_CP461_team_*/tb_logger/`
- Metadata: `experiments/experiment_metadata.json` (includes git hash)

### Resume Training

If training is interrupted, resume from last checkpoint:

```bash
# Manually update config to resume:
# In options/custom/train_realesrnet_x4.yml, set:
# path:
#   resume_state: experiments/train_RealESRNetx4_CP461_team_*/resumed_states/STATE_ITER_*.pth

python realesrgan/train.py -opt options/custom/train_realesrnet_x4.yml
```

BasicSR automatically detects the latest checkpoint in the experiment folder.

---

## Monitoring Training

### TensorBoard Live Monitoring

```bash
# In a separate terminal, monitor training in real-time
tensorboard --logdir=experiments/train_RealESRNetx4_CP461_team_*/tb_logger/
```

Then open browser: http://localhost:6006/

**Metrics to watch**:
- Loss curve (should decrease monotonically)
- Validation PSNR (should increase over time)
- Learning rate drops at 1M iterations
- No NaN/Inf loss values

### Log Files

**Training progress**:
```bash
tail -f experiments/train_RealESRNetx4_CP461_team_*/log/train_*.log
```

**Expected log output** (every 100 iterations):
```
2026-10-11 15:30:45 INFO [iter: 100, lr: 0.0002] loss: 0.2456
2026-10-11 15:31:02 INFO [iter: 200, lr: 0.0002] loss: 0.2389
...
2026-10-11 15:35:00 INFO [iter: 5000, lr: 0.0002] loss: 0.2185 PSNR: 28.45
```

### Weights & Biases (Optional)

For cloud-based experiment tracking:

```bash
# Update config: options/custom/train_realesrnet_x4.yml
logger:
  wandb:
    project: real-esrgan-cp461  # Your W&B project
    resume_id: ~

# Training will automatically log to W&B
python realesrgan/train.py -opt options/custom/train_realesrnet_x4.yml
```

---

## Training Duration & Milestones

### Expected Timeline (RTX 3060 12GB, batch=4)

| Iterations | Wall-clock Time | PSNR Approx | Notes |
|-----------|-----------------|------------|-------|
| 1k | ~10 min | ~24.0 | Warmup phase |
| 10k | ~1.5 hours | ~26.0 | Early convergence |
| 50k | ~7 hours | ~27.5 | Steady improvement |
| 100k | ~14 hours | ~28.0 | Good baseline |
| 500k | ~72 hours | ~28.8 | Close to convergence |
| 1M | ~144 hours (6 days) | ~28.9 | Full training |

**Practical checkpoint strategy**:
1. Train for 50k iterations (~8 hours) → Check PSNR improvement
2. If PSNR > 27.0, continue to 100k
3. If good results at 100k, can stop or continue to 500k
4. Full 1M iterations recommended for production

### Early Stopping

If validation PSNR plateaus (no improvement for 50k+ iterations):

```bash
# Manually stop training (Ctrl+C)
# Latest checkpoint is automatically saved
# Use checkpoint for Branch 3 (Real-ESRGAN training)
```

---

## Output: Checkpoints

### Saved Files

After training, you'll have:

```
experiments/
└── train_RealESRNetx4_CP461_team_[TIMESTAMP]/
    ├── models/
    │   ├── net_g_100000.pth          # Checkpoint at 100k iters
    │   ├── net_g_500000.pth          # Checkpoint at 500k iters
    │   └── net_g_1000000.pth         # Final trained model
    ├── log/
    │   └── train_[TIMESTAMP].log     # Training log
    ├── tb_logger/
    │   └── [tensorboard events]      # TensorBoard data
    ├── resumed_states/               # (if resumed) State files
    └── experiment_metadata.json       # Git hash + GPU info
```

### Using Trained Checkpoint for Branch 3

For Real-ESRGAN training (Branch 3), set in config:

```yaml
path:
  pretrain_network_g: experiments/train_RealESRNetx4_CP461_team_*/models/net_g_500000.pth
  param_key_g: params
  strict_load_g: true
```

This loads the Real-ESRNet weights as initialization for the GAN training.

---

## Troubleshooting

### 1. CUDA Out of Memory (OOM)

**Error**: `RuntimeError: CUDA out of memory`

**Solutions** (in order of preference):
1. Reduce batch size: `batch_size_per_gpu: 2` in config
2. Reduce patch size: `gt_size: 128` in config
3. Clear cache: `torch.cuda.empty_cache()` (not persistent)
4. Enable mixed precision (requires BasicSR setup)

### 2. Dataset Not Found

**Error**: `FileNotFoundError: meta_info_train.txt`

**Solution**:
```bash
cd datasets
python ../project/split_dataset.py --seed 42
python ../project/generate_metainfo.py --mode all
```

### 3. Missing Dependencies

**Error**: `ModuleNotFoundError: basicsr`

**Solution**:
```bash
pip install basicsr>=1.4.2
# Or from repo:
git clone https://github.com/XPixelGroup/BasicSR.git
cd BasicSR
pip install -e .
```

### 4. Training Hangs / Very Slow

**Possible causes**:
- Data loading too slow: Reduce `num_worker_per_gpu`
- Disk I/O bottleneck: Ensure datasets on SSD, not HDD
- CPU bottleneck: Close other applications

**Debug**:
```bash
# Check CPU/GPU utilization
nvidia-smi -l 1  # GPU monitoring
# Windows Task Manager: GPU/CPU/Disk tab
```

### 5. Loss Goes to NaN/Inf

**Possible causes**:
- Learning rate too high
- Numerical instability in loss calculation
- Corrupted images in dataset

**Solutions**:
1. Reduce learning rate: `lr: 1e-4` in config
2. Check dataset: Re-run `generate_metainfo.py --check`
3. Restart training with fresh config

---

## Best Practices

### 1. Checkpoint Management

```bash
# Save important checkpoints
cp experiments/train_RealESRNetx4_CP461_team_*/models/net_g_500000.pth \
   checkpoints/realesrnet_x4_500k.pth

# Verify checkpoint
python -c "import torch; m = torch.load('checkpoints/realesrnet_x4_500k.pth'); print(list(m.keys()))"
```

### 2. Config Variations

For experiments (Branch 5 - ablation), create config copies:

```bash
# Copy for ablation studies
cp options/custom/train_realesrnet_x4.yml options/custom/train_realesrnet_x4_ablation_a1.yml
# Edit: change degradation settings, etc.
```

### 3. Git Commit Before Training

```bash
# Commit current state (config + scripts included in commit)
git add options/custom/train_realesrnet_x4.yml
git add project/train_realesrnet.py
git commit -m "config: add Real-ESRNet x4 training setup for RTX 3060"

# Now training is associated with specific commit hash
# (saved in experiment_metadata.json)
```

### 4. Backup Important Checkpoints

```bash
# After successful training milestones
cp experiments/train_RealESRNetx4_CP461_team_*/models/net_g_*.pth \
   backup/realesrnet_x4/
```

---

## Next Steps

### After Training Completes (100k-500k iterations)

1. **Evaluate checkpoint** (Branch 4):
   ```bash
   # Test Real-ESRNet output on synthetic test sets
   # Use Branch 4 evaluation scripts
   ```

2. **Train Real-ESRGAN** (Branch 3):
   ```bash
   # Use trained Real-ESRNet as init for GAN training
   # Add discriminator + perceptual + GAN losses
   ```

3. **Ablation Studies** (Branch 5):
   ```bash
   # After Real-ESRGAN training, ablate degradation components
   # Change: sinc_prob, blur_sigma, second degradation on/off
   ```

---

## Performance Expectations

### PSNR Progression (on validation set)

**Typical learning curve** (RTX 3060, batch=4, patch=192):
- Iter 0: PSNR ≈ 24.0 dB (random init)
- Iter 10k: PSNR ≈ 26.0 dB
- Iter 100k: PSNR ≈ 28.0 dB (checkpoint recommended for Branch 3)
- Iter 500k: PSNR ≈ 28.8 dB
- Iter 1M: PSNR ≈ 28.9 dB (diminishing returns)

**Reference** (Real-ESRGAN paper):
- Official Real-ESRNet x4: PSNR ≈ 28.9 dB (trained on full config)

**Expected** (your config):
- Real-ESRNet x4 (RTX 3060): PSNR ≈ 28.5-28.8 dB (slightly lower, batch/patch reduced)

---

## Additional Resources

- **BasicSR Docs**: https://github.com/XPixelGroup/BasicSR
- **Real-ESRGAN Paper**: https://arxiv.org/abs/2107.10833
- **RRDBNet Architecture**: https://github.com/xinntao/ESRGAN

---

## Summary

| Step | Command | Duration |
|------|---------|----------|
| 1. Verify dataset | `ls -lh datasets/meta_info_train.txt` | <1 min |
| 2. Check GPU | `python project/train_realesrnet.py --help` | <1 min |
| 3. Start training | `python project/train_realesrnet.py -opt options/custom/train_realesrnet_x4.yml` | 6-144 hrs |
| 4. Monitor | `tensorboard --logdir=experiments/` | Ongoing |
| 5. Save checkpoint | Save final model for Branch 3 | N/A |

**Now proceed with training!** 🚀
