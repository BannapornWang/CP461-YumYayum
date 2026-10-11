
## CP461: Train and Run Real-ESRNet x4

This repository includes a Real-ESRNet x4 training configuration for Windows and an NVIDIA GPU. The short-run configuration is [`options/custom/train_realesrnet_x4_5h.yml`](options/custom/train_realesrnet_x4_5h.yml). It is set to 45,000 iterations with checkpoints every 1,000 iterations; actual runtime depends on the GPU and dataset size.

### 1. Create the Python environment

Use Python 3.12. The current BasicSR training stack is not compatible with Python 3.14.

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install --no-build-isolation basicsr==1.4.2
python -m pip install "numpy<2" opencv-python requests scikit-image scipy tb-nightly addict future lmdb pyyaml yapf
python -m pip install --no-build-isolation --no-deps -e .
```

The train and inference entry points include a compatibility alias for the `torchvision.transforms.functional_tensor` import used by BasicSR 1.4.2.

Confirm that PyTorch can see the GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 2. Prepare the training data

Follow [`project/README_data_prep.md`](project/README_data_prep.md) to obtain and prepare the source datasets. Once the raw datasets are present, create subimages and metadata:

```powershell
python project/prepare_subimages.py --base-dir datasets --output-dir datasets/DF2K_sub --datasets all
python project/split_dataset.py --base-dir datasets --subimages-dir datasets/DF2K_sub --seed 42
python project/generate_metainfo.py --base-dir datasets --mode all
```

Before training, confirm that `datasets/DF2K_sub/` and `datasets/meta_info_train.txt` exist. The number of training images in the metadata file determines how often the model sees each source image.

### 3. Start training

From the repository root, run:

```powershell
python realesrgan/train.py -opt options/custom/train_realesrnet_x4_5h.yml
```

The configuration uses 192-pixel patches and batch size 4. Checkpoints and resumable states are written under `experiments/train_RealESRNetx4_CP461_team_5h/`. The current config disables the paired validation dataset because validation HR images do not have matching LR images; it will not report validation PSNR.

Training logs are also saved under the experiment directory. For TensorBoard, open a second terminal and run:

```powershell
tensorboard --logdir tb_logger
```

### 4. Resume training

Set `path.resume_state` in the YAML file to a state file from `experiments/train_RealESRNetx4_CP461_team_5h/training_states/`, then run the same training command again.

The current run is paused at the latest fully saved checkpoint, iteration 9,000 of 45,000. Its generator weights and resume state are stored in Git LFS at `experiments/train_RealESRNetx4_CP461_team_5h/models/net_g_9000.pth` and `experiments/train_RealESRNetx4_CP461_team_5h/training_states/9000.state`. Set the config's `path.resume_state` to the `.state` path above before resuming. The training log had advanced beyond that save point before the process stopped, so iteration 9,000 is the exact resumable point.

### 5. Run inference with a trained checkpoint

After training completes, use the generated generator checkpoint with the Real-ESRNet x4 architecture:

```powershell
python inference_realesrgan.py --model_name RealESRNet_x4plus --model_path experiments/train_RealESRNetx4_CP461_team_5h/models/net_g_45000.pth --input inputs --output results --tile 256
```

The `--tile` option reduces GPU memory use. Change the checkpoint path to an earlier `net_g_<iteration>.pth` file to run inference before the final iteration.

### Evaluation, ablations, and report figures

See [`project/README_eval_ablation_report.md`](project/README_eval_ablation_report.md) for paired and real-world evaluation, ablation config generation, blind-study setup, and report figure commands. Experiment result values are intentionally omitted until measured artifacts are available.
