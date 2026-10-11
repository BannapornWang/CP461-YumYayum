# Evaluation, ablation, and report workflow

These scripts use the repository's BasicSR/Real-ESRGAN RRDBNet x4 implementation. The paired synthetic sets generated for this project have `HR/` and `LR_x4/` folders under `datasets/test_synthetic/easy`, `medium`, and `hard`.

## Evaluation

Install optional metrics and plotting dependencies:

```powershell
python -m pip install -r requirements-evaluation.txt
```

Run with the team checkpoint and any downloaded/prepared comparison checkpoints. Each model argument is `LABEL=PATH`; the checkpoint architecture currently supported by this CLI is RRDBNet x4.

```powershell
python project/evaluate.py `
  --models "Real-ESRNet team=experiments/train_RealESRNetx4_CP461_team_5h/models/net_g_45000.pth" `
           "Real-ESRGAN=weights/RealESRGAN_x4plus.pth" `
           "BSRGAN=weights/BSRGAN.pth" `
  --test_set datasets/test_synthetic `
  --output results/evaluation `
  --visualize --blind-study
```

`Bicubic` is always included. The script writes per-image predictions and measurements, grouped CSV/JSON/Markdown tables, and available metric charts. PSNR and SSIM are full-reference metrics. LPIPS, DISTS, NIQE, MUSIQ, and CLIP-IQA are requested through `pyiqa`; if that optional package or a metric implementation is unavailable, the run reports that fact and leaves the metric out rather than fabricating a value. CUDA peak allocated memory, time per image, parameter count, and an edge-mask overshoot ratio are recorded. Use `--tile 256` or another positive tile size if GPU memory is insufficient.

For PSNR/SSIM/CLIP-IQA, higher is generally better; for LPIPS/DISTS/NIQE and overshoot ratio, lower is generally better. Interpret MUSIQ with its selected pyiqa implementation's documented score direction.

The visualizer writes the six-column `Input | Bicubic | ESRGAN | BSRGAN | Ours | GT` panels. Missing model labels are shown as `Not run`. It selects the best and worst available team-model cases relative to bicubic PSNR and saves the requested number of fixed-position zoom crops (4–6, default 5). The blind-study output randomizes anonymous prediction IDs and saves an answer key separately. Participants should only receive the ID folders and score template; keep `answer_key.json` private until scoring is complete. Fill `score_1_to_5` and pass the file with `--study-scores` to create an MOS summary. The script only summarizes ratings actually supplied.

Example with a specific paired split and no optional IQA metrics:

```powershell
python project/evaluate.py --test_set datasets/test_synthetic/medium `
  --models "Ours=experiments/train_RealESRNetx4_CP461_team_5h/models/net_g_10000.pth" `
  --output results/medium --no-iqa --visualize
```

## Ablations

Generate the equal-iteration YAML pairs from the checked-in Real-ESRNet and Real-ESRGAN options:

```powershell
python project/create_ablation_configs.py --iterations 45000
```

The generated `options/ablation/A*.yml` files cover first/second-order degradation, normal/disabled sinc probabilities, VGG/UNet discriminators, spectral norm on/off, Real-ESRNet/Real-ESRGAN, and 192/160 pixel patches. The first-order model and no-spectral-norm discriminator are registered from `project/ablation_models.py`; launch all ablation configs through the wrapper so these project-local registry entries are loaded:

```powershell
python project/train_ablation.py -opt options/ablation/A1_first_order.yml
```

The first-order option synthesizes one blur/resize/noise/JPEG pass directly from GT because BasicSR's existing `high_order_degradation: false` path expects already paired LR inputs. Pixel-unshuffle is not used for x4 in this repo's training path; A6 instead compares patch sizes with batch size and iteration count held equal. The Real-ESRGAN ablation uses the Real-ESRNet initialization path set during config generation; on a clean clone, pass a real checkpoint via `--realesrnet-init PATH`.

After evaluating each ablation, combine their `summary.json` files:

```powershell
python project/summarize_ablation.py --results `
  results/A1/summary.json results/A2/summary.json results/A3/summary.json `
  --output results/ablation_summary
```

## Report figures and results

Create figures from real logs and evaluation CSVs:

```powershell
python project/report_figures.py --log experiments/path/to/train.log `
  --metrics results/evaluation/metrics.csv --output results/report_figures
```

To add an edge profile, provide a real prediction/reference pair:

```powershell
python project/report_figures.py --edge-pair results/pred.png=datasets/test_synthetic/medium/HR/example.png `
  --output results/report_figures
```

No experiment scores are included in this repository document. Populate report tables and slide captions from the generated CSV/Markdown artifacts after the corresponding runs finish. Comparison weights for BSRGAN/RealSR and author-provided Real-ESRGAN are not currently checked in; provide their checkpoint paths explicitly before claiming those comparisons.

## Results status

There are no measured evaluation or user-study scores in this README yet. The current team Real-ESRNet run has not produced its final checkpoint at the time these scripts were prepared. The BSRGAN/RealSR and author reference checkpoint files are also not present in the repository workspace. The blind-study score summary remains empty until participant ratings are supplied. Record actual values by linking the generated `summary.md`, `metrics.csv`, and `mos_summary.csv` files after each run; do not copy placeholders into presentation results.
