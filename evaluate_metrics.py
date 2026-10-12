import os
import glob
import argparse
import torch
import pandas as pd
from tabulate import tabulate
import pyiqa

def evaluate_models(sr_dir, gt_dir=None, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initializing metric evaluators
    has_gt = gt_dir is not None and os.path.exists(gt_dir)
    
    fr_metrics = {}
    if has_gt:
        print("Loading Full-Reference metrics (PSNR, SSIM, LPIPS)...")
        fr_metrics['psnr'] = pyiqa.create_metric('psnr', device=device)
        fr_metrics['ssim'] = pyiqa.create_metric('ssim', device=device)
        fr_metrics['lpips'] = pyiqa.create_metric('lpips', device=device)

    print("Loading No-Reference metrics (NIQE, MUSIQ)...")
    nr_metrics = {
        'niqe': pyiqa.create_metric('niqe', device=device),
        'musiq': pyiqa.create_metric('musiq', device=device),
    }

    # Image extensions
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp')
    
    # Subdirectories in sr_dir represent each model (e.g., Bicubic, ESRGAN, Real-ESRGAN)
    # If no subdirectories, evaluate sr_dir as a single model
    model_dirs = [d for d in glob.glob(os.path.join(sr_dir, '*')) if os.path.isdir(d)]
    if not model_dirs:
        model_dirs = [sr_dir]

    results = []

    for m_dir in model_dirs:
        model_name = os.path.basename(m_dir.rstrip('/\\'))
        print(f"\nEvaluating: {model_name}...")

        img_paths = []
        for ext in exts:
            img_paths.extend(glob.glob(os.path.join(m_dir, ext)))

        if not img_paths:
            print(f"  No images found in {m_dir}")
            continue

        scores = {k: [] for k in list(fr_metrics.keys()) + list(nr_metrics.keys())}

        for img_path in sorted(img_paths):
            fname = os.path.basename(img_path)

            # FR Metrics (PSNR, SSIM, LPIPS)
            if has_gt:
                # Find matching ground truth image
                gt_path = os.path.join(gt_dir, fname)
                if not os.path.exists(gt_path):
                    # Try matching without prefix/suffix if needed
                    base_name, _ = os.path.splitext(fname)
                    candidates = glob.glob(os.path.join(gt_dir, f"{base_name}.*"))
                    if candidates:
                        gt_path = candidates[0]
                    else:
                        gt_path = None

                if gt_path and os.path.exists(gt_path):
                    with torch.no_grad():
                        scores['psnr'].append(fr_metrics['psnr'](img_path, gt_path).item())
                        scores['ssim'].append(fr_metrics['ssim'](img_path, gt_path).item())
                        scores['lpips'].append(fr_metrics['lpips'](img_path, gt_path).item())

            # NR Metrics (NIQE, MUSIQ)
            with torch.no_grad():
                scores['niqe'].append(nr_metrics['niqe'](img_path).item())
                scores['musiq'].append(nr_metrics['musiq'](img_path).item())

        row = {'Model': model_name}
        if has_gt and len(scores['psnr']) > 0:
            row['PSNR'] = round(sum(scores['psnr']) / len(scores['psnr']), 2)
            row['SSIM'] = round(sum(scores['ssim']) / len(scores['ssim']), 4)
            row['LPIPS'] = round(sum(scores['lpips']) / len(scores['lpips']), 4)
        else:
            row['PSNR'] = 'N/A'
            row['SSIM'] = 'N/A'
            row['LPIPS'] = 'N/A'

        if len(scores['niqe']) > 0:
            row['NIQE'] = round(sum(scores['niqe']) / len(scores['niqe']), 3)
            row['MUSIQ'] = round(sum(scores['musiq']) / len(scores['musiq']), 2)
        row['MOS'] = '-' # Survey-based

        results.append(row)

    df = pd.DataFrame(results)
    
    # Save to CSV UTF-8
    output_csv = os.path.join(sr_dir, "benchmark_results.csv")
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print("\n" + "="*60)
    print("RESULTS SUMMARY TABLE:")
    print("="*60)
    print(tabulate(df, headers='keys', tablefmt='github', showindex=False))
    print(f"\nSaved results to: {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Image Super-Resolution Models")
    parser.add_argument('--sr_dir', type=str, required=True, help="Directory containing model outputs or subfolders per model")
    parser.add_argument('--gt_dir', type=str, default=None, help="Directory containing High-Resolution Ground-Truth images")
    args = parser.parse_args()

    evaluate_models(args.sr_dir, args.gt_dir)

