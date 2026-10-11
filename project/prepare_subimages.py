#!/usr/bin/env python3
"""
Wrapper script to extract subimages from datasets using repo's extract_subimages.py

This script:
1. Calls scripts/extract_subimages.py for each dataset
2. Logs extraction statistics
3. Verifies output structure

Input:
  - DIV2K_train_HR/  DIV2K_valid_HR/  Flickr2K_HR/  OTS_clear/

Output:
  - DF2K_sub/
    ├── DIV2K_train_sub/
    ├── DIV2K_val_sub/
    ├── Flickr2K_sub/
    └── OTS_clear_sub/
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_logger_file(log_path: Path) -> None:
    """Add file handler to logger."""
    fh = logging.FileHandler(log_path, mode='a')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def extract_subimages_for_dataset(
    dataset_name: str,
    input_dir: Path,
    output_dir: Path,
    crop_size: int = 512,
    step: int = 480,
    num_thread: int = 20
) -> Tuple[bool, dict]:
    """
    Call scripts/extract_subimages.py for a dataset.

    Args:
        dataset_name: Name of dataset (e.g., 'DIV2K_train')
        input_dir: Input image directory
        output_dir: Output subimages directory
        crop_size: Crop size in pixels
        step: Step size between crops
        num_thread: Number of threads

    Returns:
        (success, stats) tuple
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Extracting: {dataset_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Input:  {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Crop size: {crop_size}x{crop_size}, Step: {step}")

    # Verify input
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return False, {}

    input_images = list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg'))
    if not input_images:
        logger.error(f"No images found in {input_dir}")
        return False, {}

    logger.info(f"Found {len(input_images)} input images")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Call scripts/extract_subimages_standalone.py (no basicsr dependency)
    try:
        cmd = [
            'python',
            'scripts/extract_subimages_standalone.py',
            '--input',
            str(input_dir),
            '--output',
            str(output_dir),
            '--crop_size',
            str(crop_size),
            '--step',
            str(step),
            '--thread_num',
            str(num_thread),
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parent.parent,  # Project root
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        if result.returncode != 0:
            logger.error(f"Script failed with return code {result.returncode}")
            logger.error(f"STDOUT:\n{result.stdout}")
            logger.error(f"STDERR:\n{result.stderr}")
            return False, {}

        logger.info(result.stdout)

        # Count output subimages
        output_images = list(output_dir.glob('*.png'))
        logger.info(f"[OK] Extraction complete: {len(output_images)} subimages")

        stats = {
            'dataset': dataset_name,
            'input_count': len(input_images),
            'output_count': len(output_images),
            'crop_size': crop_size,
            'step': step,
            'success': True,
        }

        return True, stats

    except subprocess.TimeoutExpired:
        logger.error(f"Script timeout after 1 hour")
        return False, {}
    except Exception as e:
        logger.error(f"Script error: {e}")
        return False, {}


def extract_all_datasets(
    base_dir: Path,
    output_base_dir: Path,
    datasets_to_extract: List[str],
    crop_size: int = 512,
    step: int = 480,
) -> Tuple[bool, List[dict]]:
    """
    Extract subimages for all specified datasets.

    Args:
        base_dir: Base datasets directory
        output_base_dir: Output base directory
        datasets_to_extract: List of dataset names
        crop_size: Crop size
        step: Step size

    Returns:
        (all_success, stats_list) tuple
    """
    dataset_configs = {
        'div2k_train': {
            'input': base_dir / 'DIV2K' / 'DIV2K_train_HR',
            'output': output_base_dir / 'DIV2K_train_sub',
        },
        'div2k_val': {
            'input': base_dir / 'DIV2K' / 'DIV2K_valid_HR',
            'output': output_base_dir / 'DIV2K_val_sub',
        },
        'flickr2k': {
            'input': base_dir / 'Flickr2K' / 'Flickr2K_HR',
            'output': output_base_dir / 'Flickr2K_sub',
        },
        'ots': {
            'input': base_dir / 'OTS_clear',
            'output': output_base_dir / 'OTS_clear_sub',
        },
    }

    all_stats = []
    all_success = True

    for dataset_name in datasets_to_extract:
        if dataset_name not in dataset_configs:
            logger.warning(f"Unknown dataset: {dataset_name}")
            continue

        config = dataset_configs[dataset_name]
        success, stats = extract_subimages_for_dataset(
            dataset_name,
            config['input'],
            config['output'],
            crop_size=crop_size,
            step=step,
        )

        if stats:
            all_stats.append(stats)
        all_success = all_success and success

    return all_success, all_stats


def save_extraction_log(output_dir: Path, log_data: dict) -> None:
    """Save extraction metadata to JSON."""
    log_file = output_dir / '_extraction_log.json'
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2, default=str)
    logger.info(f"[OK] Extraction log saved: {log_file}")


def main() -> int:
    """
    Main entry point.

    Returns:
        0 if all successful, 1 otherwise
    """
    parser = argparse.ArgumentParser(
        description='Extract subimages from Real-ESRGAN datasets'
    )
    parser.add_argument(
        '--base-dir',
        type=Path,
        default=Path('datasets'),
        help='Base datasets directory (default: datasets/)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('datasets/DF2K_sub'),
        help='Output base directory (default: datasets/DF2K_sub/)'
    )
    parser.add_argument(
        '--datasets',
        choices=['div2k_train', 'div2k_val', 'flickr2k', 'ots', 'all', 'custom'],
        default='all',
        help='Datasets to extract (default: all)'
    )
    parser.add_argument(
        '--crop-size',
        type=int,
        default=512,
        help='Crop size in pixels (default: 512)'
    )
    parser.add_argument(
        '--step',
        type=int,
        default=480,
        help='Step size between crops (default: 480)'
    )

    args = parser.parse_args()

    # Setup
    base_dir = Path(args.base_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / '_prepare_subimages.txt'
    create_logger_file(log_file)

    logger.info("="*60)
    logger.info("Subimage Extraction Pipeline")
    logger.info("="*60)
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Determine datasets to extract
    if args.datasets == 'all':
        datasets_to_extract = ['div2k_train', 'div2k_val', 'flickr2k', 'ots']
    elif args.datasets == 'custom':
        logger.info("Interactive mode: select datasets to extract")
        datasets_to_extract = []
        # User can specify via additional arguments if needed
    else:
        datasets_to_extract = [args.datasets]

    logger.info(f"Datasets to extract: {datasets_to_extract}")

    # Extract subimages
    all_success, all_stats = extract_all_datasets(
        base_dir,
        output_dir,
        datasets_to_extract,
        crop_size=args.crop_size,
        step=args.step,
    )

    # Save log
    extraction_log = {
        'base_dir': str(base_dir),
        'output_dir': str(output_dir),
        'crop_size': args.crop_size,
        'step': args.step,
        'datasets': all_stats,
        'total_datasets': len(all_stats),
        'all_success': all_success,
    }

    save_extraction_log(output_dir, extraction_log)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("Extraction Summary")
    logger.info("="*60)
    for stat in all_stats:
        logger.info(
            f"{stat['dataset']}: "
            f"{stat['input_count']} input -> {stat['output_count']} subimages"
        )

    if all_success:
        logger.info("\n[OK] All extractions completed successfully")
        return 0
    else:
        logger.error("\n[ERROR] Some extractions failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
