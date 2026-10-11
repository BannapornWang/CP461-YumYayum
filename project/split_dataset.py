#!/usr/bin/env python3
"""
Create train/val/test dataset splits with reproducible seed.

Strategy:
  - DIV2K: 0-799 (train) | 800-899 (val → held-out test)
  - Flickr2K: 100% to train
  - OTS_clear: 100% to train
  - Fixed seed: 42

Outputs:
  - datasets/meta_info_train.txt:  ~39k subimages
  - datasets/meta_info_val.txt:    ~1k subimages (from DIV2K_val)
  - datasets/meta_info_test.txt:   test subimages (placeholder)
  - datasets/test_div2k_holdout/:  100 full-res DIV2K val images (for reference evaluation)
"""

import argparse
import copy
import json
import logging
import random
import shutil
from pathlib import Path
from typing import List, Tuple

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


def extract_image_id(filename: str) -> Tuple[int, int]:
    """
    Extract dataset ID and subimage ID from filename.
    
    Examples:
      - DIV2K_train_sub/001_s001.png → (1, 1)
      - Flickr2K_sub/1_s001.png → (1, 1)
      - OTS_clear_sub/0_s001.png → (0, 1)
    
    Args:
        filename: Image filename
        
    Returns:
        (image_id, subimage_id) tuple
    """
    name = Path(filename).stem
    parts = name.split('_')
    
    if len(parts) >= 2:
        try:
            image_id = int(parts[0])
            subimage_id = int(parts[1].replace('s', ''))
            return (image_id, subimage_id)
        except (ValueError, IndexError):
            pass
    
    return (0, 0)


def collect_subimages_by_dataset(
    subimages_dir: Path,
    dataset_name: str
) -> List[str]:
    """
    Collect all subimage paths for a dataset.
    
    Args:
        subimages_dir: DF2K_sub directory
        dataset_name: Dataset subdirectory name (DIV2K_train_sub, etc.)
        
    Returns:
        List of relative paths (relative to subimages_dir)
    """
    dataset_dir = subimages_dir / dataset_name
    
    if not dataset_dir.exists():
        logger.warning(f"Dataset directory not found: {dataset_dir}")
        return []
    
    subimages = []
    for img_path in sorted(dataset_dir.glob('*.png')):
        # Store relative path
        rel_path = f"{dataset_name}/{img_path.name}"
        subimages.append(rel_path)
    
    logger.info(f"{dataset_name}: {len(subimages)} subimages")
    return subimages


def split_div2k_train_val(
    subimages_dir: Path,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Split DIV2K into train (0-799) and val (800-899).
    
    Args:
        subimages_dir: DF2K_sub directory
        seed: Random seed
        
    Returns:
        (train_list, val_list) tuple
    """
    train_subimages = collect_subimages_by_dataset(
        subimages_dir, 'DIV2K_train_sub'
    )
    val_subimages = collect_subimages_by_dataset(
        subimages_dir, 'DIV2K_val_sub'
    )
    
    logger.info(f"DIV2K train: {len(train_subimages)} subimages")
    logger.info(f"DIV2K val (held-out test): {len(val_subimages)} subimages")
    
    return train_subimages, val_subimages


def create_dataset_splits(
    base_dir: Path,
    subimages_dir: Path,
    seed: int = 42
) -> Tuple[List[str], List[str], dict]:
    """
    Create train/val splits and log statistics.
    
    Args:
        base_dir: Datasets directory
        subimages_dir: DF2K_sub directory
        seed: Random seed for reproducibility
        
    Returns:
        (train_list, val_list, stats) tuple
    """
    random.seed(seed)
    logger.info(f"Using seed: {seed}")
    
    # Collect by dataset
    div2k_train, div2k_val = split_div2k_train_val(subimages_dir, seed)
    flickr2k = collect_subimages_by_dataset(subimages_dir, 'Flickr2K_sub')
    ots = collect_subimages_by_dataset(subimages_dir, 'OTS_clear_sub')
    
    # Combine for training
    train_list = div2k_train + flickr2k + ots
    
    # Shuffle training data
    random.shuffle(train_list)
    logger.info(f"Shuffled train list: {len(train_list)} subimages")
    
    # Val is DIV2K_val subimages (but NOT held-out full resolution images)
    val_list = div2k_val
    logger.info(f"Val list: {len(val_list)} subimages")
    
    stats = {
        'seed': seed,
        'div2k_train_count': len(div2k_train),
        'div2k_val_count': len(div2k_val),
        'flickr2k_count': len(flickr2k),
        'ots_count': len(ots),
        'train_total': len(train_list),
        'val_total': len(val_list),
    }
    
    return train_list, val_list, stats


def copy_div2k_val_full_resolution(
    base_dir: Path,
    output_dir: Path
) -> int:
    """
    Copy full-resolution DIV2K validation images to holdout test folder.
    
    Args:
        base_dir: Datasets directory
        output_dir: Output test directory
        
    Returns:
        Number of images copied
    """
    logger.info(f"\nCopying DIV2K validation (full resolution) to: {output_dir}")
    
    src_dir = base_dir / 'DIV2K' / 'DIV2K_valid_HR'
    
    if not src_dir.exists():
        logger.warning(f"DIV2K validation source not found: {src_dir}")
        return 0
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    copied = 0
    for img_path in sorted(src_dir.glob('*.png')):
        try:
            dst_path = output_dir / img_path.name
            shutil.copy2(img_path, dst_path)
            copied += 1
        except Exception as e:
            logger.error(f"Failed to copy {img_path}: {e}")
    
    logger.info(f"✓ Copied {copied} full-resolution DIV2K validation images")
    return copied


def save_metadata_file(
    output_path: Path,
    image_list: List[str],
    description: str
) -> None:
    """
    Save list of image paths to metadata file.
    
    Args:
        output_path: Output .txt file
        image_list: List of relative paths
        description: Description for logging
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for img_path in image_list:
            f.write(f"{img_path}\n")
    
    logger.info(f"✓ Saved {description}: {output_path}")


def save_split_log(base_dir: Path, log_data: dict) -> None:
    """Save split metadata to JSON."""
    log_file = base_dir / '_split_log.json'
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2, default=str)
    logger.info(f"✓ Split log saved: {log_file}")


def main() -> int:
    """
    Main entry point.
    
    Returns:
        0 if successful, 1 on error
    """
    parser = argparse.ArgumentParser(
        description='Create train/val/test dataset splits'
    )
    parser.add_argument(
        '--base-dir',
        type=Path,
        default=Path('datasets'),
        help='Base datasets directory (default: datasets/)'
    )
    parser.add_argument(
        '--subimages-dir',
        type=Path,
        default=Path('datasets/DF2K_sub'),
        help='Subimages directory (default: datasets/DF2K_sub/)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--skip-holdout-copy',
        action='store_true',
        help='Skip copying DIV2K validation to test folder (for large datasets)'
    )
    
    args = parser.parse_args()
    
    # Setup
    base_dir = Path(args.base_dir).resolve()
    subimages_dir = Path(args.subimages_dir).resolve()
    test_dir = base_dir / 'test_div2k_holdout'
    
    base_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = base_dir / '_split_dataset.txt'
    create_logger_file(log_file)
    
    logger.info("="*60)
    logger.info("Dataset Split Creation")
    logger.info("="*60)
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Subimages directory: {subimages_dir}")
    
    # Verify input
    if not subimages_dir.exists():
        logger.error(f"Subimages directory not found: {subimages_dir}")
        return 1
    
    # Create splits
    logger.info(f"\n{'='*60}")
    logger.info("Creating splits...")
    logger.info(f"{'='*60}")
    
    train_list, val_list, stats = create_dataset_splits(
        base_dir,
        subimages_dir,
        seed=args.seed
    )
    
    # Save metadata files
    logger.info(f"\n{'='*60}")
    logger.info("Saving metadata files...")
    logger.info(f"{'='*60}")
    
    save_metadata_file(
        base_dir / 'meta_info_train.txt',
        train_list,
        'train metadata'
    )
    
    save_metadata_file(
        base_dir / 'meta_info_val.txt',
        val_list,
        'val metadata'
    )
    
    # Copy DIV2K val to holdout test
    if not args.skip_holdout_copy:
        holdout_count = copy_div2k_val_full_resolution(base_dir, test_dir)
        stats['div2k_holdout_copied'] = holdout_count
    else:
        logger.info("Skipped copying DIV2K validation to test folder")
    
    # Save log
    split_log = {
        'base_dir': str(base_dir),
        'subimages_dir': str(subimages_dir),
        'stats': stats,
        'train_metadata': str(base_dir / 'meta_info_train.txt'),
        'val_metadata': str(base_dir / 'meta_info_val.txt'),
        'holdout_test_dir': str(test_dir),
    }
    
    save_split_log(base_dir, split_log)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Split Summary")
    logger.info("="*60)
    logger.info(f"Train subimages: {stats['train_total']}")
    logger.info(f"  - DIV2K: {stats['div2k_train_count']}")
    logger.info(f"  - Flickr2K: {stats['flickr2k_count']}")
    logger.info(f"  - OTS: {stats['ots_count']}")
    logger.info(f"Val subimages: {stats['val_total']}")
    logger.info(f"Holdout test: {test_dir.name}/ (full-resolution DIV2K)")
    
    logger.info("\n✓ Dataset split creation completed successfully")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
