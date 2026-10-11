#!/usr/bin/env python3
"""
Generate minimal synthetic datasets for pipeline testing.

Creates:
  - DIV2K_train_HR/: 10 synthetic RGB images (480x480)
  - DIV2K_valid_HR/: 2 synthetic RGB images
  - Flickr2K_HR/: 5 synthetic RGB images
  - OTS/original/{0,1,2}/: hazy+GT pairs

This allows testing the entire data pipeline without real datasets.
For actual training, replace with real DIV2K/Flickr2K/OTS downloads.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple
import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

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


def generate_synthetic_image(width: int, height: int, seed: int) -> np.ndarray:
    """
    Generate a synthetic RGB image using patterns.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        seed: Random seed for reproducibility

    Returns:
        RGB image as numpy array (uint8, shape=(height, width, 3))
    """
    rng = np.random.RandomState(seed)

    # Create base patterns
    x = np.linspace(0, 4*np.pi, width)
    y = np.linspace(0, 4*np.pi, height)
    xx, yy = np.meshgrid(x, y)

    # Combine sine waves for pattern
    r = (np.sin(xx) * 127 + 128).astype(np.uint8)
    g = (np.cos(yy) * 127 + 128).astype(np.uint8)
    b = (np.sin(xx + yy) * 127 + 128).astype(np.uint8)

    # Stack into RGB
    img = np.stack([r, g, b], axis=2)

    # Add some random noise
    noise = rng.randint(-20, 20, img.shape)
    img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)

    return img


def save_image(img_array: np.ndarray, path: Path) -> bool:
    """
    Save image from numpy array.

    Args:
        img_array: RGB image as numpy array
        path: Destination path

    Returns:
        True if saved successfully
    """
    if not PIL_AVAILABLE:
        logger.warning(f"PIL not available, cannot save: {path}")
        return False

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.fromarray(img_array, mode='RGB')
        img.save(path)
        logger.info(f"Saved: {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save {path}: {e}")
        return False


def create_div2k_train(base_dir: Path) -> int:
    """Create synthetic DIV2K training images."""
    logger.info("\n[START] Creating DIV2K train...")

    div2k_dir = base_dir / 'DIV2K' / 'DIV2K_train_HR'
    div2k_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for i in range(1, 11):  # 10 images for testing
        img_id = f"{i:04d}"
        img = generate_synthetic_image(480, 480, seed=i)
        path = div2k_dir / f"{img_id}.png"

        if save_image(img, path):
            count += 1

    logger.info(f"[OK] Created {count} DIV2K train images")
    return count


def create_div2k_val(base_dir: Path) -> int:
    """Create synthetic DIV2K validation images."""
    logger.info("\n[START] Creating DIV2K val...")

    div2k_dir = base_dir / 'DIV2K' / 'DIV2K_valid_HR'
    div2k_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for i in range(801, 803):  # 2 images for val
        img_id = f"{i:04d}"
        img = generate_synthetic_image(480, 480, seed=i)
        path = div2k_dir / f"{img_id}.png"

        if save_image(img, path):
            count += 1

    logger.info(f"[OK] Created {count} DIV2K val images")
    return count


def create_flickr2k(base_dir: Path) -> int:
    """Create synthetic Flickr2K images."""
    logger.info("\n[START] Creating Flickr2K...")

    flickr_dir = base_dir / 'Flickr2K' / 'Flickr2K_HR'
    flickr_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for i in range(1, 6):  # 5 images for testing
        img_id = f"{i}"
        img = generate_synthetic_image(480, 480, seed=i+100)
        path = flickr_dir / f"{img_id}.png"

        if save_image(img, path):
            count += 1

    logger.info(f"[OK] Created {count} Flickr2K images")
    return count


def create_ots(base_dir: Path) -> int:
    """Create synthetic OTS (hazy/clear pairs)."""
    logger.info("\n[START] Creating OTS pairs...")

    ots_dir = base_dir / 'OTS' / 'original'
    ots_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for folder_id in range(3):  # 3 folders
        folder = ots_dir / str(folder_id)
        folder.mkdir(exist_ok=True)

        # Each folder has 2 hazy/GT pairs
        for pair_id in range(2):
            seed_base = folder_id * 10 + pair_id

            # GT (clear) image
            gt_img = generate_synthetic_image(480, 480, seed=seed_base)
            gt_path = folder / f"{pair_id}_GT.png"
            if save_image(gt_img, gt_path):
                count += 1

            # Hazy image (slightly degraded)
            hazy_img = (gt_img * 0.8).astype(np.uint8)
            hazy_path = folder / f"{pair_id}_hazy.png"
            if save_image(hazy_img, hazy_path):
                count += 1

    logger.info(f"[OK] Created {count} OTS images (hazy+GT pairs)")
    return count


def save_metadata(base_dir: Path, counts: dict) -> None:
    """Save dataset metadata."""
    meta = {
        'type': 'synthetic_test_datasets',
        'purpose': 'Pipeline testing - replace with real data for actual training',
        'div2k_train': counts.get('div2k_train', 0),
        'div2k_val': counts.get('div2k_val', 0),
        'flickr2k': counts.get('flickr2k', 0),
        'ots': counts.get('ots', 0),
    }

    meta_file = base_dir / '_mini_dataset_info.json'
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f"[OK] Metadata saved: {meta_file}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate minimal synthetic datasets for pipeline testing'
    )
    parser.add_argument(
        '--base-dir',
        type=Path,
        default=Path('datasets'),
        help='Base directory for datasets (default: datasets/)'
    )

    args = parser.parse_args()
    base_dir = Path(args.base_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    log_file = base_dir / '_mini_dataset_create.txt'
    create_logger_file(log_file)

    logger.info("="*60)
    logger.info("Generating Synthetic Test Datasets")
    logger.info("="*60)
    logger.info(f"Base directory: {base_dir}")
    logger.info("WARNING: These are synthetic datasets for pipeline testing only")
    logger.info("Replace with real DIV2K/Flickr2K/OTS for actual training")

    if not PIL_AVAILABLE:
        logger.error("PIL/Pillow not installed - cannot generate images")
        logger.error("Install with: pip install Pillow")
        return 1

    counts = {
        'div2k_train': create_div2k_train(base_dir),
        'div2k_val': create_div2k_val(base_dir),
        'flickr2k': create_flickr2k(base_dir),
        'ots': create_ots(base_dir),
    }

    total = sum(counts.values())
    logger.info(f"\n[OK] Generated {total} synthetic images total")

    save_metadata(base_dir, counts)
    logger.info("\n[OK] Synthetic datasets ready for pipeline testing")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
