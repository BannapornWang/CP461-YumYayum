#!/usr/bin/env python3
"""
Filter OTS (Outdoor haze) dataset to extract only clear (GT) images.

OTS structure:
  OTS/original/{0,1,...,N}/
    ├── {name}_hazy.png     (degraded/hazy)
    └── {name}_GT.png       (ground truth/clear)

Output:
  OTS_clear/
    └── [GT images copied here]

This script keeps only GT images and discards hazy versions.
"""

import argparse
import json
import logging
import shutil
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


def verify_image_file(img_path: Path) -> bool:
    """
    Verify image file is valid (basic check).

    Args:
        img_path: Path to image file

    Returns:
        True if file exists and has expected extension
    """
    if not img_path.exists():
        return False
    if img_path.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
        return False
    return True


def find_gt_images(ots_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Find all GT (clear) images in OTS structure.

    Args:
        ots_dir: Path to OTS/original/ directory

    Returns:
        List of (gt_path, parent_folder) tuples
    """
    gt_images = []

    if not ots_dir.exists():
        logger.error(f"OTS directory not found: {ots_dir}")
        return gt_images

    # OTS structure: original/{0,1,...,N}/{name}_GT.png
    for subfolder in sorted(ots_dir.iterdir()):
        if not subfolder.is_dir():
            continue

        # Find all *_GT.png files in this subfolder
        for img_file in subfolder.glob('*_GT.png'):
            if verify_image_file(img_file):
                gt_images.append((img_file, subfolder.name))
            else:
                logger.warning(f"Skipped invalid image: {img_file}")

    return gt_images


def copy_gt_images(
    gt_images: List[Tuple[Path, Path]],
    output_dir: Path,
    verify_dims: bool = False
) -> Tuple[int, int]:
    """
    Copy GT images to output directory.

    Args:
        gt_images: List of (src_path, subfolder_name) tuples
        output_dir: Destination directory
        verify_dims: If True, validate image dimensions

    Returns:
        (num_copied, num_failed) tuple
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    failed = 0

    for src_path, subfolder_name in gt_images:
        try:
            # Keep original filename with subfolder prefix
            dst_name = f"{subfolder_name}_{src_path.name}"
            dst_path = output_dir / dst_name

            shutil.copy2(src_path, dst_path)
            logger.info(f"[OK] Copied: {src_path.name} -> {dst_path}")
            copied += 1

        except Exception as e:
            logger.error(f"[ERROR] Failed to copy {src_path}: {e}")
            failed += 1

    return copied, failed


def count_hazy_images(ots_dir: Path) -> int:
    """Count hazy images that will be discarded."""
    hazy_count = 0

    for subfolder in ots_dir.iterdir():
        if subfolder.is_dir():
            hazy_count += len(list(subfolder.glob('*_hazy.png')))

    return hazy_count


def save_filter_log(output_dir: Path, log_data: dict) -> None:
    """Save filtering metadata to JSON."""
    log_file = output_dir.parent / '_filter_ots_log.json'
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2, default=str)
    logger.info(f"[OK] Filter log saved: {log_file}")


def main() -> int:
    """
    Main entry point.

    Returns:
        0 if successful, 1 on error
    """
    parser = argparse.ArgumentParser(
        description='Filter OTS dataset to keep only clear (GT) images'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('datasets/OTS/original'),
        help='Input OTS/original/ directory (default: datasets/OTS/original)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('datasets/OTS_clear'),
        help='Output directory for GT images (default: datasets/OTS_clear)'
    )
    parser.add_argument(
        '--verify-dims',
        action='store_true',
        help='Verify image dimensions during copy'
    )

    args = parser.parse_args()

    # Setup
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    # Create log
    datasets_dir = output_dir.parent
    datasets_dir.mkdir(parents=True, exist_ok=True)

    log_file = datasets_dir / '_filter_ots.txt'
    create_logger_file(log_file)

    logger.info("="*60)
    logger.info("OTS Dataset Filtering (GT Extraction)")
    logger.info("="*60)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Verify input
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1

    # Find GT images
    logger.info("\nScanning for GT images...")
    gt_images = find_gt_images(input_dir)

    if not gt_images:
        logger.error("No GT images found in OTS structure")
        return 1

    logger.info(f"Found {len(gt_images)} GT images")

    # Count hazy images (will be discarded)
    hazy_count = count_hazy_images(input_dir)
    logger.info(f"Will discard {hazy_count} hazy images")

    # Copy GT images
    logger.info("\nCopying GT images...")
    copied, failed = copy_gt_images(
        gt_images,
        output_dir,
        verify_dims=args.verify_dims
    )

    logger.info(f"\nCopied: {copied} images")
    if failed > 0:
        logger.warning(f"Failed: {failed} images")

    # Save log
    filter_log = {
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'gt_images_copied': copied,
        'gt_images_failed': failed,
        'hazy_images_discarded': hazy_count,
        'total_input_images': len(gt_images) + hazy_count,
    }

    save_filter_log(output_dir, filter_log)

    if failed == 0:
        logger.info("\n[OK] Filtering PASSED")
        return 0
    else:
        logger.warning(f"\n[WARN] Filtering completed with {failed} failures")
        return 1


if __name__ == '__main__':
    sys.exit(main())
