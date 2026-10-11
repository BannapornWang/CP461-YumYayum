#!/usr/bin/env python3
"""
Simple subimage extraction without external dependencies.
Replaces scripts/extract_subimages.py for quick testing.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
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


def extract_subimages_cv2(
    input_path: Path,
    output_path: Path,
    crop_size: int,
    step: int
) -> Dict:
    """Extract subimages using OpenCV."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {
        'total_images': 0,
        'total_subimages': 0,
        'failed': [],
    }

    for img_file in sorted(input_path.glob('*.[pP][nN][gG]')) + sorted(input_path.glob('*.[jJ][pP][gG]')):
        try:
            img = cv2.imread(str(img_file))
            if img is None:
                logger.warning(f"Failed to read: {img_file}")
                stats['failed'].append(str(img_file))
                continue

            h, w = img.shape[:2]
            stats['total_images'] += 1

            # Extract overlapping patches
            sub_count = 0
            for y in range(0, h - crop_size + 1, step):
                for x in range(0, w - crop_size + 1, step):
                    sub_img = img[y:y+crop_size, x:x+crop_size]

                    suffix = f"_s{sub_count:03d}.png"
                    out_name = img_file.stem + suffix
                    out_file = output_path / out_name

                    cv2.imwrite(str(out_file), sub_img)
                    sub_count += 1

            stats['total_subimages'] += sub_count
            logger.info(f"[OK] {img_file.name}: {sub_count} subimages")

        except Exception as e:
            logger.error(f"Failed to process {img_file}: {e}")
            stats['failed'].append(str(img_file))

    return stats


def extract_subimages_pil(
    input_path: Path,
    output_path: Path,
    crop_size: int,
    step: int
) -> Dict:
    """Extract subimages using PIL (fallback)."""
    from PIL import Image

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {
        'total_images': 0,
        'total_subimages': 0,
        'failed': [],
    }

    for img_file in sorted(input_path.glob('*.[pP][nN][gG]')) + sorted(input_path.glob('*.[jJ][pP][gG]')):
        try:
            img = Image.open(img_file)
            w, h = img.size
            stats['total_images'] += 1

            sub_count = 0
            for y in range(0, h - crop_size + 1, step):
                for x in range(0, w - crop_size + 1, step):
                    sub_img = img.crop((x, y, x + crop_size, y + crop_size))

                    suffix = f"_s{sub_count:03d}.png"
                    out_name = img_file.stem + suffix
                    out_file = output_path / out_name

                    sub_img.save(out_file)
                    sub_count += 1

            stats['total_subimages'] += sub_count
            logger.info(f"[OK] {img_file.name}: {sub_count} subimages")

        except Exception as e:
            logger.error(f"Failed to process {img_file}: {e}")
            stats['failed'].append(str(img_file))

    return stats


def main():
    parser = argparse.ArgumentParser(description='Extract subimages from dataset')
    parser.add_argument('--input', type=Path, required=True, help='Input directory')
    parser.add_argument('--output', type=Path, required=True, help='Output directory')
    parser.add_argument('--crop_size', type=int, default=512, help='Crop size')
    parser.add_argument('--step', type=int, default=480, help='Step size')
    parser.add_argument('--thread_num', type=int, default=1, help='Thread count (ignored)')

    args = parser.parse_args()

    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Crop: {args.crop_size}x{args.crop_size}, Step: {args.step}")

    if not args.input.exists():
        logger.error(f"Input not found: {args.input}")
        return 1

    # Try OpenCV first, then PIL
    if CV2_AVAILABLE:
        logger.info("Using OpenCV")
        stats = extract_subimages_cv2(args.input, args.output, args.crop_size, args.step)
    elif PIL_AVAILABLE:
        logger.info("Using PIL (OpenCV not available)")
        stats = extract_subimages_pil(args.input, args.output, args.crop_size, args.step)
    else:
        logger.error("No image library available")
        return 1

    logger.info(f"\n[OK] Total: {stats['total_images']} images -> {stats['total_subimages']} subimages")
    if stats['failed']:
        logger.warning(f"Failed: {len(stats['failed'])} images")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
