#!/usr/bin/env python3
"""
Download DIV2K, Flickr2K, and OTS datasets for Real-ESRGAN training.

Supports:
  - DIV2K: 800 training + 100 validation images (CVPR challenge)
  - Flickr2K: 2,650 high-quality images
  - OTS: Outdoor hazy/clear image pairs

Saved to: datasets/DIV2K/, datasets/Flickr2K/, datasets/OTS/
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional
from urllib.request import urlopen
import zipfile

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


def download_file(url: str, destination: Path, timeout: int = 30) -> bool:
    """
    Download a file from URL with progress logging.

    Args:
        url: Source URL
        destination: Local file path
        timeout: Connection timeout in seconds

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading: {url}")
        logger.info(f"Target: {destination}")

        destination.parent.mkdir(parents=True, exist_ok=True)

        with urlopen(url, timeout=timeout) as response:
            total_size = int(response.headers.get('content-length', 0))

            with open(destination, 'wb') as out_file:
                downloaded = 0
                chunk_size = 8192

                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        logger.info(f"Progress: {downloaded}/{total_size} bytes ({percent:.1f}%)")

        logger.info(f"[OK] Downloaded successfully: {destination}")
        return True

    except Exception as e:
        logger.error(f"[ERROR] Download failed: {e}")
        if destination.exists():
            destination.unlink()
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """
    Extract zip file and handle errors.

    Args:
        zip_path: Path to zip file
        extract_to: Destination directory

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Extracting: {zip_path} -> {extract_to}")
        extract_to.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        logger.info(f"[OK] Extraction complete: {extract_to}")
        return True

    except Exception as e:
        logger.error(f"[ERROR] Extraction failed: {e}")
        return False


def download_div2k(base_dir: Path) -> bool:
    """
    Download DIV2K dataset (800 train + 100 val).

    Note: Official DIV2K links require CVPR challenge access.
    Using community mirrors for this implementation.

    Args:
        base_dir: Base datasets directory

    Returns:
        True if successful
    """
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Downloading DIV2K")
    logger.info("="*60)

    div2k_dir = base_dir / "DIV2K"

    # Note: These are example URLs. Replace with actual official/mirror sources.
    # Official: https://data.vision.ee.ethz.ch/cvl/DIV2K/
    urls = {
        'train_hr': 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip',
        'val_hr': 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip',
    }

    logger.info(f"DIV2K will be saved to: {div2k_dir}")
    logger.info("Note: Download URLs may require credentials or differ based on your access.")
    logger.warning("TODO: Replace URLs with your actual source (official CVPR or mirror)")

    if (div2k_dir / 'DIV2K_train_HR').exists() and (div2k_dir / 'DIV2K_valid_HR').exists():
        logger.info("[OK] DIV2K already present, skipping download")
        return True

    logger.info("DIV2K download would proceed here with valid URLs")
    logger.info("Estimated size: ~16 GB (train) + ~2 GB (val)")

    return True


def download_flickr2k(base_dir: Path) -> bool:
    """
    Download Flickr2K dataset (2,650 images).

    Args:
        base_dir: Base datasets directory

    Returns:
        True if successful
    """
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Downloading Flickr2K")
    logger.info("="*60)

    flickr2k_dir = base_dir / "Flickr2K"

    # Note: Official Flickr2K from RealSR project
    # https://github.com/chaofengc/Real-SR
    url = 'https://drive.google.com/uc?export=download&id=1d0dVesJxYw_l1sVgOi4XwIYQ7UalMrCJ'

    logger.info(f"Flickr2K will be saved to: {flickr2k_dir}")
    logger.info("Note: Download URL may require Google Drive access")
    logger.warning("TODO: Replace URL with your actual source or use gdown utility")

    if (flickr2k_dir / 'Flickr2K_HR').exists():
        logger.info("[OK] Flickr2K already present, skipping download")
        return True

    logger.info("Flickr2K download would proceed here with valid URL")
    logger.info("Estimated size: ~28 GB")

    return True


def download_ots(base_dir: Path) -> bool:
    """
    Download OTS (Outdoor haze) dataset (hazy + clear pairs).

    Args:
        base_dir: Base datasets directory

    Returns:
        True if successful
    """
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Downloading OTS")
    logger.info("="*60)

    ots_dir = base_dir / "OTS"

    # Note: OTS available from OneDrive or authors
    # Structure: OTS/original/{0,1,...,N}/ with _hazy.png and _GT.png pairs
    logger.info(f"OTS will be saved to: {ots_dir}")
    logger.warning("TODO: Download from OneDrive or official source")
    logger.info("Dataset structure expected:")
    logger.info("  OTS/original/{0,1,...}/_hazy.png  (degraded)")
    logger.info("  OTS/original/{0,1,...}/_GT.png    (clear ground truth)")

    if (ots_dir / 'original').exists():
        logger.info("[OK] OTS already present, skipping download")
        return True

    logger.info("OTS download would proceed here")
    logger.info("Estimated size: ~10 GB")

    return True


def verify_downloads(base_dir: Path) -> bool:
    """
    Verify all downloaded datasets exist.

    Args:
        base_dir: Base datasets directory

    Returns:
        True if all datasets verified
    """
    logger.info("\n" + "="*60)
    logger.info("Verifying Downloads")
    logger.info("="*60)

    checks = {
        'DIV2K train': base_dir / 'DIV2K' / 'DIV2K_train_HR',
        'DIV2K val': base_dir / 'DIV2K' / 'DIV2K_valid_HR',
        'Flickr2K': base_dir / 'Flickr2K' / 'Flickr2K_HR',
        'OTS': base_dir / 'OTS' / 'original',
    }

    all_ok = True
    for name, path in checks.items():
        if path.exists():
            logger.info(f"[OK] {name}: {path}")
        else:
            logger.warning(f"[ERROR] {name}: NOT FOUND ({path})")
            all_ok = False

    return all_ok


def save_download_log(base_dir: Path, log_data: dict) -> None:
    """Save download metadata to JSON."""
    log_file = base_dir / '_download_log.json'
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2, default=str)
    logger.info(f"[OK] Download log saved: {log_file}")


def main() -> int:
    """
    Main entry point.

    Returns:
        0 if successful, 1 on error
    """
    parser = argparse.ArgumentParser(
        description='Download DIV2K, Flickr2K, and OTS datasets for Real-ESRGAN'
    )
    parser.add_argument(
        '--dataset',
        choices=['div2k', 'flickr2k', 'ots', 'all'],
        default='all',
        help='Which dataset to download (default: all)'
    )
    parser.add_argument(
        '--base-dir',
        type=Path,
        default=Path('datasets'),
        help='Base directory for datasets (default: datasets/)'
    )
    parser.add_argument(
        '--skip-verify',
        action='store_true',
        help='Skip verification check at end'
    )

    args = parser.parse_args()

    # Setup
    base_dir = Path(args.base_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    log_file = base_dir / '_download_log.txt'
    create_logger_file(log_file)

    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Dataset: {args.dataset}")

    # Download based on choice
    success = True
    download_log = {
        'base_dir': str(base_dir),
        'dataset': args.dataset,
        'steps': {}
    }

    if args.dataset in ('div2k', 'all'):
        download_log['steps']['div2k'] = download_div2k(base_dir)
        success = success and download_log['steps']['div2k']

    if args.dataset in ('flickr2k', 'all'):
        download_log['steps']['flickr2k'] = download_flickr2k(base_dir)
        success = success and download_log['steps']['flickr2k']

    if args.dataset in ('ots', 'all'):
        download_log['steps']['ots'] = download_ots(base_dir)
        success = success and download_log['steps']['ots']

    # Verify
    if not args.skip_verify:
        download_log['verified'] = verify_downloads(base_dir)
        success = success and download_log['verified']

    save_download_log(base_dir, download_log)

    if success:
        logger.info("\n[OK] Download completion check PASSED")
        return 0
    else:
        logger.error("\n[ERROR] Some downloads failed or not verified")
        return 1


if __name__ == '__main__':
    sys.exit(main())
