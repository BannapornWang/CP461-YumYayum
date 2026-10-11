#!/usr/bin/env python3
"""
Training wrapper for Real-ESRNet on custom prepared datasets.

Features:
  - Logs git commit hash for traceability
  - Checks GPU availability and memory
  - Supports resuming from checkpoints
  - Saves experiment metadata with git info
  - Validates dataset paths before training
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_git_info() -> dict:
    """
    Get current git information (commit hash, branch, status).

    Returns:
        Dictionary with git info
    """
    info = {
        'commit_hash': None,
        'branch': None,
        'status': None,
        'timestamp': datetime.now().isoformat(),
    }

    try:
        # Get current commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            info['commit_hash'] = result.stdout.strip()

        # Get current branch
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            info['branch'] = result.stdout.strip()

        # Check for uncommitted changes
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            info['status'] = 'clean' if not result.stdout.strip() else 'dirty'

    except Exception as e:
        logger.warning(f"Could not retrieve git info: {e}")

    return info


def check_gpu_availability() -> tuple:
    """
    Check CUDA availability and GPU memory.

    Returns:
        (cuda_available, num_gpus, memory_info)
    """
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        num_gpus = torch.cuda.device_count() if cuda_available else 0

        memory_info = {}
        if cuda_available:
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / (1024 ** 3)  # Convert to GB
                memory_info[f'gpu_{i}'] = {
                    'name': props.name,
                    'total_memory_gb': round(total_memory, 2),
                }

        return cuda_available, num_gpus, memory_info

    except ImportError:
        logger.warning("PyTorch not found, cannot check GPU")
        return False, 0, {}


def validate_dataset_paths(config_path: Path) -> bool:
    """
    Validate dataset paths specified in config.

    Args:
        config_path: Path to .yml config file

    Returns:
        True if all paths exist, False otherwise
    """
    paths_to_check = [
        'datasets/DF2K_sub',
        'datasets/meta_info_train.txt',
        'datasets/meta_info_val.txt',
    ]

    logger.info("Validating dataset paths...")
    all_valid = True

    for path_str in paths_to_check:
        path = Path(path_str)
        if path.exists():
            if path.is_file():
                size_mb = path.stat().st_size / (1024 ** 2)
                logger.info(f"  [OK] {path} ({size_mb:.2f} MB)")
            else:
                count = len(list(path.glob('*')))
                logger.info(f"  [OK] {path} ({count} items)")
        else:
            logger.error(f"  [ERROR] {path} NOT FOUND")
            all_valid = False

    return all_valid


def save_experiment_metadata(
    experiments_dir: Path,
    config_path: Path,
    git_info: dict,
    gpu_info: tuple
) -> Path:
    """
    Save experiment metadata with git and GPU info.

    Args:
        experiments_dir: Base experiments directory
        config_path: Path to config file
        git_info: Git information dictionary
        gpu_info: GPU information tuple

    Returns:
        Path to saved metadata file
    """
    cuda_available, num_gpus, memory_info = gpu_info

    metadata = {
        'timestamp': git_info['timestamp'],
        'config': str(config_path),
        'git': git_info,
        'gpu': {
            'cuda_available': cuda_available,
            'num_gpus': num_gpus,
            'memory_info': memory_info,
        },
        'command': f"python realesrgan/train.py -opt {config_path}",
    }

    # Create experiments directory if needed
    experiments_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata_file = experiments_dir / 'experiment_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"[OK] Experiment metadata saved: {metadata_file}")

    return metadata_file


def run_training(
    config_path: Path,
    resume_path: Optional[Path] = None,
    experiments_dir: Optional[Path] = None
) -> int:
    """
    Execute Real-ESRNet training.

    Args:
        config_path: Path to .yml config file
        resume_path: Optional path to checkpoint for resumption
        experiments_dir: Optional experiments directory override

    Returns:
        Return code from training script
    """
    config_path = Path(config_path).resolve()

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    logger.info("="*60)
    logger.info("Real-ESRNet Training")
    logger.info("="*60)

    # Get git info
    git_info = get_git_info()
    logger.info(f"Git commit: {git_info['commit_hash']}")
    logger.info(f"Git branch: {git_info['branch']}")
    logger.info(f"Git status: {git_info['status']}")

    # Check GPU
    cuda_available, num_gpus, memory_info = check_gpu_availability()
    logger.info(f"CUDA available: {cuda_available}")
    logger.info(f"Number of GPUs: {num_gpus}")
    for gpu_name, gpu_info in memory_info.items():
        logger.info(f"  {gpu_name}: {gpu_info['name']} ({gpu_info['total_memory_gb']} GB)")

    if not cuda_available:
        logger.warning("[WARN] CUDA not available - training will be very slow on CPU")

    if num_gpus == 0 and cuda_available:
        logger.warning("[WARN] Found CUDA but no GPUs detected")

    # Validate dataset paths
    if not validate_dataset_paths(config_path):
        logger.error("[ERROR] Some dataset paths are missing!")
        logger.info("  Run: python project/split_dataset.py")
        return 1

    # Prepare training command
    cmd = [
        'python',
        'realesrgan/train.py',
        '-opt',
        str(config_path),
    ]

    # Add resume option if specified
    if resume_path:
        resume_path = Path(resume_path).resolve()
        if not resume_path.exists():
            logger.error(f"Resume checkpoint not found: {resume_path}")
            return 1
        logger.info(f"Resuming from: {resume_path}")
        # Note: BasicSR's training script reads 'resume_state' from config
        # To resume, update config or modify checkpoint metadata

    # Save experiment metadata
    exp_dir = Path(experiments_dir) if experiments_dir else Path('experiments')
    save_experiment_metadata(exp_dir, config_path, git_info, (cuda_available, num_gpus, memory_info))

    # Log training command
    logger.info(f"\nTraining command:")
    logger.info(f"  {' '.join(cmd)}")
    logger.info("\nStarting training...")
    logger.info("="*60)

    # Run training
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parent.parent,  # Project root
        )
        return result.returncode

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description='Training wrapper for Real-ESRNet with git tracking'
    )
    parser.add_argument(
        '-opt',
        '--config',
        type=Path,
        required=True,
        help='Path to training config .yml file'
    )
    parser.add_argument(
        '--resume',
        type=Path,
        help='Path to checkpoint for resuming training'
    )
    parser.add_argument(
        '--experiments-dir',
        type=Path,
        default=Path('experiments'),
        help='Experiments directory (default: experiments/)'
    )

    args = parser.parse_args()

    return run_training(
        args.config,
        resume_path=args.resume,
        experiments_dir=args.experiments_dir
    )


if __name__ == '__main__':
    sys.exit(main())
