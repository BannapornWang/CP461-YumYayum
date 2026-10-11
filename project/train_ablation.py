"""Run a BasicSR training config after registering project-local ablations."""

from __future__ import annotations

import sys
from pathlib import Path

from torchvision.transforms import functional as torchvision_functional

# BasicSR 1.4.2 imports this module, removed in newer torchvision releases.
sys.modules.setdefault("torchvision.transforms.functional_tensor", torchvision_functional)

import project.ablation_models  # noqa: F401  Register custom model and discriminator first.
from basicsr.train import train_pipeline
import realesrgan.archs  # noqa: F401
import realesrgan.data  # noqa: F401
import realesrgan.models  # noqa: F401


def main() -> None:
    """Pass the repository root to BasicSR's standard train pipeline."""
    train_pipeline(str(Path(__file__).resolve().parents[1]))


if __name__ == "__main__":
    main()
