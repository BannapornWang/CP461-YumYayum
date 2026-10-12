# flake8: noqa
import sys
import os.path as osp

from torchvision.transforms import functional as torchvision_functional

# BasicSR 1.4.2 imports this module, removed in newer torchvision releases.
sys.modules.setdefault('torchvision.transforms.functional_tensor', torchvision_functional)

import torch
# Enable TensorFloat-32 (TF32) and cuDNN benchmark for faster training on RTX 50-series
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

from basicsr.train import train_pipeline

import realesrgan.archs
import realesrgan.data
import realesrgan.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
