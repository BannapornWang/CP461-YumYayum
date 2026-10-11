"""Local BasicSR registry extensions used by controlled ablation runs."""

from __future__ import annotations

import random

import numpy as np
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import ARCH_REGISTRY, MODEL_REGISTRY
from torch.nn import functional as F
from torch.nn.utils import remove_spectral_norm

from realesrgan.models.realesrnet_model import RealESRNetModel
from realesrgan.archs.discriminator_arch import UNetDiscriminatorSN


@MODEL_REGISTRY.register()
class FirstOrderRealESRNetModel(RealESRNetModel):
    """Real-ESRNet model using one synthetic blur/noise/compression pass."""

    @torch.no_grad()
    def feed_data(self, data: dict) -> None:
        """Create a single-order LQ image from each supplied GT training image."""
        if not self.is_train:
            super().feed_data(data)
            return
        self.gt = data["gt"].to(self.device)
        if self.opt.get("gt_usm", False):
            self.gt = self.usm_sharpener(self.gt)
        ori_h, ori_w = self.gt.shape[-2:]
        out = filter2D(self.gt, data["kernel1"].to(self.device))
        resize_type = random.choices(["up", "down", "keep"], self.opt["resize_prob"])[0]
        if resize_type == "up":
            factor = float(np.random.uniform(1, self.opt["resize_range"][1]))
        elif resize_type == "down":
            factor = float(np.random.uniform(self.opt["resize_range"][0], 1))
        else:
            factor = 1.0
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, scale_factor=factor, mode=mode)
        gray_prob = self.opt["gray_noise_prob"]
        if np.random.uniform() < self.opt["gaussian_noise_prob"]:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.opt["noise_range"], clip=True, rounds=False, gray_prob=gray_prob)
        else:
            out = random_add_poisson_noise_pt(
                out, scale_range=self.opt["poisson_scale_range"], gray_prob=gray_prob, clip=True, rounds=False)
        out = F.interpolate(out, size=(ori_h // self.opt["scale"], ori_w // self.opt["scale"]),
                            mode=random.choice(["area", "bilinear", "bicubic"]))
        jpeg_quality = out.new_zeros(out.shape[0]).uniform_(*self.opt["jpeg_range"])
        out = self.jpeger(torch.clamp(out, 0, 1), quality=jpeg_quality)
        self.lq = torch.clamp((out * 255).round(), 0, 255) / 255
        self.gt, self.lq = paired_random_crop(self.gt, self.lq, self.opt["gt_size"], self.opt["scale"])
        self._dequeue_and_enqueue()
        self.lq = self.lq.contiguous()


@ARCH_REGISTRY.register()
class UNetDiscriminatorNoSN(UNetDiscriminatorSN):
    """UNetDiscriminatorSN with its spectral-normalization wrappers removed."""

    def __init__(self, num_in_ch: int, num_feat: int = 64, skip_connection: bool = True) -> None:
        super().__init__(num_in_ch=num_in_ch, num_feat=num_feat, skip_connection=skip_connection)
        for module in self.modules():
            try:
                remove_spectral_norm(module)
            except (AttributeError, ValueError):
                pass
