"""Generate paired, equal-iteration ablation configs from repository configs."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    """Parse config-generation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=45000,
                        help="Use this exact iteration count in every generated config")
    parser.add_argument("--output", type=Path, default=Path("options/ablation"))
    parser.add_argument("--realesrnet-init", type=Path,
                        default=Path("experiments/train_RealESRNetx4_CP461_team_5h/models/net_g_3000.pth"),
                        help="Generator checkpoint used to initialize the Real-ESRGAN ablation")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load one BasicSR YAML config."""
    with path.open(encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return result


def dump_config(config: dict[str, Any], path: Path) -> None:
    """Write a reproducible config with stable key ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def set_common(config: dict[str, Any], name: str, iterations: int) -> None:
    """Align iteration count, dataset, and RTX 3060 patch/batch settings."""
    config["name"] = name
    config["manual_seed"] = 461
    config["train"]["total_iter"] = iterations
    train_data = config["datasets"]["train"]
    train_data["dataroot_gt"] = "datasets/DF2K_sub"
    train_data["meta_info"] = "datasets/meta_info_train.txt"
    train_data["gt_size"] = 192
    train_data["batch_size_per_gpu"] = 4
    train_data["num_worker_per_gpu"] = 4
    config["gt_size"] = 192
    config["queue_size"] = 180
    config["logger"]["save_checkpoint_freq"] = 1000
    config["dist_params"]["backend"] = "gloo"


def main() -> None:
    """Write A1–A6 config pairs from the checked-in BasicSR options."""
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    esrnet = load_config(root / "options/custom/train_realesrnet_x4_5h.yml")
    esrgan = load_config(root / "options/train_realesrgan_x4plus.yml")
    for variant in (esrnet, esrgan):
        set_common(variant, "ablation", args.iterations)

    configs: dict[str, dict[str, Any]] = {}
    configs["A1_second_order"] = deepcopy(esrnet)
    configs["A1_first_order"] = deepcopy(esrnet)
    configs["A1_first_order"]["model_type"] = "FirstOrderRealESRNetModel"

    configs["A2_sinc_normal"] = deepcopy(esrnet)
    configs["A2_sinc_disabled"] = deepcopy(esrnet)
    for key in ("sinc_prob", "sinc_prob2", "final_sinc_prob"):
        configs["A2_sinc_disabled"]["datasets"]["train"][key] = 0.0

    configs["A3_unet_sn"] = deepcopy(esrgan)
    configs["A3_vgg_discriminator"] = deepcopy(esrgan)
    configs["A3_unet_sn"]["path"]["pretrain_network_g"] = None
    configs["A3_vgg_discriminator"]["path"]["pretrain_network_g"] = None
    configs["A3_vgg_discriminator"]["network_d"] = {
        "type": "VGGStyleDiscriminator", "num_in_ch": 3, "num_feat": 64, "input_size": 128}
    configs["A3_vgg_discriminator"]["gt_size"] = 128
    configs["A3_vgg_discriminator"]["datasets"]["train"]["gt_size"] = 128
    configs["A3_unet_sn"]["gt_size"] = 128
    configs["A3_unet_sn"]["datasets"]["train"]["gt_size"] = 128

    configs["A4_spectral_norm"] = deepcopy(esrgan)
    configs["A4_without_spectral_norm"] = deepcopy(esrgan)
    configs["A4_spectral_norm"]["path"]["pretrain_network_g"] = None
    configs["A4_without_spectral_norm"]["path"]["pretrain_network_g"] = None
    configs["A4_without_spectral_norm"]["network_d"]["type"] = "UNetDiscriminatorNoSN"

    configs["A5_realesrnet"] = deepcopy(esrnet)
    configs["A5_realesrgan"] = deepcopy(esrgan)
    if args.realesrnet_init:
        init_path = args.realesrnet_init
        if init_path.is_absolute():
            try:
                init_path = init_path.relative_to(root)
            except ValueError:
                pass
        configs["A5_realesrgan"]["path"]["pretrain_network_g"] = str(init_path).replace("\\", "/")
        configs["A5_realesrgan"]["path"]["param_key_g"] = "params_ema"
        configs["A5_realesrnet"]["path"]["pretrain_network_g"] = str(init_path).replace("\\", "/")
        configs["A5_realesrnet"]["path"]["param_key_g"] = "params_ema"

    configs["A6_patch_192"] = deepcopy(esrnet)
    configs["A6_patch_160"] = deepcopy(esrnet)
    configs["A6_patch_160"]["gt_size"] = 160
    configs["A6_patch_160"]["datasets"]["train"]["gt_size"] = 160

    for name, config in configs.items():
        config["name"] = name
        destination = args.output / f"{name}.yml"
        dump_config(config, destination if destination.is_absolute() else root / destination)
    print(f"Wrote {len(configs)} configs to {args.output}")


if __name__ == "__main__":
    main()
