"""Evaluate x4 super-resolution checkpoints on paired synthetic test sets."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from realesrgan import RealESRGANer

SEED = 461
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    """Parse evaluation, visualization, and user-study arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=[], metavar="NAME=CHECKPOINT",
                        help="RRDBNet x4 checkpoints; include ESRGAN, BSRGAN, Ours labels as desired")
    parser.add_argument("--test_set", type=Path, required=True,
                        help="Paired HR/LR_x4 split, parent with easy/medium/hard, or flat real-world image folder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tile", type=int, default=0, help="Inference tile size; 0 disables tiling")
    parser.add_argument("--crop-count", type=int, choices=range(4, 7), default=5)
    parser.add_argument("--no-iqa", action="store_true", help="Skip optional pyiqa metrics")
    parser.add_argument("--visualize", action="store_true", help="Write comparison panels and crop examples")
    parser.add_argument("--blind-study", action="store_true", help="Create randomized, anonymized image sets")
    parser.add_argument("--study-scores", type=Path, help="Optional MOS CSV to summarize")
    return parser.parse_args()


def image_pairs(split: Path) -> list[tuple[Path, Path]]:
    """Return matching LR and HR images for one paired split."""
    lr_dir, hr_dir = split / "LR_x4", split / "HR"
    if not lr_dir.is_dir() or not hr_dir.is_dir():
        raise FileNotFoundError(f"Expected {lr_dir} and {hr_dir}")
    pairs = [(lr, hr_dir / lr.name) for lr in sorted(lr_dir.iterdir())
             if lr.suffix.lower() in IMAGE_SUFFIXES and (hr_dir / lr.name).is_file()]
    if not pairs:
        raise ValueError(f"No matched image pairs in {split}")
    return pairs


def load_iqa(device: torch.device) -> dict[str, Any]:
    """Create available pyiqa metrics; absence is reported in the run manifest."""
    try:
        import pyiqa
    except ImportError:
        return {}
    metrics: dict[str, Any] = {}
    for name in ("lpips", "dists", "niqe", "musiq", "clipiqa"):
        try:
            metrics[name] = pyiqa.create_metric(name, device=device, as_loss=False)
        except Exception as exc:
            print(f"Skipping pyiqa metric {name}: {exc}")
    return metrics


def overshoot_ratio(pred: np.ndarray, reference: np.ndarray) -> float:
    """Measure out-of-range predictions on sharp reference edges."""
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    edge_mask = cv2.Canny(ref.astype(np.uint8), 80, 160) > 0
    if not np.any(edge_mask):
        return 0.0
    pred_gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_min = cv2.erode(ref, np.ones((3, 3), np.uint8))
    local_max = cv2.dilate(ref, np.ones((3, 3), np.uint8))
    overshoot = (pred_gray < local_min - 2) | (pred_gray > local_max + 2)
    return float(np.count_nonzero(overshoot & edge_mask) / np.count_nonzero(edge_mask))


def evaluate_split(
    split: Path, model_specs: list[tuple[str, Path]], output: Path, device: torch.device,
    tile: int, use_iqa: bool, crop_count: int, visualize: bool,
) -> list[dict[str, Any]]:
    """Evaluate available models and save per-image outputs and measurements."""
    pairs = image_pairs(split)
    model_specs = [("Bicubic", Path())] + model_specs
    iqa = load_iqa(device) if use_iqa else {}
    rows: list[dict[str, Any]] = []
    for model_name, checkpoint in model_specs:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        restorer = None
        model = None
        if model_name != "Bicubic":
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                            num_grow_ch=32, scale=4)
            restorer = RealESRGANer(scale=4, model_path=str(checkpoint), model=model,
                                    tile=tile, half=device.type == "cuda",
                                    gpu_id=device.index if device.type == "cuda" else None)
        elapsed = 0.0
        for lr_path, hr_path in pairs:
            lr = cv2.imread(str(lr_path), cv2.IMREAD_COLOR)
            hr = cv2.imread(str(hr_path), cv2.IMREAD_COLOR)
            if lr is None or hr is None:
                raise ValueError(f"Could not read pair {lr_path}, {hr_path}")
            start = time.perf_counter()
            if model_name == "Bicubic":
                pred = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)
            else:
                pred, _ = restorer.enhance(lr, outscale=4)
                if pred.shape[:2] != hr.shape[:2]:
                    pred = cv2.resize(pred, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)
            elapsed += time.perf_counter() - start
            record: dict[str, Any] = {
                "split": split.name, "image": lr_path.name, "model": model_name,
                "psnr": calculate_psnr(pred, hr, crop_border=4),
                "ssim": calculate_ssim(pred, hr, crop_border=4),
                "overshoot_ratio": overshoot_ratio(pred, hr),
            }
            if iqa:
                pred_tensor = torch.from_numpy(cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None] / 255
                hr_tensor = torch.from_numpy(cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None] / 255
                pred_tensor, hr_tensor = pred_tensor.to(device), hr_tensor.to(device)
                with torch.inference_mode():
                    for metric_name, metric in iqa.items():
                        try:
                            args = (pred_tensor, hr_tensor) if metric_name in {"lpips", "dists"} else (pred_tensor,)
                            record[metric_name] = float(metric(*args).mean().item())
                        except Exception as exc:
                            print(f"Skipping {metric_name} for {lr_path.name}: {exc}")
            rows.append(record)
            pred_dir = output / "predictions" / split.name / model_name
            pred_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(pred_dir / lr_path.name), pred)
            del pred, hr, lr
            gc.collect()
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
        params = sum(p.numel() for p in model.parameters()) if model is not None else 0
        for row in rows:
            if row["split"] == split.name and row["model"] == model_name:
                row["seconds_per_image"] = elapsed / len(pairs)
                row["peak_gpu_memory_mb"] = peak
                row["parameters"] = params
        del restorer, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if visualize:
        write_visuals(split, pairs, output, crop_count, rows)
    return rows


def evaluate_unpaired(
    split: Path, model_specs: list[tuple[str, Path]], output: Path, device: torch.device, tile: int,
    use_iqa: bool,
) -> list[dict[str, Any]]:
    """Evaluate no-reference quality and efficiency for real-world images."""
    image_paths = sorted(path for path in split.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise ValueError(f"No images found for no-reference evaluation in {split}")
    models = [("Bicubic", Path())] + model_specs
    iqa = load_iqa(device) if use_iqa else {}
    no_ref = {key: value for key, value in iqa.items() if key in {"niqe", "musiq", "clipiqa"}}
    if not no_ref:
        print("No pyiqa no-reference metrics available; install requirements-evaluation.txt")
    rows: list[dict[str, Any]] = []
    for name, checkpoint in models:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model = None
        restorer = None
        if name != "Bicubic":
            model = RRDBNet(3, 3, 64, 23, 32, scale=4)
            restorer = RealESRGANer(scale=4, model_path=str(checkpoint), model=model, tile=tile,
                                    half=device.type == "cuda", gpu_id=device.index if device.type == "cuda" else None)
        elapsed = 0.0
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            start = time.perf_counter()
            if name == "Bicubic":
                pred = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            else:
                pred, _ = restorer.enhance(image, outscale=4)
            elapsed += time.perf_counter() - start
            record: dict[str, Any] = {"split": split.name, "image": image_path.name, "model": name}
            tensor = torch.from_numpy(cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None].to(device) / 255
            with torch.inference_mode():
                for metric_name, metric in no_ref.items():
                    try:
                        record[metric_name] = float(metric(tensor).mean().item())
                    except Exception as exc:
                        print(f"Skipping {metric_name} for {image_path.name}: {exc}")
            rows.append(record)
            pred_dir = output / "predictions" / split.name / name
            pred_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(pred_dir / image_path.name), pred)
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
        params = sum(param.numel() for param in model.parameters()) if model is not None else 0
        count = max(1, len(image_paths))
        for row in rows:
            if row["split"] == split.name and row["model"] == name:
                row["seconds_per_image"] = elapsed / count
                row["peak_gpu_memory_mb"] = peak
                row["parameters"] = params
        del restorer, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def write_visuals(split: Path, pairs: list[tuple[Path, Path]],
                  output: Path, crop_count: int,
                  rows: list[dict[str, Any]]) -> None:
    """Save best/worst versus bicubic panels with four to six zoom crops."""
    target = output / "visuals" / split.name
    target.mkdir(parents=True, exist_ok=True)
    candidates = [row for row in rows if row["model"] != "Bicubic"]
    preferred = [row for row in candidates
                 if any(tag in row["model"].lower() for tag in ("ours", "team", "realesrnet"))]
    selected_model = preferred or candidates
    by_image = {row["image"]: row for row in selected_model}
    bicubic_scores = {row["image"]: float(row["psnr"]) for row in rows if row["model"] == "Bicubic"}
    ranked = sorted(by_image, key=lambda name: float(by_image[name]["psnr"]) - bicubic_scores.get(name, 0.0))
    chosen = list(dict.fromkeys([ranked[-1], ranked[0]])) if ranked else []
    (target / "selected_cases.json").write_text(json.dumps({
        "best_vs_bicubic": ranked[-1] if ranked else None,
        "worst_vs_bicubic": ranked[0] if ranked else None,
        "selection_metric": "PSNR improvement relative to bicubic on paired HR",
    }, indent=2), encoding="utf-8")
    for lr_path, hr_path in pairs:
        if lr_path.name not in chosen:
            continue
        lr = cv2.imread(str(lr_path))
        hr = cv2.imread(str(hr_path))
        panels: list[tuple[str, np.ndarray]] = [("Input", cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_NEAREST))]
        bicubic_file = output / "predictions" / split.name / "Bicubic" / lr_path.name
        panels.append(("Bicubic", cv2.imread(str(bicubic_file)) if bicubic_file.is_file() else np.zeros_like(hr)))
        aliases = {"ESRGAN": ("esrgan",), "BSRGAN": ("bsrgan", "realsr"),
                   "Ours": ("ours", "team", "realesrnet")}
        model_dirs = [p for p in (output / "predictions" / split.name).iterdir() if p.is_dir() and p.name != "Bicubic"]
        for label, tokens in aliases.items():
            match_dir = next((p for p in model_dirs if any(token in p.name.lower() for token in tokens)), None)
            pred_file = match_dir / lr_path.name if match_dir else None
            if pred_file and pred_file.is_file():
                panels.append((label, cv2.imread(str(pred_file))))
            else:
                placeholder = np.zeros_like(hr)
                cv2.putText(placeholder, f"Not run: {label}", (20, hr.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                panels.append((label, placeholder))
        panels.append(("GT", hr))
        labeled = []
        for label, image in panels:
            img = image.copy()
            cv2.putText(img, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            labeled.append(img)
        cv2.imwrite(str(target / f"{lr_path.stem}_comparison.png"), np.concatenate(labeled, axis=1))
        height, width = hr.shape[:2]
        crop_size = max(32, min(height, width) // 5)
        centers = [(0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75), (0.5, 0.25)][:crop_count]
        for idx, (cx, cy) in enumerate(centers, 1):
            x0 = max(0, min(width - crop_size, int(cx * width - crop_size / 2)))
            y0 = max(0, min(height - crop_size, int(cy * height - crop_size / 2)))
            crop_panels = [cv2.resize(img[y0:y0 + crop_size, x0:x0 + crop_size], (crop_size * 3, crop_size * 3), interpolation=cv2.INTER_NEAREST)
                           for _, img in panels]
            cv2.imwrite(str(target / f"{lr_path.stem}_crop{idx}.png"), np.concatenate(crop_panels, axis=1))


def write_reports(rows: list[dict[str, Any]], output: Path) -> None:
    """Write raw measurements, grouped summaries, Markdown, and optional charts."""
    output.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["split"], row["model"]), []).append(row)
    summary = []
    for (split, model), records in groups.items():
        item: dict[str, Any] = {"split": split, "model": model, "images": len(records)}
        for key in ("psnr", "ssim", "lpips", "dists", "niqe", "musiq", "clipiqa", "overshoot_ratio", "seconds_per_image", "peak_gpu_memory_mb", "parameters"):
            values = [float(record[key]) for record in records if record.get(key) is not None]
            if values:
                item[key] = sum(values) / len(values)
        summary.append(item)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    keys = sorted({key for item in summary for key in item})
    lines = ["# Evaluation summary", "", "Metrics are means over measured images. Missing metrics indicate an unavailable optional dependency or checkpoint.", "", "| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for item in summary:
        lines.append("| " + " | ".join(str(item.get(key, "")) for key in keys) + " |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; CSV/Markdown were written, charts skipped")
        return
    for metric in ("psnr", "ssim", "overshoot_ratio", "seconds_per_image", "peak_gpu_memory_mb"):
        for split in sorted({row["split"] for row in summary}):
            subset = [row for row in summary if row["split"] == split and metric in row]
            if not subset:
                continue
            plt.figure(figsize=(max(6, len(subset)), 4))
            plt.bar([row["model"] for row in subset], [row[metric] for row in subset])
            plt.ylabel(metric)
            plt.title(f"{metric}: {split}")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            plt.savefig(output / f"{split}_{metric}.png", dpi=160)
            plt.close()


def create_blind_study(rows: list[dict[str, Any]], test_root: Path, output: Path) -> None:
    """Create shuffled anonymous image copies and a separate answer key."""
    study_dir = output / "blind_study"
    answer: list[dict[str, str]] = []
    candidates = [row for row in rows if row["model"] != "Bicubic"]
    for row in candidates:
        source = output / "predictions" / row["split"] / row["model"] / row["image"]
        if not source.is_file():
            continue
        answer.append({"split": row["split"], "image": row["image"], "method": row["model"], "source": str(source)})
    random.Random(SEED).shuffle(answer)
    for idx, item in enumerate(answer, 1):
        item["study_id"] = f"S{idx:05d}"
        destination = study_dir / item["study_id"] / "image.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["source"], destination)
    admin_dir = output / "blind_study_admin"
    admin_dir.mkdir(parents=True, exist_ok=True)
    (admin_dir / "answer_key.json").write_text(json.dumps(answer, indent=2), encoding="utf-8")
    (study_dir / "scores_template.csv").write_text("participant_id,study_id,score_1_to_5,comment\n", encoding="utf-8")


def summarize_scores(score_file: Path, output: Path) -> None:
    """Summarize submitted MOS scores without inventing missing ratings."""
    with score_file.open(newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))
    values: dict[str, list[float]] = {}
    for record in records:
        try:
            score = float(record["score_1_to_5"])
        except (KeyError, ValueError):
            continue
        if 1 <= score <= 5:
            values.setdefault(record["study_id"], []).append(score)
    key_path = output / "blind_study_admin" / "answer_key.json"
    id_to_method = {}
    if key_path.is_file():
        id_to_method = {item["study_id"]: item["method"] for item in json.loads(key_path.read_text(encoding="utf-8"))}
    methods: dict[str, list[float]] = {}
    for study_id, scores in values.items():
        if study_id in id_to_method:
            methods.setdefault(id_to_method[study_id], []).extend(scores)
    with (output / "mos_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["level", "id_or_method", "n", "mos"])
        for study_id, scores in sorted(values.items()):
            writer.writerow(["item", study_id, len(scores), sum(scores) / len(scores)])
        for method, scores in sorted(methods.items()):
            writer.writerow(["method", method, len(scores), sum(scores) / len(scores)])


def main() -> None:
    """Run paired evaluation and write requested artifacts."""
    args = parse_args()
    random.seed(SEED)
    try:
        np.random.seed(SEED)
    except Exception:
        pass
    torch.manual_seed(SEED)
    device = torch.device(args.device)
    model_specs = []
    for raw in args.models:
        if "=" not in raw:
            raise SystemExit(f"Invalid model specification {raw!r}; use NAME=CHECKPOINT")
        name, path = raw.split("=", 1)
        checkpoint = Path(path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        model_specs.append((name, checkpoint))
    splits = [args.test_set / level for level in ("easy", "medium", "hard") if (args.test_set / level).is_dir()]
    if not splits:
        splits = [args.test_set]
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for split in splits:
        if (split / "HR").is_dir() and (split / "LR_x4").is_dir():
            rows = evaluate_split(split, model_specs, args.output, device, args.tile,
                                  not args.no_iqa, args.crop_count, args.visualize)
        else:
            rows = evaluate_unpaired(split, model_specs, args.output, device, args.tile, not args.no_iqa)
        all_rows.extend(rows)
    write_reports(all_rows, args.output)
    if args.blind_study:
        create_blind_study(all_rows, args.test_set, args.output)
    if args.study_scores:
        summarize_scores(args.study_scores, args.output)


if __name__ == "__main__":
    main()
