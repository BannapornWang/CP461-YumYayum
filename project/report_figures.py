"""Create report figures from actual BasicSR logs and evaluation CSV files."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse log, evaluation, edge-profile, and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, action="append", default=[], help="BasicSR log file; can repeat")
    parser.add_argument("--metrics", type=Path, action="append", default=[], help="evaluate.py metrics.csv; can repeat")
    parser.add_argument("--edge-pair", action="append", default=[], metavar="PREDICTION=REFERENCE",
                        help="Create an edge profile from a real prediction/reference pair")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def loss_points(log_paths: list[Path]) -> list[tuple[str, int, float]]:
    """Parse iteration/loss points from BasicSR log lines."""
    pattern = re.compile(r"iter:\s*([\d,]+).*?l_(?:pix|percep|gan|d)_\w*:\s*([0-9.eE+-]+)")
    points = []
    for path in log_paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = pattern.search(line)
            if match:
                points.append((path.stem, int(match.group(1).replace(",", "")), float(match.group(2))))
    return points


def plot_edge_profile(pred_path: Path, ref_path: Path, output: Path) -> None:
    """Plot grayscale intensity along the strongest reference edge."""
    pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
    ref = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
    if pred is None or ref is None:
        raise ValueError(f"Could not read edge pair: {pred_path}, {ref_path}")
    if pred.shape != ref.shape:
        pred = cv2.resize(pred, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_CUBIC)
    gx = cv2.Sobel(ref, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ref, cv2.CV_32F, 0, 1, ksize=3)
    y, x = np.unravel_index(np.argmax(cv2.magnitude(gx, gy)), ref.shape)
    if abs(float(gx[y, x])) >= abs(float(gy[y, x])):
        start, end = max(0, x - 24), min(ref.shape[1], x + 25)
        ref_line, pred_line = ref[y, start:end], pred[y, start:end]
        axis = np.arange(start - x, end - x)
    else:
        start, end = max(0, y - 24), min(ref.shape[0], y + 25)
        ref_line, pred_line = ref[start:end, x], pred[start:end, x]
        axis = np.arange(start - y, end - y)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(axis, ref_line, label="Reference")
    plt.plot(axis, pred_line, label="Prediction")
    plt.xlabel("Pixels across strongest edge")
    plt.ylabel("Grayscale intensity")
    plt.title("Edge profile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def main() -> None:
    """Save loss curves, metric bars, peak-memory bars, and edge profiles."""
    import matplotlib.pyplot as plt

    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    points = loss_points(args.log)
    if points:
        plt.figure(figsize=(8, 4))
        for label in sorted({item[0] for item in points}):
            series = [(iteration, value) for name, iteration, value in points if name == label]
            plt.plot([item[0] for item in series], [item[1] for item in series], label=label)
        plt.xlabel("Iteration")
        plt.ylabel("Logged loss")
        plt.title("Training loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output / "loss_curve.png", dpi=180)
        plt.close()
    records = []
    for csv_path in args.metrics:
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            records.extend(csv.DictReader(handle))
    for metric in ("psnr", "ssim", "overshoot_ratio", "peak_gpu_memory_mb"):
        grouped: dict[tuple[str, str], list[float]] = {}
        for record in records:
            try:
                grouped.setdefault((record["split"], record["model"]), []).append(float(record[metric]))
            except (KeyError, ValueError):
                continue
        if not grouped:
            continue
        labels = [f"{split}/{model}" for split, model in grouped]
        means = [sum(values) / len(values) for values in grouped.values()]
        plt.figure(figsize=(max(7, len(labels) * 0.8), 4))
        plt.bar(labels, means)
        plt.ylabel(metric)
        plt.title(metric)
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(args.output / f"{metric}_bar.png", dpi=180)
        plt.close()
    for idx, value in enumerate(args.edge_pair, 1):
        if "=" not in value:
            raise SystemExit("--edge-pair must be PREDICTION=REFERENCE")
        prediction, reference = (Path(item) for item in value.split("=", 1))
        plot_edge_profile(prediction, reference, args.output / f"edge_profile_{idx}.png")


if __name__ == "__main__":
    main()
