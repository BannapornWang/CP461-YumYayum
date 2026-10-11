"""Combine evaluation summary JSON files into one ablation table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse input summary files and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True,
                        help="One or more evaluate.py summary.json files")
    parser.add_argument("--output", type=Path, default=Path("results/ablation_summary"))
    return parser.parse_args()


def read_results(path: Path) -> list[dict[str, Any]]:
    """Read a JSON list emitted by project/evaluate.py."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return [{"experiment": path.parent.name, **item} for item in data]


def main() -> None:
    """Write a union-column CSV and Markdown table with measured values only."""
    args = parse_args()
    records = [record for path in args.results for record in read_results(path)]
    if not records:
        raise SystemExit("No measured result rows were found")
    args.output.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in records for key in row})
    with (args.output / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    lines = ["# Ablation summary", "", "Values are copied from evaluation summaries; no missing results are filled in.", "",
             "| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend("| " + " | ".join(str(record.get(field, "")) for field in fields) + " |" for record in records)
    (args.output / "ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
