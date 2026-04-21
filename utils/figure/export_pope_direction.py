#!/usr/bin/env python3
"""Export one layer-wise direction vector per POPE sample.

The analysis logs already store per-step, per-layer direction matrices in each
sample's .pt file. This script collapses them into a single per-sample curve
and writes a CSV with one row per sample.

Default output columns:
  sample_id, question_id, method_name, hallucinated_or_not, layer_1 ... layer_28
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _to_float_array(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        arr = x.detach().to(torch.float32).cpu().numpy() if x.dtype in (torch.bfloat16, torch.float16) else x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.size == 0:
        return None
    return arr.astype(np.float64)


def _label_hallucinated(index_rec: dict[str, Any]) -> int | None:
    hall = index_rec.get("is_hallucinated", None)
    if hall is not None:
        try:
            return int(hall)
        except Exception:
            pass

    corr = index_rec.get("final_prediction_correct", None)
    if corr is None:
        return None
    return 0 if bool(corr) else 1


def _infer_method_name(idx_file: Path, analysis_root: Path) -> str:
    try:
        rel_parts = idx_file.relative_to(analysis_root).parts
    except ValueError:
        rel_parts = idx_file.parts

    if "analysis" in rel_parts:
        prefix = rel_parts[: rel_parts.index("analysis")]
        if prefix:
            method = prefix[0]
            return "base" if method == "baseline" else method

    if analysis_root.name == "analysis":
        method = analysis_root.parent.name
        return "base" if method == "baseline" else method

    method = analysis_root.name
    return "base" if method == "baseline" else method


def _collapse_direction(direction: np.ndarray, layer_count: int) -> list[float]:
    """Collapse [steps, layers] into a single per-layer curve.

    We keep the first `layer_count` entries after averaging across decoding steps.
    If the array is longer, it is truncated. If it is shorter, it is padded with NaN.
    """

    if direction.ndim != 2:
        return [float("nan")] * layer_count

    curve = np.nanmean(direction, axis=0)

    if curve.size >= layer_count:
        return [float(v) for v in curve[:layer_count]]

    padded = [float(v) for v in curve.tolist()]
    padded.extend([float("nan")] * (layer_count - curve.size))
    return padded


def export_direction_csv(analysis_root: Path, out_csv: Path, layer_count: int = 28) -> None:
    rows: list[dict[str, Any]] = []
    index_files = sorted(analysis_root.rglob("analysis_index.jsonl"))

    for idx_file in index_files:
        method_name = _infer_method_name(idx_file, analysis_root)
        with idx_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                idx_rec = json.loads(line)
                pt_path = Path(idx_rec.get("log_path", ""))
                if not pt_path.exists():
                    continue

                try:
                    rec = torch.load(pt_path, map_location="cpu")
                except Exception:
                    continue

                direction = _to_float_array(rec.get("direction_scores"))
                if direction is None:
                    continue

                curve = _collapse_direction(direction, layer_count)
                row: dict[str, Any] = {
                    "sample_id": idx_rec.get("sample_id"),
                    "question_id": idx_rec.get("question_id"),
                    "method_name": method_name,
                    "hallucinated_or_not": _label_hallucinated(idx_rec),
                }
                for i, value in enumerate(curve, start=1):
                    row[f"layer_{i}"] = value
                rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "question_id", "method_name", "hallucinated_or_not"] + [f"layer_{i}" for i in range(1, layer_count + 1)]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-sample 28-layer direction curves for POPE analysis logs")
    parser.add_argument(
        "--analysis-root",
        type=str,
        default="/mnt/data/ssz/mr/results/Qwen2.5-VL/pope/base/analysis",
        help="Root folder containing split subfolders with analysis_index.jsonl",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="/mnt/data/ssz/mr/utils/figure/output/pope_direction_28.csv",
        help="CSV file to write",
    )
    parser.add_argument(
        "--layer-count",
        type=int,
        default=28,
        help="Number of layer-wise direction values to export per sample",
    )
    args = parser.parse_args()

    export_direction_csv(Path(args.analysis_root), Path(args.output_csv), layer_count=args.layer_count)
    print(f"[done] wrote {args.output_csv}")


if __name__ == "__main__":
    main()