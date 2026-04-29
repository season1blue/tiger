#!/usr/bin/env python3
"""
将 data/*.jsonl 中所有 image 路径统一为以项目根（LEAD）为前缀的绝对路径。
不依赖图片是否存在，仅修改路径字符串。
"""

import json
import os
import sys

# 项目根目录（LEAD）：脚本在 script/ 下，上一级为 LEAD
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEAD_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(LEAD_ROOT, "data")


def to_absolute_under_lead(image_path: str, dataset_name: str) -> str:
    """
    将任意 image 路径转换为以 LEAD 为根目录的绝对路径。

    - 若已是绝对路径：取文件名放到 data/images/<dataset_name>/ 下。
    - 若为相对路径：视为相对于 data/，结果为 LEAD/data/<path>。

    Args:
        image_path: 原始 image 路径（相对或绝对）。
        dataset_name: 数据集名称（用于绝对路径时的子目录）。

    Returns:
        以 LEAD 为前缀的绝对路径。
    """
    path = (image_path or "").strip()
    if not path:
        return path

    if os.path.isabs(path):
        # 绝对路径：保留文件名，放到 LEAD/data/images/<dataset_name>/ 下
        filename = os.path.basename(path)
        return os.path.join(LEAD_ROOT, "data", "images", dataset_name, filename)
    # 相对路径：视为在 data 下，即 LEAD/data/<path>
    return os.path.normpath(os.path.join(LEAD_ROOT, "data", path))


def process_jsonl(file_path: str, dataset_name: str) -> int:
    """
    处理单个 JSONL 文件，将每条的 image 改为绝对路径并写回。

    Args:
        file_path: JSONL 文件路径。
        dataset_name: 数据集名称（从文件名推断，如 physunibench）。

    Returns:
        修改的条数。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    updated = 0
    for i, line in enumerate(lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "image" not in obj:
            continue
        old_path = obj.get("image", "")
        new_path = to_absolute_under_lead(old_path, dataset_name)
        if new_path != old_path:
            obj["image"] = new_path
            updated += 1
        lines[i] = json.dumps(obj, ensure_ascii=False)

    with open(file_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    return updated


def main():
    """遍历 data/*.jsonl，统一 image 为 LEAD 下的绝对路径。"""
    if not os.path.isdir(DATA_DIR):
        print(f"Data directory not found: {DATA_DIR}", file=sys.stderr)
        sys.exit(1)

    total_updated = 0
    names = [n for n in os.listdir(DATA_DIR) if n.endswith(".jsonl")]
    for name in sorted(names):
        path = os.path.join(DATA_DIR, name)
        if not os.path.isfile(path):
            continue
        dataset_name = name[:-6]  # 去掉 .jsonl（6 个字符）
        try:
            n = process_jsonl(path, dataset_name)
            total_updated += n
            sys.stdout.write(f"{name}: {n} paths updated\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f"{name}: error - {e}\n")
            sys.stderr.flush()

    sys.stdout.write(f"\nDone. Total paths updated: {total_updated}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
