"""
数据集加载与预处理。

支持 JSONL 格式的 PhysUniBench 数据集加载，
自动解析图片相对路径、过滤无效样本、按子领域/难度筛选。
"""

import json
import os
from typing import List, Dict, Optional


def load_jsonl(path: str) -> List[Dict]:
    """
    从 JSONL 文件逐行加载为字典列表。

    Args:
        path: JSONL 文件的绝对或相对路径。

    Returns:
        list[dict]: 每行解析后的字典组成的列表。

    Raises:
        FileNotFoundError: 文件不存在时抛出。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_image_paths(
    dataset: List[Dict],
    data_dir: str,
    image_key: str = "image",
) -> List[Dict]:
    """
    将数据集中的图片相对路径解析为绝对路径。

    Args:
        dataset: 样本字典列表。
        data_dir: 数据根目录（图片相对路径的基准）。
        image_key: 样本中图片路径对应的键名。

    Returns:
        list[dict]: 路径已解析的样本列表（原地修改）。
    """
    for sample in dataset:
        image_url = sample.get(image_key, "")
        if image_url and not os.path.isabs(image_url):
            sample[image_key] = os.path.join(data_dir, image_url)
    return dataset


def filter_by_subtopic(
    dataset: List[Dict],
    subtopics: List[str],
) -> List[Dict]:
    """
    按物理子领域筛选样本。

    Args:
        dataset: 样本字典列表。
        subtopics: 需要保留的子领域列表，如 ["Mechanics", "Optics"]。

    Returns:
        list[dict]: 筛选后的样本列表。
    """
    return [s for s in dataset if s.get("subtopic", "") in subtopics]


def filter_by_difficulty(
    dataset: List[Dict],
    min_difficulty: int = 1,
    max_difficulty: int = 5,
) -> List[Dict]:
    """
    按难度等级筛选样本。

    Args:
        dataset: 样本字典列表。
        min_difficulty: 最低难度（含）。
        max_difficulty: 最高难度（含）。

    Returns:
        list[dict]: 筛选后的样本列表。
    """
    return [
        s for s in dataset
        if min_difficulty <= s.get("difficulty", 0) <= max_difficulty
    ]


def filter_by_language(
    dataset: List[Dict],
    language: str = "english",
) -> List[Dict]:
    """
    按语言筛选样本。

    Args:
        dataset: 样本字典列表。
        language: 目标语言，"english" 或 "chinese"。

    Returns:
        list[dict]: 筛选后的样本列表。
    """
    return [s for s in dataset if s.get("language", "") == language]


def load_dataset(
    dataset_path: str,
    data_dir: str,
    subtopics: Optional[List[str]] = None,
    min_difficulty: int = 1,
    max_difficulty: int = 5,
    language: Optional[str] = None,
) -> List[Dict]:
    """
    完整的数据集加载流程：读取 → 路径解析 → 可选过滤。

    Args:
        dataset_path: JSONL 文件路径。
        data_dir: 数据根目录。
        subtopics: 可选，按子领域过滤。
        min_difficulty: 最低难度（默认 1）。
        max_difficulty: 最高难度（默认 5）。
        language: 可选，按语言过滤。

    Returns:
        list[dict]: 最终的样本列表。
    """
    dataset = load_jsonl(dataset_path)
    dataset = resolve_image_paths(dataset, data_dir)

    if subtopics:
        dataset = filter_by_subtopic(dataset, subtopics)
    if min_difficulty > 1 or max_difficulty < 5:
        dataset = filter_by_difficulty(dataset, min_difficulty, max_difficulty)
    if language:
        dataset = filter_by_language(dataset, language)

    return dataset


def get_dataset_statistics(dataset: List[Dict]) -> Dict:
    """
    统计数据集的基本信息。

    Args:
        dataset: 样本字典列表。

    Returns:
        dict: 包含总数、子领域分布、难度分布、语言分布等统计信息。
    """
    stats = {
        "total": len(dataset),
        "subtopics": {},
        "difficulties": {},
        "languages": {},
    }
    for s in dataset:
        sub = s.get("subtopic", "unknown")
        diff = s.get("difficulty", 0)
        lang = s.get("language", "unknown")
        stats["subtopics"][sub] = stats["subtopics"].get(sub, 0) + 1
        stats["difficulties"][diff] = stats["difficulties"].get(diff, 0) + 1
        stats["languages"][lang] = stats["languages"].get(lang, 0) + 1

    return stats
