"""
通用工具函数。

包含设备选择、路径解析、模型加载、JSON 读写等。
"""

import json
import os
import time
from typing import Dict, List, Optional

import torch


def resolve_device(device_str: str = "auto") -> torch.device:
    """
    根据字符串描述解析目标设备。

    Args:
        device_str: "auto"（自动检测）、"cuda"、"cpu" 等。

    Returns:
        torch.device: 解析后的设备对象。
    """
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_str)


def get_project_root() -> str:
    """
    获取项目根目录（main.py 所在目录）。

    Returns:
        str: 项目根目录的绝对路径。
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dir(path: str) -> str:
    """
    确保目录存在，不存在则创建。

    Args:
        path: 目录路径。

    Returns:
        str: 传入的目录路径（方便链式调用）。
    """
    os.makedirs(path, exist_ok=True)
    return path


def save_jsonl(data: List[Dict], path: str) -> None:
    """
    将字典列表写入 JSONL 文件。

    Args:
        data: 字典列表。
        path: 输出文件路径。
    """
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_json(data, path: str, indent: int = 2) -> None:
    """
    将对象写入 JSON 文件。

    Args:
        data: 可序列化的 Python 对象。
        path: 输出文件路径。
        indent: 缩进空格数。
    """
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(path: str):
    """
    从 JSON 文件加载对象。

    Args:
        path: JSON 文件路径。

    Returns:
        解析后的 Python 对象。
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_duration(seconds: float) -> str:
    """
    将秒数格式化为人类可读的时间字符串。

    Args:
        seconds: 秒数。

    Returns:
        str: 如 "1h 23m 45s" 或 "45.2s"。
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


def get_gpu_memory_info() -> Optional[Dict[str, float]]:
    """
    获取当前 GPU 显存使用信息。

    Returns:
        dict: 包含 allocated_gb, reserved_gb, total_gb；无 GPU 返回 None。
    """
    if not torch.cuda.is_available():
        return None
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "total_gb": torch.cuda.get_device_properties(0).total_mem / 1e9,
    }


class Timer:
    """简易计时器，支持 with 语句和手动 start/stop。"""

    def __init__(self, name: str = ""):
        """
        Args:
            name: 计时器名称，用于日志输出。
        """
        self.name = name
        self.start_time = None
        self.elapsed = 0.0

    def start(self):
        """开始计时。"""
        self.start_time = time.time()
        return self

    def stop(self) -> float:
        """
        停止计时并返回耗时。

        Returns:
            float: 经过的秒数。
        """
        if self.start_time is not None:
            self.elapsed = time.time() - self.start_time
            self.start_time = None
        return self.elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __str__(self):
        return format_duration(self.elapsed)
