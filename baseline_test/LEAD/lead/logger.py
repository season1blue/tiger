"""
日志系统。

提供统一的日志配置，支持同时输出到控制台和文件。
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "lead",
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    log_to_file: bool = True,
) -> logging.Logger:
    """
    创建并配置 logger。

    Args:
        name: logger 名称。
        log_dir: 日志文件保存目录（默认为项目 output/logs/）。
        level: 日志级别。
        log_to_file: 是否同时写入文件。

    Returns:
        logging.Logger: 配置好的 logger 实例。
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    if log_to_file and log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"run_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.info(f"Log file: {log_file}")

    return logger


def get_logger(name: str = "lead") -> logging.Logger:
    """
    获取已存在的 logger，若不存在则创建默认 logger。

    Args:
        name: logger 名称。

    Returns:
        logging.Logger: logger 实例。
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name, log_to_file=False)
    return logger


class LogSection:
    """上下文管理器：为一段逻辑添加开始/结束日志。"""

    def __init__(self, logger: logging.Logger, section_name: str):
        """
        Args:
            logger: logger 实例。
            section_name: 段落名称。
        """
        self.logger = logger
        self.section_name = section_name

    def __enter__(self):
        self.logger.info(f"{'='*20} {self.section_name} START {'='*20}")
        return self

    def __exit__(self, *args):
        self.logger.info(f"{'='*20} {self.section_name} END {'='*20}")
