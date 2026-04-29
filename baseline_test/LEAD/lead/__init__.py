"""
LEAD — 核心方法包。

提供基于熵自适应切换的软/硬解码生成策略，适用于视觉语言模型推理任务。

模块列表：
    - generation_utils: LEAD 与 CoT 生成核心算法
    - inference:        输入构建与单样本推理
    - data:             数据集加载与预处理
    - evaluator:        答案评估与准确率统计
    - prompts:          提示模板管理
    - logger:           日志系统
    - utils:            通用工具函数
"""

from .generation_utils import (
    set_seed,
    get_math_symbols_ids,
    generate_cot,
    generate_lead,
    apply_sampling_filter,
)
from .inference import prepare_inputs, run_single_inference
from .data import load_dataset, load_jsonl, get_dataset_statistics
from .evaluator import (
    evaluate_dataset,
    evaluate_single,
    extract_mcq_answer,
    print_evaluation_report,
    save_evaluation_report,
)
from .prompts import (
    format_prompt_from_sample,
    format_mcq_prompt,
    format_open_ended_prompt,
    build_chat_messages,
)
from .logger import setup_logger, get_logger, LogSection
from .utils import (
    resolve_device,
    get_project_root,
    ensure_dir,
    save_jsonl,
    save_json,
    load_json,
    format_duration,
    get_gpu_memory_info,
    Timer,
)
