import logging
import os
from typing import Any, Dict, List, Tuple

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)


def _safe_text(value: Any) -> str:
    """将任意值安全转换为字符串，避免 None 导致长度计算失败。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _resolve_prompt(row: Dict[str, Any]) -> str:
    """尽可能从不同字段中解析 prompt。"""
    for key in ["prompt", "instruction", "query", "question"]:
        if key in row:
            return _safe_text(row[key])
    raise KeyError("未找到 prompt 相关字段，请检查数据集字段。")


def _resolve_pair(row: Dict[str, Any]) -> Tuple[str, str]:
    """
    解析 chosen/rejected 对。
    支持 {chosen, rejected} 或 {response_0, response_1} + 偏好标签。
    """
    if "chosen" in row and "rejected" in row:
        return _safe_text(row["chosen"]), _safe_text(row["rejected"])

    if "response_0" in row and "response_1" in row:
        response_0 = _safe_text(row["response_0"])
        response_1 = _safe_text(row["response_1"])
        better_key_candidates = [
            "better_response_id",
            "safer_response_id",
            "preferred_response_id",
            "label",
        ]
        better_id = None
        for key in better_key_candidates:
            if key in row:
                better_id = row[key]
                break
        if better_id in [0, "0"]:
            return response_0, response_1
        if better_id in [1, "1"]:
            return response_1, response_0
        logger.warning(
            "未识别偏好标签，默认使用 response_0 作为 chosen。"
        )
        return response_0, response_1

    raise KeyError("未找到 chosen/rejected 或 response_0/response_1 字段。")


def load_and_process_data(
    dataset_name: str = "PKU-Alignment/PKU-SafeRLHF-10K",
    split: str = "train[:2000]",
    cache_dir: str = "data",
) -> pd.DataFrame:
    """
    加载并处理 DPO 数据集，返回包含长度统计的 DataFrame。
    """
    os.makedirs(cache_dir, exist_ok=True)
    try:
        # 从 Hugging Face Hub 加载数据集并使用缓存目录加速二次加载
        dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    except Exception as exc:
        raise RuntimeError(
            f"数据集加载失败：{dataset_name} / {split}。"
            "请检查网络或数据集名称。"
        ) from exc

    if len(dataset) == 0:
        raise RuntimeError("数据集为空，请检查 split 配置。")

    records: List[Dict[str, Any]] = []
    for row in dataset:
        # 解析 prompt 与 chosen/rejected，对齐为 DPO 训练格式
        prompt = _resolve_prompt(row)
        chosen, rejected = _resolve_pair(row)
        # 计算文本长度，便于后续统计与可视化分析
        records.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "prompt_len": len(prompt),
                "chosen_len": len(chosen),
                "rejected_len": len(rejected),
            }
        )

    df = pd.DataFrame.from_records(records)
    return df
