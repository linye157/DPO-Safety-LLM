import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import seaborn as sns


def plot_data_distribution(df, save_path: str) -> None:
    """
    绘制 prompt 与 response 长度分布直方图。
    """
    required_cols = {"prompt_len", "chosen_len", "rejected_len"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"缺少必要列：{missing}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    # 叠加三条直方图，直观比较长度分布
    sns.histplot(df["prompt_len"], bins=40, color="#1f77b4", label="Prompt")
    sns.histplot(
        df["chosen_len"], bins=40, color="#2ca02c", label="Chosen"
    )
    sns.histplot(
        df["rejected_len"], bins=40, color="#d62728", label="Rejected"
    )
    plt.title("Prompt & Response Length Distribution")
    plt.xlabel("Length")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_training_loss(
    log_history: List[Dict[str, Any]],
    save_path: str,
) -> None:
    """
    从 Trainer 日志中提取 loss 并绘制随 step 变化曲线。
    """
    steps: List[int] = []
    losses: List[float] = []
    for item in log_history:
        if "loss" in item:
            # 保留 step 以便绘制真实的训练进度曲线
            step = item.get("step", len(steps))
            steps.append(int(step))
            losses.append(float(item["loss"]))

    if not losses:
        raise ValueError("日志中未找到 loss，请检查训练配置。")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    # 使用折线图展示训练稳定性
    plt.plot(steps, losses, color="#ff7f0e", linewidth=2)
    plt.title("DPO Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
