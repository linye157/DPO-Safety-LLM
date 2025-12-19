import argparse
import logging
import os
from typing import Optional

import torch
from datasets import Dataset
from unsloth import FastLanguageModel, PatchDPOTrainer

from src.data_loader import load_and_process_data
from src.visualizer import plot_data_distribution, plot_training_loss

# 必须在导入 DPOTrainer 前完成补丁注入
PatchDPOTrainer()

from trl import DPOConfig, DPOTrainer


logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SafetyAlign DPO Trainer")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=60,
        help="训练最大步数。",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="学习率。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/dpo_lora",
        help="LoRA 适配器保存目录。",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="每张卡的 batch size。",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="梯度累积步数。",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=512,
        help="Prompt 最大长度。",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Prompt + Response 最大长度。",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=5,
        help="日志打印频率。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子。",
    )
    parser.add_argument(
        "--use_reference_model",
        action="store_true",
        help="是否额外加载参考模型以降低 DPO 偏置。",
    )
    return parser


def _load_model(
    model_name: str,
    max_length: int,
) -> tuple:
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_length,
            dtype=None,
            load_in_4bit=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"模型加载失败：{model_name}，请检查网络或模型名称。"
        ) from exc
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _apply_lora(model):
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = _build_arg_parser().parse_args()

    if not torch.cuda.is_available():
        logger.warning("未检测到 GPU，训练可能无法正常运行。")

    # 加载并处理数据，同时生成长度分布图
    df = load_and_process_data()
    plot_data_distribution(df, save_path="plots/data_distribution.png")

    # 转换为 HF Dataset，供 DPOTrainer 使用
    dataset = Dataset.from_pandas(
        df[["prompt", "chosen", "rejected"]],
        preserve_index=False,
    )

    model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    # 加载基础模型与分词器
    model, tokenizer = _load_model(model_name, args.max_length)
    ref_model: Optional[object] = None
    if args.use_reference_model:
        # 可选加载参考模型，提升 DPO 稳定性
        ref_model, _ = _load_model(model_name, args.max_length)
    # 应用 LoRA 以降低显存占用
    model = _apply_lora(model)

    # 根据硬件自动选择 fp16 或 bf16
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16

    # 配置 DPO 训练超参数
    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        logging_dir="logs",
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
        beta=0.1,
        fp16=fp16,
        bf16=bf16,
        seed=args.seed,
    )

    # 初始化 DPOTrainer 并开始训练
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # 保存 LoRA 适配器与分词器
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 绘制训练损失曲线
    plot_training_loss(
        trainer.state.log_history,
        save_path="plots/training_loss.png",
    )


if __name__ == "__main__":
    main()
