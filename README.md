# SafetyAlign

面向安全对齐的工程化 DPO 微调项目，基于 Unsloth + TRL 训练
Qwen2.5-7B-Instruct 的安全偏好模型，并提供数据分析、训练可视化与红队评测脚本。

## 项目特性

- DPO 微调：使用 `PatchDPOTrainer` 与 LoRA 适配器。
- 数据可视化：Prompt/Response 长度分布、训练 Loss 曲线。
- 红队评测：基础安全提示集合，输出 JSON 结果。
- 工程化结构：脚本化运行、结果持久化、日志与图表归档。

## 快速开始

### 1. 安装依赖

建议使用 `uv` 管理虚拟环境与依赖：

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```
或者直接(文件夹下有有uv.lock文件和pyproject.toml文件)
```bash
uv sync
``` 


如需 GPU 训练，请根据自身环境安装对应的 PyTorch 版本。

### 2. 运行训练

```bash
bash scripts/run_train.sh
```

训练结束后，LoRA 适配器会保存到 `output/dpo_lora`。

### 3. 运行评测

```bash
bash scripts/run_eval.sh
```

评测结果会输出到 `output/evaluation_results.json`。

## 结果展示

请查看 `plots/` 目录：

- `plots/data_distribution.png`：Prompt/Response 长度分布。
- `plots/training_loss.png`：DPO Loss 曲线。

## 目录结构

```
.
├── data/                  # 数据缓存
├── src/                   # 训练与可视化代码
├── scripts/               # 运行脚本
├── output/                # 模型与评测输出
├── plots/                 # 训练与数据图表
├── logs/                  # 运行日志
├── tests/                 # 红队测试脚本
└── requirements.txt
```

## 备注

- 默认数据集：`PKU-Alignment/PKU-SafeRLHF-10K`（`train[:2000]`）。
- 若需要更严格的 DPO 参考模型，可在训练时加 `--use_reference_model`。
- Triton 运行需要系统 C 编译器（gcc/clang），未安装会导致训练中断。

## License

MIT
