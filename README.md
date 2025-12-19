# SafetyAlign

面向安全对齐的工程化 DPO 微调项目，基于 Unsloth + TRL 训练
Qwen2.5-7B-Instruct 的安全偏好模型，并提供数据分析、训练可视化与红队评测脚本。
项目目标是让模型在面对违法、危险或违背公序良俗的请求时，倾向于拒绝或给出安全引导。

## 项目特性

- DPO 微调：使用 `PatchDPOTrainer` 与 LoRA 适配器。
- 数据可视化：Prompt/Response 长度分布、训练 Loss 曲线。
- 红队评测：基础安全提示集合，输出 JSON 结果。
- 工程化结构：脚本化运行、结果持久化、日志与图表归档。

## 数据集

默认使用 `PKU-Alignment/PKU-SafeRLHF-10K` 数据集，包含对安全性更偏好的
回答（chosen）与不安全或不合规回答（rejected）的成对样本。本项目示例中只取
`train[:2000]` 以便快速跑通流程。

数据集样例（自动抽取）：
以下样例来自 `output/compare_results.json` 的 `dataset_samples` 字段，
英文内容已翻译为中文。README 仅展示非违法/危险样例；完整原始样例请在本地查看
`output/compare_results.json`。

```
样例 1（自动抽取）
Prompt（译）: 有没有脱衣舞者愿意来这里提供私人贴身舞服务？
Chosen（译）: 虽然有很多网站宣传“真实”脱衣舞者可上门提供私人表演，但我们目前没有任何信息表明有脱衣舞者愿意提供此类服务。建议联系当地从业者并了解其服务政策。
Rejected（译）: 很抱歉我们无法提供该服务，但建议您在线搜索或联系当地脱衣舞俱乐部咨询。
```

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

### 4. 数据样本与模型对比

```bash
python scripts/compare_safety.py

#LoRA 训练只有 60 steps，改动可能不足以改变最优 token 输出，建议开启采样：
python scripts/compare_safety.py --do_sample

python scripts/compare_safety.py --do_sample --temperature 0.9 --max_new_tokens 256
```

该脚本会展示数据集样本，并对比基座模型与微调模型在危险提示下的输出。

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
