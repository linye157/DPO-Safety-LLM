#!/usr/bin/env bash
set -euo pipefail

python src/trainer.py \
  --max_steps 60 \
  --learning_rate 5e-6 \
  --output_dir output/dpo_lora
