#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=. python -m src.trainer \
  --max_steps 60 \
  --learning_rate 5e-6 \
  --output_dir output/dpo_lora
