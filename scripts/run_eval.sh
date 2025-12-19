#!/usr/bin/env bash
set -euo pipefail

python tests/test_safety.py \
  --adapter_path output/dpo_lora
