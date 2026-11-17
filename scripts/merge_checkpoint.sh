#!/usr/bin/env bash
set -euo pipefail
CHECKPOINT="${1:?Usage: bash scripts/merge_checkpoint.sh CHECKPOINT [OUTPUT]}"
OUTPUT="${2:-$CHECKPOINT/huggingface}"
PYTHON="${PYTHON:-python3}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
"$PYTHON" -m verl.model_merger merge --backend fsdp \
    --local_dir "$CHECKPOINT/actor" --target_dir "$OUTPUT"
cp "$PROJECT_ROOT/chat_templates/perception_chat_template_qwen35.jinja" "$OUTPUT/chat_template.jinja"
