#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4}"

DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-eval/results/exaone}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-0}"
MODEL_PATH="${MODEL_PATH:-models/exaone}"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

if [[ "$DOWNLOAD_MODEL" == "1" ]]; then
  python eval/download_models.py --model exaone
fi

MODEL_PATH_ARGS=()
if [[ -d "$MODEL_PATH" ]]; then
  MODEL_PATH_ARGS=(--model-path "$MODEL_PATH")
fi

echo "[koreaEE] model=exaone backend=hf gpu=$CUDA_VISIBLE_DEVICES data=$DATA_DIR output=$OUTPUT_DIR"

python eval/run_hf_eval.py \
  --model exaone \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  "${MODEL_PATH_ARGS[@]}" \
  "${LIMIT_ARGS[@]}" \
  "$@"
