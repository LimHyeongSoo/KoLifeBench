#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-eval/results/solar}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-0}"
MODEL_PATH="${MODEL_PATH:-models/solar}"
BACKEND="${BACKEND:-vllm}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" && "$LIMIT" != "0" && "$LIMIT" != "all" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

if [[ "$DOWNLOAD_MODEL" == "1" ]]; then
  python eval/download_models.py --model solar
fi

MODEL_PATH_ARGS=()
if [[ -d "$MODEL_PATH" ]]; then
  MODEL_PATH_ARGS=(--model-path "$MODEL_PATH")
fi

echo "[koreaEE] model=solar backend=$BACKEND gpu=$CUDA_VISIBLE_DEVICES data=$DATA_DIR output=$OUTPUT_DIR"

if [[ "$BACKEND" == "vllm" ]]; then
  python eval/run_vllm_eval.py \
    --model solar \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    "${MODEL_PATH_ARGS[@]}" \
    "${LIMIT_ARGS[@]}" \
    "$@"
else
  python eval/run_hf_eval.py \
    --model solar \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    "${MODEL_PATH_ARGS[@]}" \
    "${LIMIT_ARGS[@]}" \
    "$@"
fi
