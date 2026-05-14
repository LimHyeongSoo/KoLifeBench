#!/usr/bin/env python
"""Evaluate local JSONL benchmark files with vLLM."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

from run_hf_eval import (
    build_prompt,
    discover_files,
    finalize_summary,
    load_jsonl,
    parse_prediction,
    resolve_model,
    resolve_model_source,
    score_prediction,
    update_summary,
)
from run_hf_eval import print_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vLLM evaluation on JSONL benchmark files.")
    parser.add_argument("--model", required=True, help="Model alias or HuggingFace model ID.")
    parser.add_argument("--model-path", default=None, help="Optional local model directory to load.")
    parser.add_argument("--data-dir", default="data", help="Directory containing JSONL files.")
    parser.add_argument("--output-dir", required=True, help="Directory for predictions and summary.")
    parser.add_argument("--file", action="append", default=None, help="Specific JSONL file(s) to evaluate.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max examples per file.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def render_chat_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return "### User:\n" f"{prompt}\n\n" "### Assistant:\n"


def main() -> None:
    args = parse_args()
    spec = resolve_model(args.model)
    model_source = resolve_model_source(spec, args.model_path)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading vLLM model: {model_source}")
    if model_source != spec.model_id:
        print(f"Original HuggingFace model ID: {spec.model_id}")

    llm_kwargs = {
        "model": model_source,
        "tokenizer": model_source,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )

    files = discover_files(data_dir, args.file)
    if not files:
        raise FileNotFoundError(f"No JSONL files found in {data_dir}")
    file_rows = [(path, load_jsonl(path, args.limit)) for path in files]
    total_examples = sum(len(rows) for _, rows in file_rows)
    print(f"Found {len(files)} data file(s), {total_examples} example(s).")

    summary: dict[str, Any] = {
        "backend": "vllm",
        "model_alias": spec.alias,
        "model_id": spec.model_id,
        "model_source": model_source,
        "overall": Counter(total=0, correct=0),
        "by_subset": defaultdict(lambda: Counter(total=0, correct=0)),
        "by_category": defaultdict(lambda: Counter(total=0, correct=0)),
        "slot_scores": Counter(total_slots=0, correct_slots=0),
        "files": [str(path) for path in files],
    }

    predictions_path = output_dir / "predictions.jsonl"
    count = 0
    with predictions_path.open("w", encoding="utf-8") as out:
        progress = tqdm(total=total_examples, desc="Evaluating", unit="ex")
        for path, rows in file_rows:
            progress.write(f"Evaluating {path} ({len(rows)} rows)")
            for row in rows:
                count += 1
                prompt = render_chat_prompt(tokenizer, build_prompt(row))
                outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
                raw_output = outputs[0].outputs[0].text.strip()
                prediction = parse_prediction(row, raw_output)
                correct, detail = score_prediction(row, prediction)
                update_summary(summary, row, correct, detail)

                record = {
                    "id": row["id"],
                    "file": str(path),
                    "benchmark": row["benchmark"],
                    "subset": row["subset"],
                    "category": row["category"],
                    "task_type": row["task_type"],
                    "gold_answer": row.get("gold_answer"),
                    "gold_output": row.get("gold_output"),
                    "raw_output": raw_output,
                    "prediction": prediction,
                    "correct": correct,
                    "score_detail": detail,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

                overall = summary["overall"]
                acc = overall["correct"] / overall["total"] if overall["total"] else 0.0
                progress.set_postfix(acc=f"{acc:.3f}")
                progress.update(1)
        progress.close()

    final = finalize_summary(summary)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions: {predictions_path}")
    print(f"Wrote summary: {summary_path}")
    print_summary(final)


if __name__ == "__main__":
    main()
