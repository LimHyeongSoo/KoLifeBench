#!/usr/bin/env python
"""Evaluate local JSONL benchmark files with HuggingFace causal LMs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    model_id: str
    default_dtype: str
    local_dir: str | None = None


MODEL_SPECS = {
    "exaone": ModelSpec("exaone", "LGAI-EXAONE/EXAONE-4.0-1.2B", "bfloat16", "models/exaone"),
    "solar": ModelSpec("solar", "upstage/SOLAR-10.7B-Instruct-v1.0", "float16", "models/solar"),
}


CHOICE_RE = re.compile(r"(?<![A-Z0-9])([ABCD])(?![A-Z0-9])")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HF model evaluation on JSONL benchmark files.")
    parser.add_argument("--model", required=True, help="Model alias or HuggingFace model ID.")
    parser.add_argument("--model-path", default=None, help="Optional local model directory to load.")
    parser.add_argument("--data-dir", default="data", help="Directory containing JSONL files.")
    parser.add_argument("--output-dir", required=True, help="Directory for predictions and summary.")
    parser.add_argument("--file", action="append", default=None, help="Specific JSONL file(s) to evaluate.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max examples per file.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default=None, choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true", help="Use EXAONE reasoning mode if supported.")
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def resolve_model(model_arg: str) -> ModelSpec:
    if model_arg in MODEL_SPECS:
        return MODEL_SPECS[model_arg]
    return ModelSpec(alias=safe_name(model_arg), model_id=model_arg, default_dtype="auto", local_dir=None)


def resolve_model_source(spec: ModelSpec, explicit_model_path: str | None) -> str:
    if explicit_model_path:
        return explicit_model_path
    if spec.local_dir and Path(spec.local_dir).exists():
        return spec.local_dir
    return spec.model_id


def safe_name(value: str) -> str:
    return value.replace("/", "__").replace(":", "_")


def dtype_from_name(name: str | None, default: str) -> Any:
    dtype_name = name or default
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def discover_files(data_dir: Path, selected: list[str] | None) -> list[Path]:
    if selected:
        files = []
        for item in selected:
            path = Path(item)
            if path.exists():
                files.append(path)
                continue
            if not path.is_absolute():
                path = data_dir / item
            files.append(path)
        return files
    return sorted(data_dir.glob("*.jsonl"))


def format_choices(choices: dict[str, str]) -> str:
    return "\n".join(f"{key}. {value}" for key, value in choices.items())


def build_prompt(row: dict[str, Any]) -> str:
    task_type = row.get("task_type")

    if task_type == "multiple_choice":
        return (
            "다음 상황을 읽고 질문에 가장 적절한 답을 고르세요.\n\n"
            f"상황:\n{row.get('context') or row.get('utterance') or ''}\n\n"
            f"질문:\n{row['question']}\n\n"
            f"보기:\n{format_choices(row['choices'])}\n\n"
            "정답은 A, B, C, D 중 하나만 출력하세요."
        )

    if task_type == "classification":
        labels = "\n".join(f"- {label}" for label in row["label_set"])
        return (
            "다음 입력을 보고 가장 적절한 label을 하나만 고르세요.\n\n"
            f"상황:\n{row.get('context') or '없음'}\n\n"
            f"발화:\n{row.get('utterance') or ''}\n\n"
            f"가능한 label:\n{labels}\n\n"
            "정답은 label 이름만 출력하세요."
        )

    if task_type in {"slot_extraction", "normalization"}:
        return (
            "다음 입력에서 필요한 정보를 추출해 JSON으로 출력하세요.\n\n"
            f"입력:\n{row.get('utterance') or row.get('context') or ''}\n\n"
            f"질문:\n{row['question']}\n\n"
            "JSON만 출력하세요."
        )

    raise ValueError(f"Unsupported task_type: {task_type}")


def apply_chat_template(tokenizer: Any, prompt: str, enable_thinking: bool) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
    }
    if enable_thinking:
        kwargs["enable_thinking"] = True
    try:
        return extract_input_ids(tokenizer.apply_chat_template(messages, **kwargs))
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return extract_input_ids(tokenizer.apply_chat_template(messages, **kwargs))
    except Exception:
        text = (
            "### User:\n"
            f"{prompt}\n\n"
            "### Assistant:\n"
        )
        return extract_input_ids(tokenizer(text, return_tensors="pt"))


def extract_input_ids(encoded: Any) -> torch.Tensor:
    if isinstance(encoded, torch.Tensor):
        return encoded
    if hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids
        return input_ids if isinstance(input_ids, torch.Tensor) else torch.tensor(input_ids)
    if isinstance(encoded, dict) and "input_ids" in encoded:
        input_ids = encoded["input_ids"]
        return input_ids if isinstance(input_ids, torch.Tensor) else torch.tensor(input_ids)
    if isinstance(encoded, list):
        return torch.tensor([encoded] if encoded and isinstance(encoded[0], int) else encoded)
    raise TypeError(f"Expected tokenizer output with input_ids, got {type(encoded)!r}")


def generate_answer(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    enable_thinking: bool,
) -> str:
    input_ids = apply_chat_template(tokenizer, prompt, enable_thinking).to(model.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][input_ids.shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def parse_prediction(row: dict[str, Any], raw_output: str) -> Any:
    task_type = row.get("task_type")
    text = raw_output.strip()

    if task_type == "multiple_choice":
        match = CHOICE_RE.search(text.upper())
        return match.group(1) if match else text[:32]

    if task_type == "classification":
        labels = row["label_set"]
        normalized = text.strip()
        for label in labels:
            if normalized == label:
                return label
        for label in labels:
            if label in normalized:
                return label
        return normalized.splitlines()[0].strip() if normalized else ""

    if task_type in {"slot_extraction", "normalization"}:
        return parse_json_from_text(text)

    return text


def parse_json_from_text(text: str) -> Any:
    text = text.strip()
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.insert(0, text[start : end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def score_prediction(row: dict[str, Any], prediction: Any) -> tuple[bool, dict[str, Any]]:
    task_type = row.get("task_type")
    if task_type in {"multiple_choice", "classification"}:
        gold = row["gold_answer"]
        return prediction == gold, {"gold": gold, "prediction": prediction}

    if task_type in {"slot_extraction", "normalization"}:
        gold = row.get("gold_output")
        if not isinstance(prediction, dict):
            total = len(gold) if isinstance(gold, dict) else 0
            return False, {
                "gold": gold,
                "prediction": prediction,
                "slot_accuracy": 0.0,
                "slots_correct": 0,
                "slots_total": total,
            }
        total = 0
        correct = 0
        for key, gold_value in gold.items():
            total += 1
            if prediction.get(key) == gold_value:
                correct += 1
        slot_accuracy = correct / total if total else 0.0
        return correct == total, {
            "gold": gold,
            "prediction": prediction,
            "slot_accuracy": slot_accuracy,
            "slots_correct": correct,
            "slots_total": total,
        }

    raise ValueError(f"Unsupported task_type: {task_type}")


def update_summary(summary: dict[str, Any], row: dict[str, Any], correct: bool, detail: dict[str, Any]) -> None:
    summary["overall"]["total"] += 1
    summary["overall"]["correct"] += int(correct)

    subset = row["subset"]
    category = f"{row['subset']}::{row['category']}"
    for key, name in [("by_subset", subset), ("by_category", category)]:
        summary[key][name]["total"] += 1
        summary[key][name]["correct"] += int(correct)

    if "slot_accuracy" in detail:
        summary["slot_scores"]["total_slots"] += detail["slots_total"]
        summary["slot_scores"]["correct_slots"] += detail["slots_correct"]


def finalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    def add_accuracy(bucket: dict[str, Any]) -> dict[str, Any]:
        total = bucket["total"]
        bucket["accuracy"] = bucket["correct"] / total if total else 0.0
        return bucket

    summary["overall"] = add_accuracy(summary["overall"])
    summary["by_subset"] = {key: add_accuracy(dict(value)) for key, value in summary["by_subset"].items()}
    summary["by_category"] = {key: add_accuracy(dict(value)) for key, value in summary["by_category"].items()}

    total_slots = summary["slot_scores"]["total_slots"]
    correct_slots = summary["slot_scores"]["correct_slots"]
    summary["slot_scores"]["slot_accuracy"] = correct_slots / total_slots if total_slots else None
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    overall = summary["overall"]
    print("\n=== Evaluation Summary ===")
    print(
        f"Overall: {overall['correct']}/{overall['total']} "
        f"({overall['accuracy']:.3f})"
    )

    print("\nBy subset:")
    for subset, item in sorted(summary["by_subset"].items()):
        print(
            f"- {subset}: {item['correct']}/{item['total']} "
            f"({item['accuracy']:.3f})"
        )

    slot_accuracy = summary["slot_scores"].get("slot_accuracy")
    if slot_accuracy is not None:
        correct_slots = summary["slot_scores"]["correct_slots"]
        total_slots = summary["slot_scores"]["total_slots"]
        print(f"\nJSON slot accuracy: {correct_slots}/{total_slots} ({slot_accuracy:.3f})")


def main() -> None:
    args = parse_args()
    spec = resolve_model(args.model)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = dtype_from_name(args.torch_dtype, spec.default_dtype)
    model_source = resolve_model_source(spec, args.model_path)
    print(f"Loading model: {model_source}")
    if model_source != spec.model_id:
        print(f"Original HuggingFace model ID: {spec.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=args.trust_remote_code)
    model_kwargs = {
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    if dtype != "auto":
        model_kwargs["dtype"] = dtype
    else:
        model_kwargs["dtype"] = "auto"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    except TypeError:
        dtype_value = model_kwargs.pop("dtype")
        model_kwargs["torch_dtype"] = dtype_value
        model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    model.eval()

    files = discover_files(data_dir, args.file)
    if not files:
        raise FileNotFoundError(f"No JSONL files found in {data_dir}")
    file_rows = [(path, load_jsonl(path, args.limit)) for path in files]
    total_examples = sum(len(rows) for _, rows in file_rows)
    print(f"Found {len(files)} data file(s), {total_examples} example(s).")

    summary: dict[str, Any] = {
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
                prompt = build_prompt(row)
                raw_output = generate_answer(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    enable_thinking=args.enable_thinking,
                )
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
