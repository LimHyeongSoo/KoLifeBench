# KoreaEE Benchmark Evaluation

This project contains Korean LLM benchmark datasets and HuggingFace evaluation scripts.

## 1. Conda Environment

Create and activate the conda environment:

```bash
conda create -n koreaEE python=3.11 -y
conda activate koreaEE
python -m pip install -U pip
```

Install all dependencies from the project root:

```bash
pip install -r requirements.txt
```

The unified environment uses `transformers>=4.54.0` because `LGAI-EXAONE/EXAONE-4.0-1.2B` requires a recent Transformers version. The same environment is used for SOLAR as well.

## 2. Models

Supported model aliases:

| Alias | HuggingFace model ID |
| --- | --- |
| `exaone` | `LGAI-EXAONE/EXAONE-4.0-1.2B` |
| `solar` | `upstage/SOLAR-10.7B-Instruct-v1.0` |

If HuggingFace authentication is needed:

```bash
huggingface-cli login
```

Download models manually. By default, models are saved under `models/<alias>/`.

```bash
python eval/download_models.py --model exaone
python eval/download_models.py --model solar
```

Or download both:

```bash
python eval/download_models.py --model all
```

Default local paths:

| Alias | Local path |
| --- | --- |
| `exaone` | `models/exaone/` |
| `solar` | `models/solar/` |

If these directories exist, the evaluation scripts load the local model from `models/<alias>/` instead of downloading from HuggingFace at runtime.

## 3. Run Evaluation

Run EXAONE:

```bash
scripts/exaone.sh
```

Run SOLAR:

```bash
scripts/solar.sh
```

Both wrappers use GPU 3 and 4 by default:

```bash
CUDA_VISIBLE_DEVICES=3,4 scripts/exaone.sh
CUDA_VISIBLE_DEVICES=3,4 scripts/solar.sh
```

You can override this if needed:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/exaone.sh
CUDA_VISIBLE_DEVICES=0,1 scripts/solar.sh
```

SOLAR uses vLLM by default because the model is much larger than EXAONE. The default tensor parallel size is 2, matching GPUs 3 and 4.

The SOLAR wrapper defaults to the full dataset. It still uses conservative vLLM memory settings:

| Setting | Default |
| --- | --- |
| `LIMIT` | full dataset |
| `GPU_MEMORY_UTILIZATION` | `0.65` |
| `MAX_MODEL_LEN` | `2048` |

```bash
scripts/solar.sh
```

To run a small smoke test with SOLAR:

```bash
LIMIT=3 bash ./scripts/solar.sh
```

To change vLLM tensor parallelism:

```bash
TENSOR_PARALLEL_SIZE=1 CUDA_VISIBLE_DEVICES=3 scripts/solar.sh
```

The SOLAR wrapper uses `GPU_MEMORY_UTILIZATION=0.65` by default. If vLLM still reports that free memory is lower than requested memory, lower this value:

```bash
GPU_MEMORY_UTILIZATION=0.55 bash ./scripts/solar.sh
```

To force the old HuggingFace backend for SOLAR:

```bash
BACKEND=hf scripts/solar.sh
```

The wrappers use these local paths by default when present:

```bash
MODEL_PATH=models/exaone scripts/exaone.sh
MODEL_PATH=models/solar scripts/solar.sh
```

Download the model before evaluation:

```bash
DOWNLOAD_MODEL=1 scripts/exaone.sh
DOWNLOAD_MODEL=1 scripts/solar.sh
```

Quick smoke test:

```bash
LIMIT=3 scripts/exaone.sh
LIMIT=3 scripts/solar.sh
```

Evaluate only one dataset file:

```bash
scripts/exaone.sh --file pragmatics_sentence_ending.jsonl
scripts/solar.sh --file kolife_admin.jsonl
```

## 4. Outputs

Results are saved under:

```text
eval/results/<model_alias>/
```

Each run writes:

| File | Description |
| --- | --- |
| `predictions.jsonl` | Raw model output, parsed prediction, and correctness per example |
| `summary.json` | Overall, subset-level, and category-level accuracy |

During evaluation, the terminal shows a progress bar and running accuracy. At the end, it prints the overall score, subset-level scores, and JSON slot accuracy when applicable.

## 5. Hardware Notes

EXAONE 1.2B is relatively small. SOLAR 10.7B requires substantially more GPU memory.

SOLAR uses vLLM by default. If memory is tight, reduce `MAX_NEW_TOKENS`, lower `GPU_MEMORY_UTILIZATION`, lower `MAX_MODEL_LEN`, or keep the default limited run first:

```bash
MAX_NEW_TOKENS=32 GPU_MEMORY_UTILIZATION=0.55 MAX_MODEL_LEN=1024 scripts/solar.sh
```
