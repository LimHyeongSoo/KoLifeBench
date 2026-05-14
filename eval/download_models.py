#!/usr/bin/env python
"""Download supported HuggingFace models into ./models/<alias>."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    model_id: str
    local_dir: Path


MODEL_SPECS = {
    "exaone": ModelSpec("exaone", "LGAI-EXAONE/EXAONE-4.0-1.2B", Path("models/exaone")),
    "solar": ModelSpec("solar", "upstage/SOLAR-10.7B-Instruct-v1.0", Path("models/solar")),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download benchmark evaluation models.")
    parser.add_argument(
        "--model",
        choices=["exaone", "solar", "all"],
        required=True,
        help="Model alias to download.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional HuggingFace revision. Applied to every selected model.",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Base directory for local model snapshots.",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Optional exact local directory. Use only when downloading one model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aliases = list(MODEL_SPECS) if args.model == "all" else [args.model]

    if args.local_dir and len(aliases) > 1:
        raise ValueError("--local-dir can only be used when downloading one model.")

    models_dir = Path(args.models_dir)
    for alias in aliases:
        spec = MODEL_SPECS[alias]
        local_dir = Path(args.local_dir) if args.local_dir else models_dir / alias
        local_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {alias}: {spec.model_id}")
        print(f"Target directory: {local_dir}")
        path = snapshot_download(
            repo_id=spec.model_id,
            revision=args.revision,
            local_dir=str(local_dir),
        )
        print(f"Downloaded {alias} to: {path}")


if __name__ == "__main__":
    main()

