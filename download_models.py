#!/usr/bin/env python
"""Download supported HuggingFace models into the local HF cache."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    model_id: str


MODEL_SPECS = {
    "exaone": ModelSpec("exaone", "LGAI-EXAONE/EXAONE-4.0-1.2B"),
    "solar": ModelSpec("solar", "upstage/SOLAR-10.7B-Instruct-v1.0"),
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
        "--local-dir",
        default=None,
        help="Optional local directory. Use this only when downloading one model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aliases = list(MODEL_SPECS) if args.model == "all" else [args.model]

    if args.local_dir and len(aliases) > 1:
        raise ValueError("--local-dir can only be used when downloading one model.")

    for alias in aliases:
        spec = MODEL_SPECS[alias]
        print(f"Downloading {alias}: {spec.model_id}")
        path = snapshot_download(
            repo_id=spec.model_id,
            revision=args.revision,
            local_dir=args.local_dir,
        )
        print(f"Downloaded {alias} to: {path}")


if __name__ == "__main__":
    main()

