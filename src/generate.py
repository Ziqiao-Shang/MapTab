#!/usr/bin/env python3
"""Compatibility entry point modeled after MapTab-main/src/generate.py."""

from __future__ import annotations

import argparse
import os
import sys

from maptab_infer.cli import main as cli_main


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MapTab responses"
    )
    parser.add_argument(
        "--task",
        choices=("metromap", "travelmap", "all"),
        required=True,
        help="Map domain",
    )
    parser.add_argument("--subtask", required=True)
    parser.add_argument(
        "--model_path",
        default=os.getenv("MODEL_PATH", "gemini-3.5-flash"),
    )
    parser.add_argument(
        "--data_root",
        default=os.getenv("MAPTAB_DATA_ROOT", "data"),
    )
    parser.add_argument(
        "--provider",
        choices=("openlux", "vllm"),
        default=os.getenv("MAPTAB_PROVIDER", "openlux"),
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "all"),
        default="test",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output_dir",
        default="results/response_generate",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    argv = [
        "maptab-infer",
        "generate",
        "--data-root",
        args.data_root,
        "--domain",
        args.task,
        "--task",
        args.subtask,
        "--model",
        args.model_path,
        "--provider",
        args.provider,
        "--split",
        args.split,
        "--seed",
        str(args.seed),
        "--output-dir",
        args.output_dir,
    ]
    if args.limit is not None:
        argv.extend(("--limit", str(args.limit)))
    if args.overwrite:
        argv.append("--overwrite")
    sys.argv = argv
    cli_main()


if __name__ == "__main__":
    main()
