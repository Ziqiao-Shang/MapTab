#!/usr/bin/env python3
"""Compatibility QA evaluator matching MapTab-main's entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from maptab_infer.evaluate import evaluate_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file")
    args = parser.parse_args()
    input_path = Path(args.input_file)
    output = args.output_file
    if output is None:
        folder = input_path.parent.name
        output = (
            Path("results_evaluate")
            / f"evaluate_qa_{folder}"
            / f"evaluate_qa_{input_path.name}"
        )
    print(
        json.dumps(
            evaluate_file(input_path, "qa", output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
