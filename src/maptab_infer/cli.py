from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from .data import (
    build_content,
    load_records,
    prompt_text,
    referenced_assets,
)
from .evaluate import evaluate_file
from .providers import OpenAICompatibleProvider, VLLMProvider
from .runner import run_generation
from .tasks import TaskSpec, get_task, list_tasks, task_counts

_DOMAINS = ("metromap", "travelmap")


def _selection(name: str) -> list[TaskSpec]:
    if name == "all":
        return list_tasks()
    if name == "all-qa":
        return list_tasks("qa")
    if name == "all-planning":
        return list_tasks("planning")
    if name == "all-probes":
        return list_tasks("probe")
    if name == "canonical-qa":
        return list_tasks("qa", include_ablations=False)
    if name == "canonical-planning":
        return list_tasks("planning", include_ablations=False)
    return [get_task(name)]


def _domains(value: str) -> Iterable[str]:
    return _DOMAINS if value == "all" else (value,)


def cmd_list(args: argparse.Namespace) -> None:
    tasks = list_tasks(args.family, not args.canonical_only)
    if args.json:
        print(
            json.dumps(
                {
                    "counts": task_counts(),
                    "tasks": [
                        {
                            "name": task.name,
                            "family": task.family,
                            "variant": task.variant,
                            "modalities": list(task.modalities),
                            "output_protocol": task.output_protocol,
                            "description": task.description,
                        }
                        for task in tasks
                    ],
                },
                indent=2,
            )
        )
        return
    for task in tasks:
        modalities = ",".join(task.modalities)
        print(
            f"{task.family:8} {task.variant:10} "
            f"{task.name:62} {modalities}"
        )


def cmd_download(args: argparse.Namespace) -> None:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.data_root,
    )
    print(path)


def cmd_validate(args: argparse.Namespace) -> None:
    errors: list[str] = []
    checked_files: set[Path] = set()
    checked_assets: set[Path] = set()
    row_count = 0
    tasks = _selection(args.task)
    if args.canonical_only:
        tasks = [
            task
            for task in tasks
            if task.variant in {"canonical", "additional"}
        ]

    for domain in _domains(args.domain):
        for spec in tasks:
            split = "test" if spec.family in {"qa", "probe"} else args.split
            try:
                prompt_text(domain, spec)
                rows, path = load_records(
                    args.data_root,
                    domain,
                    spec,
                    split,
                )
                checked_files.add(path)
                row_count += len(rows)
                for row in rows:
                    checked_assets.update(
                        referenced_assets(args.data_root, spec, row)
                    )
            except Exception as exc:
                errors.append(f"{domain}/{spec.name}: {exc}")

    print(
        json.dumps(
            {
                "tasks": len(tasks) * len(tuple(_domains(args.domain))),
                "data_files": len(checked_files),
                "rows_checked": row_count,
                "assets": len(checked_assets),
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if errors:
        raise SystemExit(1)


def cmd_inspect(args: argparse.Namespace) -> None:
    spec = get_task(args.task)
    split = "test" if spec.family in {"qa", "probe"} else args.split
    rows, source = load_records(
        args.data_root,
        args.domain,
        spec,
        split,
    )
    if args.index < 0 or args.index >= len(rows):
        raise SystemExit(
            f"index {args.index} outside [0, {len(rows) - 1}]"
        )
    content = build_content(
        args.data_root,
        args.domain,
        spec,
        rows[args.index],
    )
    print(
        json.dumps(
            {
                "task": spec.name,
                "family": spec.family,
                "variant": spec.variant,
                "source": str(source),
                "source_index": args.index,
                "content": content,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _provider(args: argparse.Namespace):
    if args.provider == "vllm":
        return VLLMProvider(
            args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
            max_pixels=args.max_pixels,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=args.trust_remote_code,
        )

    key = os.getenv(args.api_key_env)
    if not key:
        raise SystemExit(
            f"environment variable {args.api_key_env} is required"
        )
    return OpenAICompatibleProvider(
        args.model,
        key,
        args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        max_pixels=args.max_pixels,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
        timeout=args.timeout,
    )


def cmd_generate(args: argparse.Namespace) -> None:
    provider = _provider(args)
    for domain in _domains(args.domain):
        for spec in _selection(args.task):
            output = run_generation(
                provider=provider,
                model=args.model,
                data_root=args.data_root,
                output_dir=args.output_dir,
                domain=domain,
                spec=spec,
                split=args.split,
                offset=args.offset,
                limit=args.limit,
                overwrite=args.overwrite,
                continue_on_error=args.continue_on_error,
                retry_errors=args.retry_errors,
            )
            print(output)


def _infer_family(path: Path) -> str:
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        family = row.get("benchmark_family")
        if family in {"qa", "planning"}:
            return family
    raise ValueError(
        f"cannot infer scored family from {path}; pass --family"
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    path = Path(args.input)
    family = (
        _infer_family(path)
        if args.family == "auto"
        else args.family
    )
    metrics = evaluate_file(path, family, args.output)
    print(json.dumps(metrics, indent=2))


def cmd_evaluate_dir(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(args.pattern))
    files = [
        path
        for path in files
        if ".evaluated." not in path.name
        and not path.name.endswith(".summary.json")
    ]
    if not files:
        raise SystemExit(f"no files match {args.pattern!r} in {input_dir}")

    summaries: dict[str, dict] = {}
    for path in files:
        actual_family = _infer_family(path)
        if args.family != "auto" and actual_family != args.family:
            continue
        family = actual_family
        output_dir = (
            Path(args.output_dir)
            / f"evaluate_{family}_{input_dir.name}"
        )
        output = output_dir / f"evaluate_{family}_{path.name}"
        summaries[path.name] = evaluate_file(path, family, output)
    if not summaries:
        raise SystemExit(
            f"no {args.family} result files match {args.pattern!r} "
            f"in {input_dir}"
        )
    print(json.dumps(summaries, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="maptab-infer",
        description="Complete MapTab QA and route-planning inference",
    )
    commands = root.add_subparsers(dest="command", required=True)

    command = commands.add_parser("list", help="list registered tasks")
    command.add_argument(
        "--family",
        choices=("qa", "planning", "probe"),
    )
    command.add_argument("--canonical-only", action="store_true")
    command.add_argument("--json", action="store_true")
    command.set_defaults(func=cmd_list)

    command = commands.add_parser(
        "download",
        help="download a dataset snapshot from Hugging Face",
    )
    command.add_argument(
        "--repo-id",
        default="szq-nju/MapTab",
    )
    command.add_argument("--revision", default="main")
    command.add_argument("--data-root", default="data")
    command.set_defaults(func=cmd_download)

    command = commands.add_parser(
        "validate",
        help="validate data, prompts, and referenced assets",
    )
    command.add_argument("--data-root", required=True)
    command.add_argument(
        "--domain",
        choices=("all",) + _DOMAINS,
        default="all",
    )
    command.add_argument(
        "--task",
        default="all",
        help="task name, all, all-qa, all-planning, or all-probes",
    )
    command.add_argument(
        "--split",
        choices=("train", "test", "all"),
        default="test",
    )
    command.add_argument("--canonical-only", action="store_true")
    command.set_defaults(func=cmd_validate)

    command = commands.add_parser(
        "inspect",
        help="render one model input without running inference",
    )
    command.add_argument("--data-root", required=True)
    command.add_argument(
        "--domain",
        choices=_DOMAINS,
        required=True,
    )
    command.add_argument("--task", required=True)
    command.add_argument(
        "--split",
        choices=("train", "test", "all"),
        default="test",
    )
    command.add_argument("--index", type=int, default=0)
    command.set_defaults(func=cmd_inspect)

    command = commands.add_parser(
        "generate",
        help="run one task or a complete task family",
    )
    command.add_argument("--data-root", required=True)
    command.add_argument(
        "--domain",
        choices=("all",) + _DOMAINS,
        default="all",
    )
    command.add_argument(
        "--task",
        required=True,
        help=(
            "task name, all, all-qa, all-planning, all-probes, "
            "canonical-qa, or canonical-planning"
        ),
    )
    command.add_argument(
        "--split",
        choices=("train", "test", "all"),
        default="test",
    )
    command.add_argument(
        "--provider",
        choices=("openai", "vllm"),
        default="openai",
    )
    command.add_argument("--model", required=True)
    command.add_argument("--base-url")
    command.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
    )
    command.add_argument("--temperature", type=float, default=0.0)
    command.add_argument("--max-tokens", type=int, default=2048)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--max-pixels", type=int, default=10_000_000)
    command.add_argument("--max-retries", type=int, default=10)
    command.add_argument("--retry-backoff", type=float, default=1.0)
    command.add_argument("--timeout", type=float)
    command.add_argument("--tensor-parallel-size", type=int, default=1)
    command.add_argument("--max-model-len", type=int, default=128_000)
    command.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
    )
    command.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    command.add_argument("--offset", type=int, default=0)
    command.add_argument("--limit", type=int)
    command.add_argument(
        "--output-dir",
        default="results/response_generate",
    )
    command.add_argument("--overwrite", action="store_true")
    command.add_argument("--retry-errors", action="store_true")
    command.add_argument("--continue-on-error", action="store_true")
    command.set_defaults(func=cmd_generate)

    command = commands.add_parser(
        "evaluate",
        help="evaluate one generated JSON file",
    )
    command.add_argument(
        "--family",
        choices=("auto", "qa", "planning"),
        default="auto",
    )
    command.add_argument("--input", required=True)
    command.add_argument("--output")
    command.set_defaults(func=cmd_evaluate)

    command = commands.add_parser(
        "evaluate-dir",
        help="evaluate every generated result in a directory",
    )
    command.add_argument("--input-dir", required=True)
    command.add_argument(
        "--output-dir",
        default="results_evaluate",
    )
    command.add_argument(
        "--family",
        choices=("auto", "qa", "planning"),
        default="auto",
    )
    command.add_argument(
        "--pattern",
        default="*_results.json",
    )
    command.set_defaults(func=cmd_evaluate_dir)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
