from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .data import build_content, load_records
from .parsing import normalize_generation
from .providers import Provider
from .tasks import TaskSpec


def model_slug(model: str) -> str:
    value = model.rstrip("/").split("/")[-1]
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value or "model"


def result_path(
    output_dir: str | Path,
    domain: str,
    spec: TaskSpec,
    model: str,
) -> Path:
    return Path(output_dir) / (
        f"{domain}_{spec.name}_{model_slug(model)}_results.json"
    )


def _read_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"existing output is not a JSON list: {path}")
    return {
        str(row["_id"]): row
        for row in rows
        if isinstance(row, dict) and "_id" in row
    }


def _atomic_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def run_generation(
    *,
    provider: Provider,
    model: str,
    data_root: str | Path,
    output_dir: str | Path,
    domain: str,
    spec: TaskSpec,
    split: str,
    offset: int = 0,
    limit: int | None = None,
    overwrite: bool = False,
    continue_on_error: bool = False,
    retry_errors: bool = False,
) -> Path:
    effective_split = "test" if spec.family in {"qa", "probe"} else split
    rows, source = load_records(data_root, domain, spec, effective_split)
    indexed = list(enumerate(rows))
    indexed = indexed[offset:]
    if limit is not None:
        indexed = indexed[:limit]

    output = result_path(output_dir, domain, spec, model)
    existing = {} if overwrite else _read_existing(output)

    for source_index, row in tqdm(
        indexed,
        desc=f"{domain}/{spec.name}",
    ):
        item_id = (
            f"{domain}:{spec.family}:{spec.name}:"
            f"{effective_split}:{source_index:06d}"
        )
        prior = existing.get(item_id)
        if prior is not None and (
            not retry_errors or not prior.get("error")
        ):
            continue

        generated: dict[str, Any]
        reasoning: str | None = None
        error: str | None = None
        try:
            content = build_content(data_root, domain, spec, row)
            result = provider.generate(content)
            generated = normalize_generation(spec, result.text)
            reasoning = result.reasoning
        except Exception as exc:
            if not continue_on_error:
                raise
            error = f"{type(exc).__name__}: {exc}"
            generated = {"raw_response": "", "response": ""}

        item = dict(row)
        item.update(
            {
                "_id": item_id,
                "_source_index": source_index,
                "benchmark_domain": domain,
                "benchmark_task": spec.name,
                "benchmark_family": spec.family,
                "benchmark_variant": spec.variant,
                "benchmark_split": effective_split,
                "source_file": str(source),
                "model": model,
                "reasoning_content": reasoning,
                "error": error,
            }
        )
        item.update(generated)
        existing[item_id] = item
        ordered = sorted(
            existing.values(),
            key=lambda value: int(value.get("_source_index", 0)),
        )
        _atomic_write(output, ordered)

    if not output.exists():
        _atomic_write(
            output,
            sorted(
                existing.values(),
                key=lambda value: int(value.get("_source_index", 0)),
            ),
        )
    return output
