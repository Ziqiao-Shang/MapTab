from __future__ import annotations

import csv
import io
import json
from importlib import resources
from pathlib import Path
from typing import Any, Iterable

from .tasks import TaskSpec


# These policies reproduce the column masking performed by MapTab-main before a
# table is placed in the model context. The two domains intentionally differ.
_COLUMN_POLICIES: dict[str, dict[str, tuple[tuple[str, ...], tuple[str, ...]]]] = {
    "metromap": {
        "shortest_path_only_tab": (
            ("Time", "Price", "Comfort Level", "Reliability"),
            (),
        ),
        "shortest_path_map_and_tab_no_constraint": (
            ("Time", "Price", "Comfort Level", "Reliability"),
            (),
        ),
        "shortest_path_map_and_tab_with_constraint_1": (
            ("Price", "Comfort Level", "Reliability"),
            ("Price", "Comfort Level", "Reliability"),
        ),
        "shortest_path_map_and_tab_with_constraint_2": (
            ("Time", "Comfort Level", "Reliability"),
            ("Time", "Comfort Level", "Reliability", "Transfer Time"),
        ),
        "shortest_path_map_and_tab_with_constraint_3": (
            ("Time", "Price", "Reliability"),
            ("Time", "Price", "Reliability", "Transfer Time"),
        ),
        "shortest_path_map_and_tab_with_constraint_4": (
            ("Time", "Price", "Comfort Level"),
            ("Time", "Price", "Comfort Level", "Transfer Time"),
        ),
        "shortest_path_map_and_tab_with_constraint_1_2_3_4": ((), ()),
        "shortest_path_map_and_tab_with_constraint_1_2_4": (
            ("Comfort Level",),
            ("Comfort Level",),
        ),
        "shortest_path_map_and_tab_with_constraint_1_3_4": (
            ("Price",),
            ("Price",),
        ),
        "shortest_path_map_and_tab_with_constraint_2_3_4": (
            ("Time",),
            ("Time", "Transfer Time"),
        ),
        "only_vertex2": ((), ("Line",)),
        "10_qa_pic_and_tab_global": ((), ("Line",)),
        "11_qa_pic_and_tab_part": ((), ("Line",)),
        "12_qa_pic_and_tab_spatial_judge": ((), ("Line",)),
    },
    "travelmap": {
        "shortest_path_only_tab": (
            ("Time", "Price", "Comfort Level", "Reliability"),
            (),
        ),
        "shortest_path_map_and_tab_no_constraint": (
            ("Time", "Price", "Comfort Level", "Reliability"),
            (),
        ),
        "shortest_path_map_and_tab_with_constraint_1": (
            ("Price", "Comfort Level", "Reliability"),
            ("Price", "Comfort Level", "Reliability"),
        ),
        "shortest_path_map_and_tab_with_constraint_2": (
            ("Comfort Level", "Reliability"),
            ("Comfort Level", "Reliability"),
        ),
        "shortest_path_map_and_tab_with_constraint_3": (
            ("Time", "Price", "Reliability"),
            ("Time", "Price", "Reliability"),
        ),
        "shortest_path_map_and_tab_with_constraint_4": (
            ("Time", "Price", "Comfort Level"),
            ("Time", "Price", "Comfort Level"),
        ),
        "shortest_path_map_and_tab_with_constraint_1_2_3_4": ((), ()),
        "shortest_path_map_and_tab_with_constraint_1_2_4": (
            ("Comfort Level",),
            ("Comfort Level",),
        ),
        "shortest_path_map_and_tab_with_constraint_1_3_4": (
            ("Price",),
            ("Price",),
        ),
        "shortest_path_map_and_tab_with_constraint_2_3_4": ((), ()),
        "only_vertex2": ((), ()),
    },
}


def excluded_columns(domain: str, spec: TaskSpec, table_kind: str) -> tuple[str, ...]:
    edge, vertex = _COLUMN_POLICIES.get(domain, {}).get(
        spec.column_policy or spec.name,
        ((), ()),
    )
    return edge if table_kind == "edge" else vertex


def source_filename(domain: str, spec: TaskSpec, split: str) -> str:
    if spec.family in {"qa", "probe"}:
        return f"{domain}_{spec.source}.json"
    if split == "all":
        return f"{domain}_shortest_path_query_{spec.source}.json"
    source_split = "training" if split == "train" else "test"
    return (
        f"{domain}_shortest_path_query_{spec.source}_"
        f"{source_split}_set.json"
    )


def _find_named(root: Path, filename: str, preferred: Iterable[Path]) -> Path:
    for path in preferred:
        if path.is_file():
            return path
    matches = sorted(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"cannot find {filename} below {root}")
    matches.sort(
        key=lambda path: (
            "release" in path.parts,
            len(path.parts),
            str(path),
        )
    )
    return matches[0]


def data_file(
    data_root: str | Path,
    domain: str,
    spec: TaskSpec,
    split: str,
) -> Path:
    root = Path(data_root).expanduser().resolve()
    filename = source_filename(domain, spec, split)
    if spec.family in {"qa", "probe"}:
        preferred = (
            root / domain / "qa_data" / filename,
            root / "raw" / domain / "qa_data" / filename,
        )
    else:
        folder = {
            "train": "training_set",
            "test": "test_set",
            "all": "all",
        }[split]
        preferred = (
            root / domain / "data" / folder / filename,
            root / "raw" / domain / "data" / folder / filename,
        )
    return _find_named(root, filename, preferred)


def load_records(
    data_root: str | Path,
    domain: str,
    spec: TaskSpec,
    split: str,
) -> tuple[list[dict[str, Any]], Path]:
    path = data_file(data_root, domain, spec, split)
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"expected a JSON list in {path}")
    return rows, path


def prompt_text(domain: str, spec: TaskSpec) -> str:
    if spec.prompt is None:
        return ""
    filename = f"{domain}_{spec.prompt}.txt"
    path = resources.files("maptab_infer").joinpath(
        "prompts",
        domain,
        filename,
    )
    if not path.is_file():
        raise FileNotFoundError(f"packaged prompt is missing: {filename}")
    return path.read_text(encoding="utf-8").strip()


def resolve_asset(
    data_root: str | Path,
    relative: str,
    *,
    csv_format: bool = False,
) -> Path:
    root = Path(data_root).expanduser().resolve()
    rel = Path(relative)
    if rel.is_absolute() and rel.is_file():
        return rel
    if csv_format:
        rel = rel.with_suffix(".csv")

    candidates = (
        root / rel,
        root / "raw" / rel,
        root / "assets" / rel,
    )
    for path in candidates:
        if path.is_file():
            return path

    # Upstream TravelMap QA-13 references vertex2 assets that were not
    # released. The complete dataset package audits labels against vertex and
    # preserves the bad raw path; inference uses the available matching asset.
    for suffix in (".json", ".csv"):
        marker = f"_vertex2{suffix}"
        if rel.name.endswith(marker):
            fallback = rel.with_name(
                rel.name.removesuffix(marker) + f"_vertex{suffix}"
            )
            for path in (
                root / fallback,
                root / "raw" / fallback,
                root / "assets" / fallback,
            ):
                if path.is_file():
                    return path

    matches = sorted(root.rglob(rel.name))
    if not matches:
        raise FileNotFoundError(f"cannot resolve asset {rel} below {root}")
    suffix_parts = rel.parts[-3:]
    exact = [
        path
        for path in matches
        if tuple(path.parts[-len(suffix_parts):]) == tuple(suffix_parts)
    ]
    return (exact or matches)[0]


def _format_prompt(template: str, row: dict[str, Any]) -> str:
    weights = list(row.get("weights") or [])
    values: dict[str, Any] = {"question": row.get("question", "")}
    for index in range(4):
        values[f"w{index + 1}"] = (
            weights[index] if index < len(weights) else ""
        )
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(
            f"failed to format prompt for {row.get('question')!r}: {exc}"
        ) from exc


def _json_table(path: Path, excluded: tuple[str, ...]) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    blocked = set(excluded)
    if isinstance(value, list):
        value = [
            {
                key: item_value
                for key, item_value in item.items()
                if key not in blocked
            }
            if isinstance(item, dict)
            else item
            for item in value
        ]
    elif isinstance(value, dict):
        value = {
            key: item_value
            for key, item_value in value.items()
            if key not in blocked
        }
    return json.dumps(value, ensure_ascii=False, indent=2)


def _csv_table(path: Path, excluded: tuple[str, ...]) -> str:
    # MapTab-main uses pandas first, which also fixes its CSV type rendering.
    try:
        import pandas as pd

        frame = pd.read_csv(path, encoding="utf-8")
        if excluded:
            frame = frame.drop(
                columns=[
                    name
                    for name in excluded
                    if name in frame.columns
                ],
                errors="ignore",
            )
        return frame.to_csv(index=False)
    except Exception:
        # Retain the same stdlib fallback used by the upstream providers.
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return path.read_text(encoding="utf-8-sig")
            fieldnames = [
                name
                for name in reader.fieldnames
                if name not in set(excluded)
            ]
            rows = [
                {name: row.get(name, "") for name in fieldnames}
                for row in reader
            ]
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(
            buffer,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue()


def serialize_table(
    path: Path,
    table_kind: str,
    table_format: str,
    excluded: tuple[str, ...],
) -> str:
    body = (
        _csv_table(path, excluded)
        if table_format == "csv"
        else _json_table(path, excluded)
    )
    label = "Edge" if table_kind == "edge" else "Vertex"
    return f"{label} Table ({table_format.upper()}):\n{body}"


def build_content(
    data_root: str | Path,
    domain: str,
    spec: TaskSpec,
    row: dict[str, Any],
) -> list[dict[str, str]]:
    """Build the exact ordered multimodal context for one source record."""

    content: list[dict[str, str]] = []
    template = prompt_text(domain, spec)
    if template:
        content.append(
            {"type": "text", "text": _format_prompt(template, row)}
        )

    for modality in spec.modalities:
        if modality == "image":
            path = resolve_asset(data_root, row["figure"])
            if template:
                content.append(
                    {
                        "type": "text",
                        "text": "This is the subway map image.",
                    }
                )
            content.append({"type": "image_path", "path": str(path)})
            continue

        table_kind = "edge" if modality.startswith("edge_") else "vertex"
        table_format = "csv" if modality.endswith("_csv") else "json"
        field = f"{table_kind}_tab"
        path = resolve_asset(
            data_root,
            row[field],
            csv_format=table_format == "csv",
        )
        if template:
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"This is a {table_kind} table "
                        "of a subway map."
                    ),
                }
            )
        content.append(
            {
                "type": "text",
                "text": serialize_table(
                    path,
                    table_kind,
                    table_format,
                    excluded_columns(domain, spec, table_kind),
                ),
            }
        )

    if not content:
        raise ValueError(f"task {spec.name} produced an empty model context")
    return content


def referenced_assets(
    data_root: str | Path,
    spec: TaskSpec,
    row: dict[str, Any],
) -> list[Path]:
    paths: list[Path] = []
    for modality in spec.modalities:
        if modality == "image":
            paths.append(resolve_asset(data_root, row["figure"]))
        else:
            kind = "edge" if modality.startswith("edge_") else "vertex"
            paths.append(
                resolve_asset(
                    data_root,
                    row[f"{kind}_tab"],
                    csv_format=modality.endswith("_csv"),
                )
            )
    return paths
