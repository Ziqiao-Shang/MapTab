from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Family = Literal["qa", "planning", "probe"]
Variant = Literal["canonical", "additional", "csv", "qa_guided", "probe"]
OutputProtocol = Literal["numeric_answer", "route", "route_metrics", "none"]


@dataclass(frozen=True)
class TaskSpec:
    """Declarative description of one MapTab inference subtask."""

    name: str
    family: Family
    source: str
    prompt: str | None
    modalities: tuple[str, ...]
    variant: Variant
    description: str
    column_policy: str | None = None
    output_protocol: OutputProtocol = "route"

    @property
    def ablation(self) -> bool:
        return self.variant in {"csv", "qa_guided", "probe"}

    @property
    def scored(self) -> bool:
        return self.output_protocol != "none"


_TASKS: dict[str, TaskSpec] = {}


def _add(
    name: str,
    family: Family,
    source: str,
    prompt: str | None,
    modalities: tuple[str, ...],
    variant: Variant,
    description: str,
    *,
    column_policy: str | None = None,
    output_protocol: OutputProtocol,
) -> None:
    if name in _TASKS:
        raise ValueError(f"duplicate task: {name}")
    _TASKS[name] = TaskSpec(
        name,
        family,
        source,
        prompt,
        modalities,
        variant,
        description,
        column_policy or name,
        output_protocol,
    )


_QA_TASKS = (
    ("1_qa_only_pic_global", ("image",), "Global map-image counting"),
    ("2_qa_only_pic_part", ("image",), "Local map-image counting"),
    ("3_qa_only_pic_spatial_judge", ("image",), "Map-image spatial judgment"),
    ("4_qa_edge_tab_global", ("edge_json",), "Global edge-table question"),
    ("5_qa_edge_tab_part", ("edge_json",), "Local edge-table question"),
    ("6_qa_edge_tab_spatial_judge", ("edge_json",), "Edge-table spatial judgment"),
    ("7_qa_vertex_tab_global", ("vertex_json",), "Global vertex-table question"),
    ("8_qa_vertex_tab_part", ("vertex_json",), "Local vertex-table question"),
    ("9_qa_vertex_tab_spatial_judge", ("vertex_json",), "Vertex-table spatial judgment"),
    ("10_qa_pic_and_tab_global", ("vertex_json", "image"), "Global image-and-table question"),
    ("11_qa_pic_and_tab_part", ("vertex_json", "image"), "Local image-and-table question"),
    ("12_qa_pic_and_tab_spatial_judge", ("vertex_json", "image"), "Image-and-table spatial judgment"),
)
for name, modalities, description in _QA_TASKS:
    _add(
        name,
        "qa",
        name,
        name,
        modalities,
        "canonical",
        description,
        output_protocol="numeric_answer",
    )

# Released task 13 has no dedicated prompt; reuse the vertex-global prompt.
_add(
    "13_qa_vertex2_tab_global",
    "qa",
    "13_qa_vertex2_tab_global",
    "7_qa_vertex_tab_global",
    ("vertex_json",),
    "additional",
    "Global question over the released vertex2 table",
    output_protocol="numeric_answer",
)

_QA_CSV = (
    ("4_csv_edge_global", "4_qa_edge_tab_global", ("edge_csv",)),
    ("5_csv_edge_part", "5_qa_edge_tab_part", ("edge_csv",)),
    ("6_csv_edge_spatial_judge", "6_qa_edge_tab_spatial_judge", ("edge_csv",)),
    ("7_csv_vertex_global", "7_qa_vertex_tab_global", ("vertex_csv",)),
    ("8_csv_vertex_part", "8_qa_vertex_tab_part", ("vertex_csv",)),
    ("9_csv_vertex_spatial_judge", "9_qa_vertex_tab_spatial_judge", ("vertex_csv",)),
    ("10_csv_and_pic_global", "10_qa_pic_and_tab_global", ("vertex_csv", "image")),
    ("11_csv_and_pic_part", "11_qa_pic_and_tab_part", ("vertex_csv", "image")),
    (
        "12_csv_and_pic_spatial_judge",
        "12_qa_pic_and_tab_spatial_judge",
        ("vertex_csv", "image"),
    ),
)
for name, source, modalities in _QA_CSV:
    _add(
        name,
        "qa",
        source,
        source,
        modalities,
        "csv",
        f"CSV counterpart of {source}",
        output_protocol="numeric_answer",
    )

# Six MapTab-main table-only branches are diagnostics, not scored QA.
_PROBES = (
    ("j_e_tab", "4_qa_edge_tab_global", "edge_json"),
    ("j_v_tab", "7_qa_vertex_tab_global", "vertex_json"),
    ("j_v2_tab", "13_qa_vertex2_tab_global", "vertex_json"),
    ("c_e_tab", "4_qa_edge_tab_global", "edge_csv"),
    ("c_v_tab", "7_qa_vertex_tab_global", "vertex_csv"),
    ("c_v2_tab", "13_qa_vertex2_tab_global", "vertex_csv"),
)
for name, source, modality in _PROBES:
    _add(
        name,
        "probe",
        source,
        None,
        (modality,),
        "probe",
        f"{modality} serialization probe",
        output_protocol="none",
    )


_PLANNING = (
    (
        "shortest_path_only_map",
        "only_map",
        "shortest_path_only_map",
        ("image",),
        "canonical",
        "Shortest-stop route from the map image",
        "shortest_path_only_map",
    ),
    (
        "shortest_path_only_tab",
        "only_tab",
        "shortest_path_only_tab",
        ("edge_json",),
        "canonical",
        "Shortest route from the JSON edge table",
        "shortest_path_only_tab",
    ),
    (
        "shortest_path_only_csv",
        "only_tab",
        "shortest_path_only_tab",
        ("edge_csv",),
        "csv",
        "CSV ablation of table-only planning",
        "shortest_path_only_tab",
    ),
    (
        "shortest_path_map_and_tab_no_constraint",
        "map_and_tab",
        "shortest_path_map_and_tab",
        ("edge_json", "image"),
        "canonical",
        "Unconstrained map-and-table planning",
        "shortest_path_map_and_tab_no_constraint",
    ),
    (
        "shortest_path_map_and_csv",
        "map_and_tab",
        "shortest_path_map_and_tab",
        ("edge_csv", "image"),
        "csv",
        "CSV ablation of unconstrained planning",
        "shortest_path_map_and_tab_no_constraint",
    ),
)
for name, source, prompt, modalities, variant, description, policy in _PLANNING:
    _add(
        name,
        "planning",
        source,
        prompt,
        modalities,
        variant,
        description,
        column_policy=policy,
        output_protocol="route",
    )

_CONSTRAINTS = ("1", "2", "3", "4", "1_2_3_4", "1_2_4", "1_3_4", "2_3_4")
for constraints in _CONSTRAINTS:
    source = f"map_and_tab_with_constraint_{constraints}"
    policy = f"shortest_path_map_and_tab_with_constraint_{constraints}"
    _add(
        policy,
        "planning",
        source,
        f"shortest_path_with_constraint_{constraints}",
        ("edge_json", "vertex_json", "image"),
        "canonical",
        f"Planning with constraint set {constraints.replace('_', ', ')}",
        column_policy=policy,
        output_protocol="route",
    )

_add(
    "only_vertex2",
    "planning",
    "map_and_tab_with_constraint_1_2_3_4_only_vertex2",
    "shortest_path_with_constraint_1_2_3_4_only_vertex2",
    ("vertex_json", "image"),
    "additional",
    "Vertex2-and-image full-constraint planning",
    column_policy="only_vertex2",
    output_protocol="route",
)
_add(
    "shortest_path_csv_vertex2",
    "planning",
    "map_and_tab_with_constraint_1_2_3_4_only_vertex2",
    "shortest_path_with_constraint_1_2_3_4_only_vertex2",
    ("vertex_csv", "image"),
    "csv",
    "CSV ablation of vertex2 planning",
    column_policy="only_vertex2",
    output_protocol="route",
)
_add(
    "shortest_path_map_and_tab_csv_constraint_1_2_3_4",
    "planning",
    "map_and_tab_with_constraint_1_2_3_4",
    "shortest_path_with_constraint_1_2_3_4",
    ("edge_csv", "vertex_csv", "image"),
    "csv",
    "CSV ablation of full-constraint planning",
    column_policy="shortest_path_map_and_tab_with_constraint_1_2_3_4",
    output_protocol="route",
)

for constraints in _CONSTRAINTS:
    _add(
        f"shortest_path_with_qa_and_constraint_{constraints}",
        "planning",
        f"map_and_tab_with_constraint_{constraints}",
        f"shortest_path_with_qa_and_constraint_{constraints}",
        ("edge_json", "vertex_json", "image"),
        "qa_guided",
        f"QA-guided planning with constraint set {constraints.replace('_', ', ')}",
        column_policy=f"shortest_path_map_and_tab_with_constraint_{constraints}",
        output_protocol="route_metrics",
    )


def get_task(name: str) -> TaskSpec:
    try:
        return _TASKS[name]
    except KeyError as exc:
        raise KeyError(f"unknown task {name!r}; run maptab-infer list") from exc


def list_tasks(
    family: Family | None = None,
    include_ablations: bool = True,
) -> list[TaskSpec]:
    """Return tasks in stable order and retain the old public signature."""

    values = list(_TASKS.values())
    if not include_ablations:
        values = [
            task
            for task in values
            if task.variant in {"canonical", "additional"}
        ]
    return sorted(
        [task for task in values if family is None or task.family == family],
        key=lambda task: (task.family, task.name),
    )


def task_counts() -> dict[str, int]:
    return {
        family: len(list_tasks(family))
        for family in ("qa", "planning", "probe")
    }
