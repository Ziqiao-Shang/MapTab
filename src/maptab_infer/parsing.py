from __future__ import annotations

import re
from typing import Any

from .tasks import TaskSpec

_TAG_FIELDS = {
    "Total_Time": "Answer_Total_Time",
    "Total_Price": "Answer_Total_Price",
    "Average_Comfort Level": "Answer_Average_Comfort_Level",
    "Average_Reliability": "Answer_Average_Reliability",
}


def extract_tag(text: str, tag: str) -> str | None:
    pattern = (
        rf"<{re.escape(tag)}_begin>\s*(.*?)\s*"
        rf"<{re.escape(tag)}_end>"
    )
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def route_from_response(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    for tag in ("route", "response"):
        extracted = extract_tag(text, tag)
        if extracted is not None:
            return extracted
    fenced = re.fullmatch(r"\s*```(?:text)?\s*(.*?)\s*```\s*", text, re.DOTALL)
    return fenced.group(1).strip() if fenced else text


def answer_from_response(value: Any) -> str | None:
    # Keep QA scoring as strict as MapTab-main: lowercase tags, one line.
    text = "" if value is None else str(value)
    match = re.search(r"<answer_begin>(.*?)<answer_end>", text)
    return match.group(1) if match else None


def normalize_generation(spec: TaskSpec, text: str) -> dict[str, Any]:
    """Preserve raw output and materialize MapTab's task-specific fields."""

    raw = text.strip()
    result: dict[str, Any] = {"raw_response": raw}
    if spec.output_protocol == "route_metrics":
        result["response"] = route_from_response(raw)
        for tag, field in _TAG_FIELDS.items():
            result[field] = extract_tag(raw, tag)
    else:
        result["response"] = raw
    return result
