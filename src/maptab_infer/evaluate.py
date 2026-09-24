from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

from .parsing import answer_from_response, route_from_response

_DIFFICULTY = {
    ("easy", "easy"): 2,
    ("easy", "medium"): 3,
    ("easy", "hard"): 4,
    ("medium", "easy"): 3,
    ("medium", "medium"): 4,
    ("medium", "hard"): 5,
    ("hard", "easy"): 4,
    ("hard", "medium"): 5,
    ("hard", "hard"): 6,
}


def semantic_similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(
        None,
        left.strip().lower(),
        right.strip().lower(),
    ).ratio()


def is_same_station(left: str, right: str) -> bool:
    return semantic_similarity(left, right) >= 0.5


def calc_part_acc(gold: list[str], predicted: list[str]) -> float:
    if not gold:
        return 0.0
    matched = 0
    for expected, actual in zip(gold, predicted):
        if not is_same_station(expected, actual):
            break
        matched += 1
    return matched / len(gold)


def calc_all_acc(gold: list[str], predicted: list[str]) -> int:
    if len(gold) != len(predicted):
        return 0
    for expected, actual in zip(gold, predicted):
        if "(transfer)" in expected or "(transfer)" in actual:
            if expected != actual:
                return 0
        elif not is_same_station(expected, actual):
            return 0
    return 1


def _route_parts(value: Any) -> list[str]:
    route = "" if value is None else str(value)
    return route.split("-")


def evaluate_qa(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated: list[dict[str, Any]] = []
    correct_count = 0
    for source in rows:
        row = dict(source)
        raw = row.get("response")
        if raw is None:
            raw = row.get("raw_response", "")
        extracted = answer_from_response(raw)
        try:
            response_num = round(float(extracted), 2) if extracted is not None else None
        except (TypeError, ValueError):
            response_num = None
        try:
            answer_num = round(float(row.get("answer")), 2)
        except (TypeError, ValueError):
            answer_num = None
        correct = int(
            response_num is not None
            and answer_num is not None
            and response_num == answer_num
        )
        row.update(
            {
                "response_num": response_num,
                "answer_num": answer_num,
                "correct": correct,
            }
        )
        evaluated.append(row)
        correct_count += correct
    count = len(evaluated)
    return {
        "accuracy": correct_count / count if count else 0.0,
        "count": count,
        "items": evaluated,
    }


def evaluate_planning(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated: list[dict[str, Any]] = []
    exact_values: list[int] = []
    partial_values: list[float] = []
    difficulty_total = 0

    for source in rows:
        row = dict(source)
        response = row.get("response")
        if response is None:
            response = row.get("raw_response", "")
            if row.get("benchmark_variant") == "qa_guided":
                response = route_from_response(response)
        predicted = _route_parts(response)
        gold_routes = [
            route.split("-")
            for route in row.get("routes", [])
        ]
        exact = max(
            (calc_all_acc(gold, predicted) for gold in gold_routes),
            default=0,
        )
        partial = max(
            (calc_part_acc(gold, predicted) for gold in gold_routes),
            default=0.0,
        )
        map_difficulty = str(
            row.get("Map_Difficulty", row.get("map_difficulty", ""))
        ).lower()
        query_difficulty = str(
            row.get("Query_Difficulty", row.get("query_difficulty", ""))
        ).lower()
        difficulty = _DIFFICULTY.get(
            (map_difficulty, query_difficulty),
            0,
        )
        difficulty_total += difficulty * exact
        row.update(
            {
                "parsed_route": "-".join(predicted),
                "all_acc": exact,
                "part_acc": round(partial, 4),
                "Difficulty_score": difficulty,
            }
        )
        evaluated.append(row)
        exact_values.append(exact)
        partial_values.append(round(partial, 4))

    count = len(evaluated)
    return {
        "all_acc": sum(exact_values) / count if count else 0.0,
        "part_acc": sum(partial_values) / count if count else 0.0,
        "difficulty_score_total": difficulty_total,
        "count": count,
        "items": evaluated,
    }


def evaluate_file(
    input_file: str | Path,
    family: str,
    output_file: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(input_file)
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"expected a JSON list in {path}")
    if family == "qa":
        result = evaluate_qa(rows)
    elif family == "planning":
        result = evaluate_planning(rows)
    else:
        raise ValueError("family must be qa or planning")

    destination = (
        Path(output_file)
        if output_file is not None
        else path.with_name(path.stem + ".evaluated.json")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result["items"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {key: value for key, value in result.items() if key != "items"}
    destination.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
