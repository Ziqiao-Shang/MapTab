from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from maptab_infer.cli import parser as build_parser
from maptab_infer.data import build_content, resolve_asset, source_filename
from maptab_infer.evaluate import evaluate_planning, evaluate_qa
from maptab_infer.parsing import normalize_generation
from maptab_infer.providers import (
    OPENLUX_MODELS,
    VLLM_MODELS,
    VLLMProvider,
    validate_model,
)
from maptab_infer.tasks import get_task, list_tasks, task_counts


class CliInterfaceTests(unittest.TestCase):
    def test_openlux_is_the_zero_configuration_hosted_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            args = build_parser().parse_args(
                [
                    "generate",
                    "--data-root",
                    "data",
                    "--task",
                    "all-qa",
                ]
            )
        self.assertEqual(args.provider, "openlux")
        self.assertEqual(args.model, "gemini-3.5-flash")
        self.assertFalse(hasattr(args, "base_url"))
        self.assertFalse(hasattr(args, "api_key_env"))

    def test_only_openlux_and_vllm_providers_are_exposed(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                [
                    "generate",
                    "--data-root",
                    "data",
                    "--task",
                    "all-qa",
                    "--provider",
                    "openai",
                ]
            )

    def test_download_defaults_to_aligned_test_package(self) -> None:
        args = build_parser().parse_args(["download"])
        self.assertEqual(args.repo_id, "szq-nju/MapTab")
        self.assertEqual(args.subset, "test")

    def test_supported_model_whitelist(self) -> None:
        for model in OPENLUX_MODELS:
            validate_model("openlux", model)
        for model in VLLM_MODELS:
            validate_model("vllm", model)
        validate_model("vllm", "/models/Qwen3.5-9B")
        with self.assertRaises(ValueError):
            validate_model("openlux", "another-model")
        with self.assertRaises(ValueError):
            validate_model("vllm", "another-model")


    def test_vllm_forces_nonthinking_chat_template(self) -> None:
        seen: dict = {}

        class FakeLLM:
            def __init__(self, **kwargs) -> None:
                seen["init"] = kwargs

            def chat(self, messages, **kwargs):
                seen["messages"] = messages
                seen["chat"] = kwargs
                return [
                    SimpleNamespace(
                        outputs=[SimpleNamespace(text="ok")]
                    )
                ]

        fake_vllm = SimpleNamespace(
            LLM=FakeLLM,
            SamplingParams=lambda **kwargs: kwargs,
        )
        with patch.dict("sys.modules", {"vllm": fake_vllm}):
            provider = VLLMProvider(
                "Qwen/Qwen3.5-9B",
                temperature=0.0,
                max_tokens=16,
                seed=42,
                max_pixels=1_000_000,
                tensor_parallel_size=1,
                max_model_len=4096,
                gpu_memory_utilization=0.9,
                trust_remote_code=True,
            )
            result = provider.generate(
                [{"type": "text", "text": "hello"}]
            )

        self.assertEqual(result.text, "ok")
        self.assertEqual(
            seen["chat"]["chat_template_kwargs"],
            {"enable_thinking": False},
        )


class RegistryTests(unittest.TestCase):
    def test_complete_task_registry(self) -> None:
        self.assertEqual(
            task_counts(),
            {"qa": 22, "planning": 24, "probe": 6},
        )
        self.assertEqual(len(list_tasks()), 52)
        self.assertEqual(
            len({task.name for task in list_tasks()}),
            52,
        )

    def test_representative_variants_are_registered(self) -> None:
        self.assertEqual(
            get_task("13_qa_vertex2_tab_global").variant,
            "additional",
        )
        self.assertEqual(
            get_task("shortest_path_only_csv").variant,
            "csv",
        )
        self.assertEqual(
            get_task(
                "shortest_path_with_qa_and_constraint_1_2_3_4"
            ).output_protocol,
            "route_metrics",
        )
        self.assertFalse(get_task("j_e_tab").scored)

    def test_source_filenames_match_upstream_layout(self) -> None:
        qa = get_task("4_csv_edge_global")
        planning = get_task("shortest_path_only_map")
        self.assertEqual(
            source_filename("metromap", qa, "test"),
            "metromap_4_qa_edge_tab_global.json",
        )
        self.assertEqual(
            source_filename("travelmap", planning, "train"),
            "travelmap_shortest_path_query_only_map_training_set.json",
        )
        self.assertEqual(
            source_filename("travelmap", planning, "test"),
            "travelmap_shortest_path_query_only_map_test_set.json",
        )


class InputAssemblyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "maps").mkdir()
        (self.root / "tables").mkdir()
        (self.root / "maps" / "sample.png").write_bytes(b"image")
        edge = [
            {
                "Source": "A",
                "Destination": "B",
                "Time": 1,
                "Price": 2,
                "Comfort Level": 3,
                "Reliability": 4,
            }
        ]
        vertex = [
            {
                "Vertex": "A",
                "Line": "L1",
                "Time": 5,
                "Transfer Time": 6,
                "Price": 7,
                "Comfort Level": 8,
                "Reliability": 9,
            }
        ]
        (self.root / "tables" / "edge.json").write_text(
            json.dumps(edge),
            encoding="utf-8",
        )
        (self.root / "tables" / "vertex.json").write_text(
            json.dumps(vertex),
            encoding="utf-8",
        )
        (self.root / "tables" / "edge.csv").write_text(
            (
                "Source,Destination,Time,Price,Comfort Level,Reliability\n"
                "A,B,1,2,3,4\n"
            ),
            encoding="utf-8",
        )
        (self.root / "tables" / "vertex.csv").write_text(
            (
                "Vertex,Line,Time,Transfer Time,Price,"
                "Comfort Level,Reliability\n"
                "A,L1,5,6,7,8,9\n"
            ),
            encoding="utf-8",
        )
        self.row = {
            "question": "A to B",
            "weights": [0.1, 0.2, 0.3, 0.4],
            "figure": "maps/sample.png",
            "edge_tab": "tables/edge.json",
            "vertex_tab": "tables/vertex.json",
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _text(content: list[dict[str, str]]) -> str:
        return "\n".join(
            part["text"]
            for part in content
            if part["type"] == "text"
        )

    def test_metromap_constraint_one_masks_non_time_costs(self) -> None:
        content = build_content(
            self.root,
            "metromap",
            get_task("shortest_path_map_and_tab_with_constraint_1"),
            self.row,
        )
        text = self._text(content)
        self.assertIn('"Time": 1', text)
        self.assertNotIn('"Price"', text)
        self.assertNotIn('"Comfort Level"', text)
        self.assertNotIn('"Reliability"', text)
        self.assertIn(
            "This is a edge table of a subway map.",
            text,
        )
        self.assertIn("This is the subway map image.", text)
        self.assertEqual(content[-1]["type"], "image_path")

    def test_travelmap_constraint_two_keeps_time_and_price(self) -> None:
        content = build_content(
            self.root,
            "travelmap",
            get_task("shortest_path_map_and_tab_with_constraint_2"),
            self.row,
        )
        text = self._text(content)
        self.assertIn('"Time": 1', text)
        self.assertIn('"Price": 2', text)
        self.assertNotIn('"Comfort Level"', text)
        self.assertNotIn('"Reliability"', text)

    def test_csv_serialization_uses_source_column_masking(self) -> None:
        content = build_content(
            self.root,
            "metromap",
            get_task("shortest_path_only_csv"),
            self.row,
        )
        text = self._text(content)
        self.assertIn("Edge Table (CSV):", text)
        self.assertIn("Source,Destination", text)
        self.assertNotIn("Comfort Level", text)
        self.assertNotIn("Reliability", text)

    def test_travelmap_vertex2_falls_back_to_released_vertex(self) -> None:
        requested = "tables/sample_vertex2.json"
        available = self.root / "tables" / "sample_vertex.json"
        available.write_text("[]", encoding="utf-8")
        self.assertEqual(
            resolve_asset(self.root, requested),
            available,
        )


class ParsingAndEvaluationTests(unittest.TestCase):
    def test_qa_guided_output_materializes_route_and_metrics(self) -> None:
        spec = get_task("shortest_path_with_qa_and_constraint_1")
        result = normalize_generation(
            spec,
            (
                "<route_begin>A-B-C<route_end>"
                "<Total_Time_begin>12<Total_Time_end>"
                "<Total_Price_begin>8.5<Total_Price_end>"
                "<Average_Comfort Level_begin>3.5"
                "<Average_Comfort Level_end>"
                "<Average_Reliability_begin>4"
                "<Average_Reliability_end>"
            ),
        )
        self.assertEqual(result["response"], "A-B-C")
        self.assertEqual(result["Answer_Total_Time"], "12")
        self.assertEqual(result["Answer_Total_Price"], "8.5")
        self.assertEqual(result["Answer_Average_Comfort_Level"], "3.5")
        self.assertEqual(result["Answer_Average_Reliability"], "4")

    def test_qa_requires_answer_tags_and_rounds_to_two_decimals(self) -> None:
        result = evaluate_qa(
            [
                {
                    "answer": "3.141",
                    "raw_response": "<answer_begin>3.14<answer_end>",
                },
                {"answer": 3.14, "raw_response": "3.14"},
                {
                    "answer": 3.14,
                    "raw_response": "<ANSWER_BEGIN>3.14<ANSWER_END>",
                },
            ]
        )
        self.assertEqual(result["accuracy"], 1 / 3)

    def test_planning_uses_upstream_exact_partial_and_difficulty_rules(self) -> None:
        result = evaluate_planning(
            [
                {
                    "routes": ["A-B-C"],
                    "response": "A-B-C",
                    "Map_Difficulty": "medium",
                    "Query_Difficulty": "hard",
                },
                {
                    "routes": ["A-B-C"],
                    "response": "A-B-X",
                    "Map_Difficulty": "easy",
                    "Query_Difficulty": "easy",
                },
            ]
        )
        self.assertEqual(result["all_acc"], 0.5)
        self.assertAlmostEqual(result["part_acc"], (1.0 + 0.6667) / 2)
        self.assertEqual(result["difficulty_score_total"], 5)

    def test_standard_planning_does_not_relax_output_format(self) -> None:
        result = evaluate_planning(
            [
                {
                    "routes": ["A-B-C"],
                    "response": "<route_begin>A-B-C<route_end>",
                    "Map_Difficulty": "easy",
                    "Query_Difficulty": "easy",
                }
            ]
        )
        self.assertEqual(result["all_acc"], 0.0)


if __name__ == "__main__":
    unittest.main()
