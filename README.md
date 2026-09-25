# MapTab: Complete Inference and Evaluation

**Official resources:** [Project page](https://ziqiao-shang.github.io/MapTab-Leaderboard/) · [arXiv paper](https://arxiv.org/abs/2602.18600) · [Hugging Face dataset](https://huggingface.co/datasets/szq-nju/MapTab) · [Official code](https://github.com/Ziqiao-Shang/MapTab)

This repository is an inference-only, reproducible implementation of **MapTab: A Diagnostic Benchmark for Long-Horizon Multi-Criteria Multimodal Reasoning on Heterogeneous Topological Graphs**. It follows the complete query-construction and scoring method in `MapTab-main` while covering both MetroMap and TravelMap, all QA and route-planning branches, CSV ablations, the released vertex2 task, and QA-guided planning.

The raw dataset and model weights are not committed here. Download MapTab from the [official Hugging Face dataset](https://huggingface.co/datasets/szq-nju/MapTab) and pass its root to the command line.

![MapTab overview](assets/fig_1.png)

## Table of Contents

- [Overview](#overview)
- [Compatibility with MapTab-main](#compatibility-with-maptab-main)
- [Complete Task Inventory](#complete-task-inventory)
- [Dataset Layout](#dataset-layout)
- [Installation](#installation)
- [Supported Models](#supported-models)
- [End-to-End Test Workflow](#end-to-end-test-workflow)
- [Public Interfaces](#public-interfaces)
- [Inference Backends](#inference-backends)
- [Evaluation](#evaluation)
- [Outputs and Resume Behavior](#outputs-and-resume-behavior)
- [Project Structure](#project-structure)
- [Reproducibility Checks](#reproducibility-checks)
- [Citation](#citation)

## Overview

MapTab tests whether a vision-language model can combine maps, JSON or CSV tables, and natural-language constraints to solve two families of tasks:

1. **Map question answering:** global counting, local counting, and spatial judgment over images, edge tables, vertex tables, or image-table combinations.
2. **Route planning:** shortest-path and weighted multi-criteria planning under time, price, comfort, and reliability constraints.

The implementation exposes 52 inference entries per domain-independent registry:

| Family | Count | Included settings |
| --- | ---: | --- |
| Scored QA | 22 | 12 original QA tasks, released task 13, and 9 CSV variants |
| Scored planning | 24 | 16 standard/ablation variants and 8 QA-guided variants |
| Serialization probes | 6 | JSON/CSV edge, vertex, and vertex2 table-only probes |
| **Total** | **52** | Every branch represented in the source query builders |

Run `maptab-infer list --json` to inspect the machine-readable registry.

## Compatibility with MapTab-main

The implementation was rebuilt from the complete workflow in `MapTab-main/MapTab-main`, especially `src/generate.py`, `src/metromap_utils.py`, `src/travelmap_utils.py`, `src/evaluate_qa.py`, and `src/evaluate_planning.py`.

The following benchmark behavior is preserved:

- The same source JSON file is selected for each domain and subtask.
- The complete MetroMap and TravelMap prompt sets are packaged with the code.
- Model input order follows the source builders: formatted task prompt, table labels and serialized tables, then the map image when required.
- JSON and CSV ablations use the same underlying examples.
- Constraint-specific edge/vertex columns are masked with separate MetroMap and TravelMap policies, matching the source logic.
- QA answers are accepted only from `<answer_begin>...<answer_end>` and compared numerically after rounding to two decimals.
- Planning routes use hyphen-separated stations, SequenceMatcher station similarity at threshold 0.5, strict transfer-station matching, prefix partial accuracy, and the original difficulty-score mapping.
- QA-guided planning preserves `<route_begin>...<route_end>` plus the four metric fields for total time, total price, average comfort, and average reliability.

The release also makes several operational repairs without changing the benchmark definition:

- Explicit `--data-root` replaces machine-specific `WORKSPACE_DIR` path concatenation.
- A declarative task registry replaces two long duplicated conditional builders.
- The supported OpenLux Gemini routes and local vLLM Qwen routes share the same input builder.
- Stable IDs, atomic writes, resume, error retention, and retry are built into generation.
- The released QA task 13 is registered. Because the release has no dedicated task-13 prompt, it uses the corresponding vertex-global prompt.
- Some TravelMap task-13 rows refer to unreleased `*_vertex2.json/.csv` assets. The loader preserves the source record but resolves the verified matching `*_vertex.json/.csv` asset when needed.
- The source query builders define eight QA-guided planning branches, but the provider-side column-routing code does not recognize their names. This implementation maps each branch to its matching constraint policy so all eight are executable.
- QA-guided outputs are normalized before route evaluation, and mixed result directories are evaluated by their recorded family rather than filename guesses.

## Complete Task Inventory

### QA Tasks

The 13 direct QA tasks are:

| Input | Global | Local | Spatial |
| --- | --- | --- | --- |
| Map image | `1_qa_only_pic_global` | `2_qa_only_pic_part` | `3_qa_only_pic_spatial_judge` |
| Edge JSON | `4_qa_edge_tab_global` | `5_qa_edge_tab_part` | `6_qa_edge_tab_spatial_judge` |
| Vertex JSON | `7_qa_vertex_tab_global` | `8_qa_vertex_tab_part` | `9_qa_vertex_tab_spatial_judge` |
| Map + vertex JSON | `10_qa_pic_and_tab_global` | `11_qa_pic_and_tab_part` | `12_qa_pic_and_tab_spatial_judge` |
| Released vertex2 JSON | `13_qa_vertex2_tab_global` | — | — |

The 9 scored CSV counterparts are:

- `4_csv_edge_global`, `5_csv_edge_part`, `6_csv_edge_spatial_judge`
- `7_csv_vertex_global`, `8_csv_vertex_part`, `9_csv_vertex_spatial_judge`
- `10_csv_and_pic_global`, `11_csv_and_pic_part`, `12_csv_and_pic_spatial_judge`

The six source-compatible serialization probes are `j_e_tab`, `j_v_tab`, `j_v2_tab`, `c_e_tab`, `c_v_tab`, and `c_v2_tab`. They send only the selected table representation and intentionally have no benchmark score. Use `all-probes` to run them; `all-qa` contains only scored QA tasks.

### Route-Planning Tasks

The 16 standard and ablation variants are:

| Group | Task names |
| --- | --- |
| Single/combined inputs | `shortest_path_only_map`, `shortest_path_only_tab`, `shortest_path_only_csv`, `shortest_path_map_and_tab_no_constraint`, `shortest_path_map_and_csv` |
| One constraint | `shortest_path_map_and_tab_with_constraint_1`, `shortest_path_map_and_tab_with_constraint_2`, `shortest_path_map_and_tab_with_constraint_3`, `shortest_path_map_and_tab_with_constraint_4` |
| Multiple constraints | `shortest_path_map_and_tab_with_constraint_1_2_3_4`, `shortest_path_map_and_tab_with_constraint_1_2_4`, `shortest_path_map_and_tab_with_constraint_1_3_4`, `shortest_path_map_and_tab_with_constraint_2_3_4` |
| Vertex2 and CSV | `only_vertex2`, `shortest_path_csv_vertex2`, `shortest_path_map_and_tab_csv_constraint_1_2_3_4` |

The constraint IDs retain the upstream meaning: 1 = time, 2 = price, 3 = comfort, and 4 = reliability. The input builder applies the exact domain-specific column policy before serialization.

The eight QA-guided planning variants are:

- `shortest_path_with_qa_and_constraint_1`
- `shortest_path_with_qa_and_constraint_2`
- `shortest_path_with_qa_and_constraint_3`
- `shortest_path_with_qa_and_constraint_4`
- `shortest_path_with_qa_and_constraint_1_2_3_4`
- `shortest_path_with_qa_and_constraint_1_2_4`
- `shortest_path_with_qa_and_constraint_1_3_4`
- `shortest_path_with_qa_and_constraint_2_3_4`

`all-planning` runs all 24 planning entries. `canonical-planning` excludes CSV and QA-guided ablations.

## Dataset Layout

The inference repository consumes the exact directory tree published by
`szq-nju/MapTab`. No conversion step is required:

```text
DATA_ROOT/
|-- data/
|   |-- metromap_qa/test-*.parquet
|   |-- travelmap_qa/test-*.parquet
|   |-- metromap_planning/test-*.parquet
|   `-- travelmap_planning/test-*.parquet
|-- assets/
|   |-- metromap/{images,tabulars}/
|   `-- travelmap/{images,tabulars}/
`-- raw/
    |-- metromap/{qa_data,data/test_set,prompts}/
    `-- travelmap/{qa_data,data/test_set,prompts}/
```

The normalized Parquet files work with `datasets.load_dataset`. Exact
benchmark inference reads the lossless query records under `raw/` and
resolves their repository-relative image and table paths under `assets/`.
This is the layout generated in `Maptab_hug`; once that directory is
uploaded, the GitHub code and Hugging Face dataset are directly aligned.

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/Ziqiao-Shang/MapTab.git
cd MapTab
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For local Qwen inference, install the vLLM extra in a CUDA-compatible
environment:

```bash
python -m pip install -e ".[local]"
```

## Supported Models

Only these four model routes are accepted:

| Backend | Supported model | Execution |
| --- | --- | --- |
| OpenLux | `gemini-3-flash` | Hosted multimodal API |
| OpenLux | `gemini-3.5-flash` | Hosted multimodal API; default |
| vLLM | `Qwen/Qwen3.5-9B` | Local multimodal inference; non-thinking |
| vLLM | `Qwen/Qwen3-VL-8B-Instruct` | Local multimodal inference |

The CLI rejects every other provider or model. A local directory is accepted
only when its final directory name matches one of the two supported Qwen
models. The vLLM adapter always passes
`chat_template_kwargs={"enable_thinking": False}`, which forces Qwen3.5
into non-thinking mode.

## End-to-End Test Workflow

### 1. Download the Hugging Face test package

The default command downloads only test queries, normalized test Parquet
shards, referenced assets, and package metadata:

```bash
maptab-infer download \
  --repo-id szq-nju/MapTab \
  --revision main \
  --subset test \
  --data-root ./data
```

Equivalent direct Hugging Face code:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="szq-nju/MapTab",
    repo_type="dataset",
    revision="main",
    local_dir="./data",
    allow_patterns=[
        "README.md",
        "dataset_card.md",
        "manifest.json",
        "maptab_loader.py",
        "requirements.txt",
        "data/metromap_qa/test-*.parquet",
        "data/travelmap_qa/test-*.parquet",
        "data/metromap_planning/test-*.parquet",
        "data/travelmap_planning/test-*.parquet",
        "assets/metromap/**",
        "assets/travelmap/**",
        "raw/metromap/qa_data/**",
        "raw/travelmap/qa_data/**",
        "raw/metromap/data/test_set/**",
        "raw/travelmap/data/test_set/**",
        "raw/metromap/prompts/**",
        "raw/travelmap/prompts/**",
    ],
)
```

Use `--subset full` only when training data is also required. The Bash
wrapper is `bash scripts/download_data.sh ./data test`.

### 2. Validate every test input

```bash
maptab-infer validate \
  --data-root ./data \
  --domain all \
  --task all \
  --split test
```

Validation loads every query source, formats every packaged prompt, and
resolves every image and table before API or GPU inference begins.

### 3. Inspect one exact request

```bash
maptab-infer inspect \
  --data-root ./data \
  --domain metromap \
  --task shortest_path_map_and_tab_with_constraint_1_2_3_4 \
  --split test \
  --index 0
```

### 4. Run a supported model

OpenLux:

```bash
export OPENLUX_API_KEY=your_key_here

maptab-infer generate \
  --data-root ./data \
  --domain all \
  --task all-qa \
  --provider openlux \
  --model gemini-3.5-flash
```

vLLM with Qwen3.5-9B in non-thinking mode:

```bash
maptab-infer generate \
  --data-root ./data \
  --domain all \
  --task all-planning \
  --split test \
  --provider vllm \
  --model Qwen/Qwen3.5-9B \
  --tensor-parallel-size 4 \
  --max-model-len 128000
```

For Qwen3-VL-8B, change only
`--model Qwen/Qwen3-VL-8B-Instruct`.

### 5. Evaluate

```bash
maptab-infer evaluate-dir \
  --input-dir results/response_generate \
  --output-dir results_evaluate \
  --family auto
```

This is the complete path from the Hugging Face release to validation,
generation, and benchmark scoring.

## Public Interfaces

The supported public surface contains:

1. `maptab-infer list|download|validate|inspect|generate|evaluate|evaluate-dir`.
2. Bash entry points under `scripts/`.
3. The Python task registry, input builder, `OpenLuxProvider`,
   `VLLMProvider`, runner, and evaluators.

See [the API reference](docs/api.md) for exact command and Python signatures.

## Inference Backends

### OpenLux

OpenLux uses the fixed `https://api.openlux.ai/v1` endpoint, reads
`OPENLUX_API_KEY`, and accepts only `gemini-3-flash` or
`gemini-3.5-flash`. Keys are never accepted as command-line arguments or
written to results.

### In-process vLLM

vLLM accepts only `Qwen/Qwen3.5-9B` and
`Qwen/Qwen3-VL-8B-Instruct`, or matching local directory names.
`TENSOR_PARALLEL_SIZE`, `MAX_MODEL_LEN`, and
`GPU_MEMORY_UTILIZATION` control local execution.

### Bash entry points

| Scope | Script |
| --- | --- |
| Download test/full package | `scripts/download_data.sh DATA_ROOT [test|full]` |
| Generic QA | `scripts/generate_qa.sh` |
| Generic planning | `scripts/generate_rp.sh` |
| MetroMap/TravelMap QA | `scripts/run_metromap_qa.sh`, `scripts/run_travelmap_qa.sh` |
| MetroMap/TravelMap planning | `scripts/run_metromap_planning.sh`, `scripts/run_travelmap_planning.sh` |
| One task | `scripts/run_qa_task.sh`, `scripts/run_planning_task.sh` |
| All QA | `scripts/run_all_qa.sh DATA_ROOT [MODEL]` |
| All planning | `scripts/run_all_planning.sh DATA_ROOT [MODEL] [SPLIT]` |
| Complete generation | `scripts/run_all.sh` |
| Evaluation | `scripts/evaluate_qa.sh`, `scripts/evaluate_rp.sh`, `scripts/evaluate_all.sh` |

OpenLux:

```bash
export MAPTAB_DATA_ROOT="$PWD/data"
export OPENLUX_API_KEY=your_key_here
export PROVIDER=openlux
export MODEL_PATH=gemini-3-flash
bash scripts/run_all.sh
```

vLLM:

```bash
export MAPTAB_DATA_ROOT="$PWD/data"
export PROVIDER=vllm
export MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct
export TENSOR_PARALLEL_SIZE=4
bash scripts/run_all.sh
```

The compatibility wrapper retains the upstream argument names:

```bash
PYTHONPATH=src python src/generate.py \
  --task metromap \
  --subtask shortest_path_only_map \
  --provider openlux \
  --model_path gemini-3.5-flash \
  --data_root ./data
```

Useful selectors are `all`, `all-qa`, `all-planning`, `all-probes`,
`canonical-qa`, and `canonical-planning`. Planning supports
`--split train|test|all`; a test-only download supports test inference.

## Evaluation

Evaluate one file with automatic family detection:

```bash
maptab-infer evaluate \
  --family auto \
  --input results/response_generate/metromap_1_qa_only_pic_global_MODEL_results.json
```

Evaluate a mixed directory:

```bash
maptab-infer evaluate-dir \
  --input-dir results/response_generate \
  --output-dir results_evaluate \
  --family auto
```

The metrics are:

| Family | Metrics |
| --- | --- |
| QA | Numeric `accuracy` after strict tag extraction and two-decimal rounding |
| Planning | `all_acc`, prefix `part_acc`, and `difficulty_score_total` |

The evaluator writes an item-level `*.evaluated.json` file and a machine-readable `*.summary.json` next to it in the selected evaluation directory.

## Outputs and Resume Behavior

Generation uses the source-compatible filename:

```text
{domain}_{subtask}_{model-name}_results.json
```

Each item preserves the complete source record and adds provenance and normalized generation fields:

```json
{
  "_id": "metromap:qa:1_qa_only_pic_global:test:000000",
  "_source_index": 0,
  "benchmark_domain": "metromap",
  "benchmark_task": "1_qa_only_pic_global",
  "benchmark_family": "qa",
  "benchmark_variant": "canonical",
  "benchmark_split": "test",
  "source_file": "...",
  "model": "gemini-3.5-flash",
  "raw_response": "<answer_begin>42<answer_end>",
  "response": "<answer_begin>42<answer_end>",
  "reasoning_content": null,
  "error": null
}
```

Writes are atomic. Existing successful IDs are skipped by default, `--retry-errors` reruns only failed items, and `--overwrite` starts that output file again. `--offset` and `--limit` support bounded runs.

## Project Structure

```text
MapTab_git/
├── assets/
│   └── fig_1.png
├── docs/
│   └── api.md
├── scripts/
│   ├── download_data.sh
│   ├── generate_qa.sh
│   ├── generate_rp.sh
│   ├── run_metromap_qa.sh
│   ├── run_travelmap_qa.sh
│   ├── run_metromap_planning.sh
│   ├── run_travelmap_planning.sh
│   ├── run_qa_task.sh
│   ├── run_planning_task.sh
│   ├── run_all_qa.sh
│   ├── run_all_planning.sh
│   ├── run_all.sh
│   ├── evaluate_qa.sh
│   ├── evaluate_rp.sh
│   └── evaluate_all.sh
├── src/
│   ├── generate.py
│   ├── evaluate_qa.py
│   ├── evaluate_planning.py
│   └── maptab_infer/
│       ├── cli.py
│       ├── data.py
│       ├── evaluate.py
│       ├── parsing.py
│       ├── providers.py
│       ├── runner.py
│       ├── tasks.py
│       └── prompts/{metromap,travelmap}/
├── tests/
│   └── test_registry.py
├── README.md
└── pyproject.toml
```

## Reproducibility Checks

The following checks require no model API:

```bash
python -m compileall -q src tests
PYTHONPATH=src python -m unittest discover -s tests -v
bash -n scripts/*.sh

maptab-infer validate \
  --data-root /path/to/MapTab \
  --domain all \
  --task all \
  --split test
```

The test suite checks registry completeness, source-compatible filenames, domain-specific table masking, TravelMap vertex2 fallback, QA-guided parsing, and the original QA/planning metric behavior.

## Security and Release Scope

No raw MapTab dataset, model weights, historical responses, caches, or credentials are included. The code reads API credentials only from an environment variable. Before publishing results, retain the dataset revision, model identifier or checkpoint hash, decoding settings, prompt files, and generated raw responses.

## Citation

If you use MapTab in your research, please cite:

```bibtex
@article{shang2026maptab,
  title={MapTab: A Diagnostic Benchmark for Long-Horizon Multi-Criteria Multimodal Reasoning on Heterogeneous Topological Graphs},
  author={Shang, Ziqiao and Ge, Lingyue and Xu, Zian and Cheng, Zi-Jian and Tian, Shi-Yu and Huang, Zhenyu and Fu, Wenbo and Wu, Weiming and Chen, Yang and Zhang, Xiangwen and Hu, Yulan and Liu, Bin and Guo, Lan-Zhe},
  journal={arXiv preprint arXiv:2602.18600},
  year={2026}
}
```

## License

No standalone license file was present in the source snapshot used for this inference release. Add the intended code license before public redistribution. Dataset use is governed separately by the terms published with the official MapTab dataset.
