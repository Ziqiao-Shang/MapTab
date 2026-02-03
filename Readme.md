# MapTab: A Benchmark for Evaluating Vision-Language Models on Metromap/Travelmap Understanding

MapTab is a comprehensive benchmark designed to evaluate the map understanding and spatial reasoning capabilities of Vision-Language Models (VLMs). The benchmark focuses on two core tasks: **route planning** and **map-based QA**, using both metro maps and travel maps.

## 📋 Table of Contents

- [Overview](#overview)
- [Task Description](#task-description)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Models](#supported-models)
- [Citation](#citation)

## Overview

MapTab evaluates VLMs on their ability to:
1. **Understand map visualizations** - Parse and interpret visual map information
2. **Process tabular data** - Understand structured information in tables (JSON/CSV)
3. **Perform spatial reasoning** - Plan routes and answer spatial questions
4. **Follow complex constraints** - Handle multi-constraint route planning tasks

## Task Description

### 1. Route Planning Tasks

The benchmark includes various route planning subtasks with different input modalities and constraint levels:

| Subtask | Input Modality | Description |
|---------|---------------|-------------|
| `shortest_path_only_map` | Map Image | Route planning using only visual map |
| `shortest_path_only_tab` | Table (JSON) | Route planning using only tabular data |
| `shortest_path_only_csv` | Table (CSV) | Route planning using only CSV data (ablation) |
| `shortest_path_map_and_tab_no_constraint` | Map + Table | Combined input without constraints |
| `shortest_path_map_and_csv` | Map + CSV | Combined with CSV format (ablation) |
| `shortest_path_map_and_tab_with_constraint_1` | Map + Table | With constraint type 1 |
| `shortest_path_map_and_tab_with_constraint_2` | Map + Table | With constraint type 2 |
| `shortest_path_map_and_tab_with_constraint_3` | Map + Table | With constraint type 3 |
| `shortest_path_map_and_tab_with_constraint_4` | Map + Table | With constraint type 4 |
| `shortest_path_map_and_tab_with_constraint_1_2_3_4` | Map + Table | With all four constraints |
| `shortest_path_map_and_tab_with_constraint_1_2_4` | Map + Table | With constraints 1, 2, and 4 |
| `shortest_path_map_and_tab_with_constraint_1_3_4` | Map + Table | With constraints 1, 3, and 4 |
| `shortest_path_map_and_tab_with_constraint_2_3_4` | Map + Table | With constraints 2, 3, and 4 |
| `only_vertex2` | Map + Table | Special vertex subset task |
| `shortest_path_csv_vertex2` | Map + CSV | CSV format with vertex subset (ablation) |
| `shortest_path_map_and_tab_csv_constraint_1_2_3_4` | Map + CSV | CSV format with all constraints (ablation) |

## Dataset

The dataset includes two map types:

- **MetroMap**: Synthetic metro/subway network maps
- **TravelMap**: Travel route maps with geographic information

We have only released a portion of the benchmark. The complete training set and test set will be released in the future.

### Dataset Status

The current release includes:
- ✅ **Route planning task data** - Available in `{metromap,travelmap}/data/test_set/`
- ✅ **Map images** - Available in `{metromap,travelmap}/images/`
- ✅ **Tabular data** - Available in `{metromap,travelmap}/tabulars/`
- ✅ **Prompt templates** - Available in `{metromap,travelmap}/prompts/`
- ⏳ **QA task data** - To be released separately

> **Note**: Question Answering (QA) task data will be released separately. Please check our project page for updates on the QA dataset release.

## Installation

### Requirements

```bash
# Clone the repository
git clone https://github.com/your-repo/MapTab.git
cd MapTab
```

### Dependencies

- Python >= 3.8
- PyTorch >= 2.0
- OpenAI SDK (for API-based models)
- VLLM (for local model inference)
- NumPy
- Other dependencies

## Quick Start

### 1. Set Environment Variables

```bash
export WORKSPACE_DIR="/path/to/MapTab"
export API_KEY="your-api-key"  # For API-based models
```

### 2. Run with local model (planning task)

```bash
bash scripts/generate.sh 
```

### 3. Evaluate planning results

```bash
bash scripts/evaluate.sh
```

## Supported Models

### API-Based Models

| Provider | Models | API Key Environment Variable |
|----------|--------|------------------------------|
| Azure OpenAI | gpt-4o, gpt-4.1, gpt-5 | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` |
| Qwen/DashScope | qwen3-vl-plus, qwen3-max, qwen3-vl-32b-instruct, qwen3-vl-8b-instruct, qwen3-vl-30b-a3b-instruct, qwen3-vl-32b-thinking, qwen3-vl-8b-thinking | `DASHSCOPE_API_KEY` |
| Doubao/Volcengine | doubao-seed-1-6-251015 | `ARK_API_KEY` |
| Gemini (via Yunwu) | gemini-3-flash-preview | `YUNWU_TOKEN` |

### Local Models (via vLLM)

**Primary Evaluation Models:**
- Qwen/Qwen3-VL-8B-Instruct
- Qwen/Qwen3-VL-8B-Thinking
- Qwen/Qwen3-VL-30B-A3B-Thinking
- Qwen/Qwen2.5-VL-7B-Instruct
- Kimi/Kimi-VL-A3B-Thinking-2506
- Kimi/Kimi-VL-A3B-Instruct
- microsoft/Phi-4-multimodal-instruct
- microsoft/Phi-3.5-vision-instruct

**Additional Supported Models:**
- Qwen/Qwen3-VL-2B-Instruct
- llava-hf/llava-v1.6-mistral-7b-hf
- OpenGVLab/InternVL3_5-30B-A3B
- OpenGVLab/InternVL3_5-8B
- AIDC-AI/Ovis2.5-9B

> Note: For local models, ensure you have vLLM installed and properly configured. Model paths should match HuggingFace repository names.

**Output:** Results are saved to `results/response_generate/{task}_{subtask}_{model_name}_results.json`

### Metrics
- `all_acc`: Exact match accuracy (complete route correctness)
- `part_acc`: Partial accuracy (proportion of correct route segments)
- `Difficulty_score`: Difficulty-weighted score based on map and query complexity

## Citation

Paper Information is coming soon.
