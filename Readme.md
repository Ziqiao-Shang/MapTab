# MapTab: A Benchmark for Evaluating Vision-Language Models on Map Understanding

MapTab is a comprehensive benchmark designed to evaluate the map understanding and spatial reasoning capabilities of Vision-Language Models (VLMs). The benchmark focuses on two core tasks: **route planning** and **map-based question answering**, using both metro maps and travel maps.

## 📋 Table of Contents

- [Overview](#overview)
- [Task Description](#task-description)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Models](#supported-models)
- [Usage](#usage)
  - [Response Generation](#response-generation)
  - [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Citation](#citation)
- [License](#license)

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

### 2. Question Answering Tasks

> **Note**: QA task data files will be released separately. The code infrastructure is ready to support the following QA subtasks.

QA tasks evaluate map comprehension across different aspects:

| Subtask ID | Task Type | Description |
|------------|-----------|-------------|
| 1 | `1_qa_only_pic_global` | Global questions using only map image |
| 2 | `2_qa_only_pic_part` | Local/partial questions using only map image |
| 3 | `3_qa_only_pic_spatial_judge` | Spatial judgment using only map image |
| 4 | `4_qa_edge_tab_global` | Global edge questions with table |
| 5 | `5_qa_edge_tab_part` | Local edge questions with table |
| 6 | `6_qa_edge_tab_spatial_judge` | Spatial edge judgment with table |
| 7 | `7_qa_vertex_tab_global` | Global vertex questions with table |
| 8 | `8_qa_vertex_tab_part` | Local vertex questions with table |
| 9 | `9_qa_vertex_tab_spatial_judge` | Spatial vertex judgment with table |
| 10 | `10_qa_pic_and_tab_global` | Global questions with map and table |
| 11 | `11_qa_pic_and_tab_part` | Local questions with map and table |
| 12 | `12_qa_pic_and_tab_spatial_judge` | Spatial judgment with map and table |

## Dataset

The dataset includes two map types:

- **MetroMap**: Synthetic metro/subway network maps
- **TravelMap**: Travel route maps with geographic information

### Dataset Status

The current release includes:
- ✅ **Route planning task data** - Available in `{metromap,travelmap}/data/test_set/`
- ✅ **Map images** - Available in `{metromap,travelmap}/images/`
- ✅ **Tabular data** - Available in `{metromap,travelmap}/tabulars/`
- ✅ **Prompt templates** - Available in `{metromap,travelmap}/prompts/`
- ⏳ **QA task data** - To be released separately

> **Note**: Question Answering (QA) task data will be released separately. Please check our project page for updates on the QA dataset release.

### Data Format

Each data sample contains:
```json
{
    "country": "cambodia",
    "scenic_spot": "angkorWar",
    "figure": "travelmap/images/cambodia/angkorWar.png",
    "edge_tab": "travelmap/tabulars/cambodia/angkorWar_edge.json",
    "vertex_tab": "travelmap/tabulars/cambodia/angkorWar_vertex.json",
    "spot_1": "巴方寺",
    "spot_2": "比粒寺",
    "Map_Difficulty": "Hard",
    "Query_Difficulty": "Hard",
    "question": "According to the Scenic Area Planning Map, Edge Table, what is the shortest path from 巴方寺 to 比粒寺?",
    "routes": [
        "巴方寺-巴戎寺-圣剑寺-龙蟠水池-塔逊庙-东梅硼-比粒寺"
    ],
    "route_vertex_numbers": [
        7
    ],
    "qdi_rank": 133
}
```

## Installation

### Requirements

```bash
# Clone the repository
git clone https://github.com/your-repo/MapTab.git
cd MapTab

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- Python >= 3.8
- PyTorch >= 2.0
- OpenAI SDK (for API-based models)
- vLLM (for local model inference)
- NumPy
- Other dependencies in `requirements.txt`

## Quick Start

### 1. Set Environment Variables

```bash
export WORKSPACE_DIR="/path/to/MapTab"
export API_KEY="your-api-key"  # For API-based models
```

### 2. Run Generation

# Run with local model (planning task)
bash scripts/generate.sh 

### 3. Run Evaluation

# Evaluate planning results
bash scripts/evaluate.sh


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

## Usage

### Response Generation

```bash
python src/generate.py \
    --task <task_name> \
    --subtask <subtask_name> \
    --model_path <model_path_or_name> \
    [--api_key <api_key>] \
    [--seed <random_seed>]
```

**Arguments:**
- `--task`: Task name (`metromap` or `travelmap`)
- `--subtask`: Subtask name (see Task Description section for available subtasks)
- `--model_path`: Model identifier or HuggingFace model path
  - For API models: Use model name (e.g., `gpt-4o`, `qwen3-vl-plus`)
  - For local models: Use HuggingFace path (e.g., `Qwen/Qwen3-VL-8B-Instruct`)
- `--api_key`: API key for API-based models (optional for local models)
- `--seed`: Random seed for reproducibility (default: 42, used only for local models)

**Output:** Results are saved to `results/response_generate/{task}_{subtask}_{model_name}_results.json`

### Evaluation

#### Planning Evaluation

```bash
python src/evaluate_planning.py --input_file <path_to_results.json>
```

**Metrics:**
- `all_acc`: Exact match accuracy (complete route correctness)
- `part_acc`: Partial accuracy (proportion of correct route segments)
- `Difficulty_score`: Difficulty-weighted score based on map and query complexity

#### QA Evaluation

```bash
python src/evaluate_qa.py --input_file <path_to_results.json>
```

**Metrics:**
- `accuracy`: Answer accuracy (numeric answers extracted from `<answer_begin>...<answer_end>` tags)

### Batch Processing with Scripts

We provide shell scripts for batch processing multiple tasks and models.

#### Generation Script

Edit `scripts/generate.sh` to configure:

```bash
# 1. Set your workspace directory
export WORKSPACE_DIR="/path/to/your/workspace"  # TODO: Set your workspace directory

# 2. Set your API key (optional for local models)
# export API_KEY="your-api-key"

# 3. Configure GPU (for local models)
export CUDA_VISIBLE_DEVICES=0

# 4. Configure models and subtasks in the script
MODEL_PATHS=(
    "Qwen/Qwen3-VL-8B-Instruct"
    "Qwen/Qwen3-VL-2B-Instruct"
)

SUBTASKS=(
    "shortest_path_csv_vertex2"
    "shortest_path_map_and_tab_with_constraint_1_2_3_4"
    "shortest_path_map_and_tab_no_constraint"
    "shortest_path_only_map"
    "shortest_path_only_tab"
)
```

Run the script:
```bash
bash scripts/generate.sh
```

#### Evaluation Script

```bash
# Evaluate all results
# Edit BASE_DIRECTORY in the script first
bash scripts/evaluate.sh
```

## Project Structure

```
MapTab/
├── src/
│   ├── generate.py              # Main generation script
│   ├── evaluate_planning.py     # Planning task evaluation
│   ├── evaluate_qa.py           # QA task evaluation
│   ├── metromap_utils.py        # MetroMap data utilities
│   ├── travelmap_utils.py       # TravelMap data utilities
│   └── generate_lib/            # Model API wrappers
│       ├── azure_gpt.py         # Azure OpenAI API
│       ├── qwen_api.py          # Qwen/DashScope API
│       ├── qwen_thinking_api.py # Qwen Thinking API
│       ├── doubao.py            # Doubao/Volcengine API
│       ├── gemini_yunwu.py      # Gemini via Yunwu proxy
│       ├── vllm_LLMengine.py    # vLLM local inference
│       ├── utils.py             # Utility functions
│       └── example/             # Example code
├── scripts/
│   ├── generate.sh              # Batch generation script
│   └── evaluate.sh              # Batch evaluation script
├── metromap/                    # MetroMap dataset
│   ├── data/
│   │   └── test_set/            # Test dataset for planning tasks
│   ├── images/
│   │   └── america/             # Map images
│   ├── prompts/                 # Prompt templates
│   └── tabulars/
│       └── america/             # Tabular data
├── travelmap/                   # TravelMap dataset
│   ├── data/
│   │   └── test_set/            # Test dataset for planning tasks
│   ├── images/
│   │   └── cambodia/            # Map images
│   ├── prompts/                 # Prompt templates
│   └── tabulars/
│       └── cambodia/            # Tabular data
├── results/                     # Generation results
│   └── response_generate/       # Model responses
└── results_evaluate/            # Evaluation results
```

## Configuration

### Environment Variables

| Variable | Description | Required For |
|----------|-------------|--------------|
| `WORKSPACE_DIR` | Project root directory | All operations |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | Azure GPT models |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint | Azure GPT models |
| `AZURE_API_VERSION` | Azure API version | Azure GPT models |
| `DASHSCOPE_API_KEY` | DashScope API key | Qwen API models |
| `ARK_API_KEY` | Volcengine ARK API key | Doubao models |
| `YUNWU_TOKEN` | Yunwu proxy token | Gemini models |

### Model Configuration

Edit the shell scripts in `scripts/` to configure:
- Model paths and names
- Subtasks to run
- Output directories
- GPU allocation (`CUDA_VISIBLE_DEVICES`)

## Evaluation Metrics

### Planning Task Metrics

| Metric | Description |
|--------|-------------|
| **all_acc** | Exact match - entire route must be correct |
| **part_acc** | Partial accuracy - proportion of correctly identified stops/edges |
| **Difficulty_score** | Weighted score: `map_difficulty × query_difficulty` (1-9 scale) |

### QA Task Metrics

| Metric | Description |
|--------|-------------|
| **accuracy** | Proportion of correct numeric answers |

## Citation

If you use MapTab in your research, please cite:

```bibtex
@article{maptab2024,
  title={MapTab: A Benchmark for Evaluating Vision-Language Models on Map Understanding},
  author={Your Name et al.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open a GitHub issue or contact us at [your-email@example.com].

## Troubleshooting

### Common Issues

**1. `FileNotFoundError: Data file not found`**
- Ensure `WORKSPACE_DIR` environment variable is set correctly
- Verify that the data files exist in the expected directories
- For QA tasks: Note that QA data files will be released separately

**2. `ValueError: Model {model_name} not supported`**
- Check if your model name matches exactly with supported models list
- For local models, use the full HuggingFace path (e.g., `Qwen/Qwen3-VL-8B-Instruct`)
- For API models, ensure the model name is correct (e.g., `gpt-4o`, not `gpt-4-o`)

**3. API Key Issues**
- Verify environment variables are set correctly:
  - Azure: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_API_VERSION`
  - Qwen: `DASHSCOPE_API_KEY`
  - Doubao: `ARK_API_KEY`
  - Gemini: `YUNWU_TOKEN`

**4. vLLM Installation for Local Models**
- Install vLLM: `pip install vllm`
- Ensure CUDA is available for GPU acceleration
- Check GPU memory requirements for your chosen model

---

**Note**: This is the evaluation code release for MapTab benchmark. 

**Current Release Status:**
- ✅ Complete route planning task evaluation pipeline
- ✅ Planning task dataset (metromap & travelmap)
- ✅ Evaluation metrics implementation
- ✅ Support for multiple VLM models (API and local)
- ⏳ QA task dataset (to be released)

Please check our project page for updates on the QA dataset release and future enhancements.
