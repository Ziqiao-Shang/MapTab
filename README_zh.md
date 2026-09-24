# MapTab：完整推理与评测实现

[English](README.md)

**官方资源：**[项目主页](https://ziqiao-shang.github.io/MapTab-Leaderboard/) · [arXiv 论文](https://arxiv.org/abs/2602.18600) · [Hugging Face 数据集](https://huggingface.co/datasets/szq-nju/MapTab) · [官方代码](https://github.com/Ziqiao-Shang/MapTab)

本仓库是 **MapTab: A Diagnostic Benchmark for Long-Horizon Multi-Criteria Multimodal Reasoning on Heterogeneous Topological Graphs** 的纯推理、可复现实现。代码按照 `MapTab-main` 的完整查询构造与评分方法重建，覆盖 MetroMap 和 TravelMap、全部 QA 与路径规划分支、CSV 消融、已发布的 vertex2 任务，以及 QA 引导的规划任务。

本仓库不提交原始数据集和模型权重。请从 [MapTab 官方 Hugging Face 数据集](https://huggingface.co/datasets/szq-nju/MapTab)下载数据，并把数据根目录传给命令行。

![MapTab 概览](assets/fig_1.png)

## 目录

- [项目概览](#项目概览)
- [与 MapTab-main 的方法一致性](#与-maptab-main-的方法一致性)
- [完整任务清单](#完整任务清单)
- [数据目录结构](#数据目录结构)
- [安装](#安装)
- [快速开始](#快速开始)
- [推理后端](#推理后端)
- [评测](#评测)
- [输出与断点续跑](#输出与断点续跑)
- [项目结构](#项目结构)
- [复现检查](#复现检查)
- [引用](#引用)

## 项目概览

MapTab 用于评估视觉语言模型能否联合地图、JSON/CSV 表格和自然语言约束，完成两类任务：

1. **地图问答：**在图像、边表、点表或图表联合输入上完成全局计数、局部计数和空间判断。
2. **路径规划：**在时间、价格、舒适度和可靠性约束下完成最短路径与多指标加权规划。

统一任务注册表提供 52 个推理入口：

| 任务族 | 数量 | 包含设置 |
| --- | ---: | --- |
| 可评分 QA | 22 | 原始 12 个 QA、已发布任务 13、9 个 CSV 变体 |
| 可评分 Planning | 24 | 16 个标准/消融变体、8 个 QA 引导变体 |
| 序列化探针 | 6 | JSON/CSV 的边表、点表和 vertex2 纯表格探针 |
| **合计** | **52** | 源查询构造器中的全部分支 |

运行 `maptab-infer list --json` 可查看机器可读的完整注册表。

## 与 MapTab-main 的方法一致性

本实现以 `MapTab-main/MapTab-main` 的完整流程为依据，重点逐项对照了 `src/generate.py`、`src/metromap_utils.py`、`src/travelmap_utils.py`、`src/evaluate_qa.py` 和 `src/evaluate_planning.py`。

以下基准行为保持一致：

- 每个领域和子任务选择相同的源 JSON 文件。
- 代码中打包了 MetroMap 与 TravelMap 的完整提示模板。
- 模型输入顺序遵循源构造器：格式化任务提示、表格说明与序列化表格，最后在需要时加入地图图像。
- JSON 和 CSV 消融使用相同的底层样本。
- 约束任务按照 MetroMap 与 TravelMap 各自的规则裁剪边表/点表字段。
- QA 仅从 `<answer_begin>...<answer_end>` 中取答案，并四舍五入到两位小数后比较。
- Planning 按连字符拆分站点，使用阈值 0.5 的 SequenceMatcher 站名相似度；换乘站严格匹配，同时计算前缀部分准确率和原始难度分数。
- QA 引导的 Planning 保留 `<route_begin>...<route_end>`，并解析总时间、总价格、平均舒适度和平均可靠性四个字段。

在不改变基准定义的前提下，本仓库还补齐了源实现中的工程问题：

- 用明确的 `--data-root` 替代机器相关的 `WORKSPACE_DIR` 路径拼接。
- 用声明式任务注册表替代两份冗长、重复的条件分支。
- OpenAI-compatible 托管接口与进程内 vLLM 共用同一套输入构造逻辑。
- 生成过程支持稳定样本 ID、原子写入、断点续跑、错误保留和失败重试。
- 注册已发布的 QA 任务 13；发布数据没有单独的任务 13 提示，因此复用对应的点表全局问题提示。
- 部分 TravelMap 任务 13 记录引用未发布的 `*_vertex2.json/.csv`；加载器保留原始记录，并解析到已验证匹配的 `*_vertex.json/.csv`。
- 为 8 个 QA 引导规划分支补齐约束字段路由，使其均可实际运行。
- QA 引导输出会先规范化再做路径评测；混合结果目录按记录中的任务族评测，不依赖文件名猜测。

## 完整任务清单

### QA 任务

13 个直接 QA 任务如下：

| 输入 | 全局 | 局部 | 空间判断 |
| --- | --- | --- | --- |
| 地图图像 | `1_qa_only_pic_global` | `2_qa_only_pic_part` | `3_qa_only_pic_spatial_judge` |
| 边表 JSON | `4_qa_edge_tab_global` | `5_qa_edge_tab_part` | `6_qa_edge_tab_spatial_judge` |
| 点表 JSON | `7_qa_vertex_tab_global` | `8_qa_vertex_tab_part` | `9_qa_vertex_tab_spatial_judge` |
| 地图 + 点表 JSON | `10_qa_pic_and_tab_global` | `11_qa_pic_and_tab_part` | `12_qa_pic_and_tab_spatial_judge` |
| 已发布 vertex2 JSON | `13_qa_vertex2_tab_global` | — | — |

9 个可评分 CSV 对应任务如下：

- `4_csv_edge_global`、`5_csv_edge_part`、`6_csv_edge_spatial_judge`
- `7_csv_vertex_global`、`8_csv_vertex_part`、`9_csv_vertex_spatial_judge`
- `10_csv_and_pic_global`、`11_csv_and_pic_part`、`12_csv_and_pic_spatial_judge`

六个与源实现兼容的序列化探针是 `j_e_tab`、`j_v_tab`、`j_v2_tab`、`c_e_tab`、`c_v_tab` 和 `c_v2_tab`。它们只发送指定表格，不计算基准分数。使用 `all-probes` 单独运行；`all-qa` 只包含可评分 QA。

### 路径规划任务

16 个标准与消融变体如下：

| 分组 | 任务名 |
| --- | --- |
| 单一/联合输入 | `shortest_path_only_map`、`shortest_path_only_tab`、`shortest_path_only_csv`、`shortest_path_map_and_tab_no_constraint`、`shortest_path_map_and_csv` |
| 单约束 | `shortest_path_map_and_tab_with_constraint_1`、`shortest_path_map_and_tab_with_constraint_2`、`shortest_path_map_and_tab_with_constraint_3`、`shortest_path_map_and_tab_with_constraint_4` |
| 多约束 | `shortest_path_map_and_tab_with_constraint_1_2_3_4`、`shortest_path_map_and_tab_with_constraint_1_2_4`、`shortest_path_map_and_tab_with_constraint_1_3_4`、`shortest_path_map_and_tab_with_constraint_2_3_4` |
| Vertex2 与 CSV | `only_vertex2`、`shortest_path_csv_vertex2`、`shortest_path_map_and_tab_csv_constraint_1_2_3_4` |

约束编号沿用上游定义：1 = 时间、2 = 价格、3 = 舒适度、4 = 可靠性。

8 个 QA 引导的规划变体是：

- `shortest_path_with_qa_and_constraint_1`
- `shortest_path_with_qa_and_constraint_2`
- `shortest_path_with_qa_and_constraint_3`
- `shortest_path_with_qa_and_constraint_4`
- `shortest_path_with_qa_and_constraint_1_2_3_4`
- `shortest_path_with_qa_and_constraint_1_2_4`
- `shortest_path_with_qa_and_constraint_1_3_4`
- `shortest_path_with_qa_and_constraint_2_3_4`

`all-planning` 运行全部 24 个 Planning 入口；`canonical-planning` 排除 CSV 和 QA 引导消融。

## 数据目录结构

加载器同时兼容原始 MapTab 目录和规范化 Hugging Face 数据包。

原始目录：

```text
DATA_ROOT/
├── metromap/
│   ├── data/{training_set,test_set,all}/
│   ├── qa_data/
│   ├── images/
│   └── tabulars/
└── travelmap/
    ├── data/{training_set,test_set,all}/
    ├── qa_data/
    ├── images/
    └── tabulars/
```

规范化目录：

```text
DATA_ROOT/
├── raw/
│   ├── metromap/{data,qa_data,prompts}/
│   └── travelmap/{data,qa_data,prompts}/
└── assets/
    ├── metromap/{images,tabulars}/
    └── travelmap/{images,tabulars}/
```

CLI 会读取 `raw/` 下的原始查询，并在 `assets/` 下解析对应资源；不会静默重采样样本或改写答案。

## 安装

需要 Python 3.10 或更高版本。

```bash
git clone https://github.com/Ziqiao-Shang/MapTab.git
cd MapTab

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

使用进程内 vLLM 推理时，请在与 CUDA 环境兼容的环境中安装可选依赖：

```bash
python -m pip install -e ".[local]"
```

## 快速开始

### 1. 下载并校验数据集

```bash
maptab-infer download \
  --repo-id szq-nju/MapTab \
  --data-root ./data

maptab-infer validate \
  --data-root ./data \
  --domain all \
  --task all \
  --split test
```

校验命令会加载全部选定源文件、格式化必需提示，并解析每个引用的图像和表格。建议在任何付费 API 或 GPU 推理前先运行。

### 2. 查看模型实际输入

```bash
maptab-infer inspect \
  --data-root ./data \
  --domain metromap \
  --task shortest_path_map_and_tab_with_constraint_1_2_3_4 \
  --split test \
  --index 0
```

该命令打印有序的文本、表格和图像输入，不调用模型。

## 推理后端

### OpenAI-compatible API

```bash
export OPENAI_API_KEY=your_key_here

maptab-infer generate \
  --data-root ./data \
  --domain metromap \
  --task all-qa \
  --provider openai \
  --model qwen3-vl-plus \
  --base-url https://your-endpoint.example/v1 \
  --temperature 0 \
  --max-tokens 2048 \
  --output-dir results/response_generate
```

如果凭据保存在其他环境变量中，可使用 `--api-key-env NAME`。密钥不会从已提交的配置文件中读取。

### 进程内 vLLM

```bash
maptab-infer generate \
  --data-root ./data \
  --domain travelmap \
  --task all-planning \
  --split test \
  --provider vllm \
  --model /path/to/multimodal-model \
  --tensor-parallel-size 4 \
  --max-model-len 128000 \
  --gpu-memory-utilization 0.9 \
  --output-dir results/response_generate
```

### Bash 运行入口

所有 Bash 入口都复用同一份任务注册表和推理实现。先统一配置后端：

```bash
export MAPTAB_DATA_ROOT=/path/to/MapTab
export MODEL_PATH=qwen3-vl-plus
export PROVIDER=openai
export OPENAI_API_KEY=your_key_here
export OPENAI_BASE_URL=https://your-endpoint.example/v1
```

| 范围 | 脚本 | 行为 |
| --- | --- | --- |
| 通用 QA | `scripts/generate_qa.sh` | 运行 `DOMAIN` 和 `QA_TASKS`，默认 `all`/`all-qa` |
| 通用 Planning | `scripts/generate_rp.sh` | 运行 `DOMAIN`、`RP_TASKS` 和 `SPLIT`，默认 `all`/`all-planning`/`test` |
| MetroMap QA | `scripts/run_metromap_qa.sh` | 运行 MetroMap QA |
| TravelMap QA | `scripts/run_travelmap_qa.sh` | 运行 TravelMap QA |
| MetroMap Planning | `scripts/run_metromap_planning.sh` | 运行 MetroMap Planning |
| TravelMap Planning | `scripts/run_travelmap_planning.sh` | 运行 TravelMap Planning |
| 单个 QA | `scripts/run_qa_task.sh DOMAIN QA_TASK` | 在指定领域运行一个 QA 任务 |
| 单个 Planning | `scripts/run_planning_task.sh DOMAIN TASK [SPLIT]` | 在指定领域和 split 运行一个 Planning 任务 |
| 全部 QA | `scripts/run_all_qa.sh DATA_ROOT MODEL BASE_URL` | 位置参数形式运行两个领域 |
| 全部 Planning | `scripts/run_all_planning.sh DATA_ROOT MODEL BASE_URL [SPLIT]` | 位置参数形式运行两个领域 |
| 完整生成 | `scripts/run_all.sh` | 依次运行全部 QA 与 Planning |
| 评测 | `scripts/evaluate_qa.sh`、`scripts/evaluate_rp.sh`、`scripts/evaluate_all.sh` | 评测一个或两个任务族 |

常用调用：

```bash
bash scripts/run_metromap_qa.sh
bash scripts/run_travelmap_planning.sh
bash scripts/run_qa_task.sh metromap 10_qa_pic_and_tab_global
bash scripts/run_planning_task.sh travelmap shortest_path_only_map test
bash scripts/run_all.sh
bash scripts/evaluate_all.sh
```

通用脚本还支持 `OUTPUT_DIR`、`API_KEY_ENV`、`TEMPERATURE`、`MAX_TOKENS`、`MAX_PIXELS`、`MAX_RETRIES`、`RETRY_BACKOFF`、`TIMEOUT`、`SEED`、`OFFSET` 和 `LIMIT`。设置 `OVERWRITE=1`、`RETRY_ERRORS=1` 或 `CONTINUE_ON_ERROR=1` 可启用相应开关。使用 vLLM 时设置 `PROVIDER=vllm`，并可进一步设置 `TENSOR_PARALLEL_SIZE`、`MAX_MODEL_LEN` 和 `GPU_MEMORY_UTILIZATION`。

Python 兼容入口也接受原来的参数名：

```bash
PYTHONPATH=src python src/generate.py \
  --task metromap \
  --subtask shortest_path_only_map \
  --model_path qwen3-vl-plus \
  --provider openai \
  --base_url https://your-endpoint.example/v1 \
  --data_root ./data
```

集合选择器包括 `all`、`all-qa`、`all-planning`、`all-probes`、`canonical-qa` 和 `canonical-planning`。Planning 支持 `--split train|test|all`；QA 和探针固定使用已发布 QA 记录。

## 评测

自动识别任务族并评测单个文件：

```bash
maptab-infer evaluate \
  --family auto \
  --input results/response_generate/metromap_1_qa_only_pic_global_MODEL_results.json
```

评测混合结果目录：

```bash
maptab-infer evaluate-dir \
  --input-dir results/response_generate \
  --output-dir results_evaluate \
  --family auto
```

| 任务族 | 指标 |
| --- | --- |
| QA | 严格标签抽取并保留两位小数后的数值 `accuracy` |
| Planning | `all_acc`、前缀 `part_acc`、`difficulty_score_total` |

评测器会写入逐样本 `*.evaluated.json` 和机器可读的 `*.summary.json`。

## 输出与断点续跑

生成文件沿用源仓库兼容命名：

```text
{domain}_{subtask}_{model-name}_results.json
```

每条结果保留完整源记录，并增加稳定 ID、任务来源、模型、原始响应、规范化响应、推理内容和错误字段。

写入采用原子替换。默认跳过已有成功 ID；`--retry-errors` 只重跑失败项；`--overwrite` 重新开始当前输出文件；`--offset` 和 `--limit` 可用于分段运行。

## 项目结构

```text
MapTab/
├── assets/fig_1.png
├── scripts/                 # 下载、QA/Planning 推理与评测入口
├── src/
│   ├── generate.py          # 上游参数兼容入口
│   ├── evaluate_qa.py
│   ├── evaluate_planning.py
│   └── maptab_infer/        # CLI、数据、任务、provider、runner、评测与完整 prompts
├── tests/test_registry.py
├── README.md
├── README_zh.md
└── pyproject.toml
```

## 复现检查

以下检查均不调用模型 API：

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

测试覆盖任务注册完整性、源兼容文件名、分领域表格字段裁剪、TravelMap vertex2 回退、QA 引导输出解析，以及原始 QA/Planning 指标行为。

## 安全与发布范围

本仓库不包含 MapTab 原始数据、模型权重、历史推理响应、缓存或凭据。代码只从环境变量读取 API 凭据。正式发布结果时，建议保留数据集 revision、模型标识或 checkpoint hash、解码配置、提示文件与原始生成响应。

## 引用

如果本项目对你的研究有帮助，请引用：

```bibtex
@article{shang2026maptab,
  title={MapTab: A Diagnostic Benchmark for Long-Horizon Multi-Criteria Multimodal Reasoning on Heterogeneous Topological Graphs},
  author={Shang, Ziqiao and Ge, Lingyue and Xu, Zian and Cheng, Zi-Jian and Tian, Shi-Yu and Huang, Zhenyu and Fu, Wenbo and Wu, Weiming and Chen, Yang and Zhang, Xiangwen and Hu, Yulan and Liu, Bin and Guo, Lan-Zhe},
  journal={arXiv preprint arXiv:2602.18600},
  year={2026}
}
```

## 许可证

用于构建本推理发布版的源快照中没有独立许可证文件。公开再分发前应补充预期的代码许可证；数据集使用另行受 MapTab 官方数据集条款约束。
