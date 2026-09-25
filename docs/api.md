# Public Interfaces

MapTab deliberately supports two inference backends and four model routes.

## Supported models

| Provider | Model | Notes |
| --- | --- | --- |
| `openlux` | `gemini-3-flash` | Hosted OpenLux route |
| `openlux` | `gemini-3.5-flash` | Hosted OpenLux route and CLI default |
| `vllm` | `Qwen/Qwen3.5-9B` | Local; forced non-thinking mode |
| `vllm` | `Qwen/Qwen3-VL-8B-Instruct` | Local Instruct checkpoint |

No generic OpenAI-compatible provider, arbitrary hosted endpoint, or other
model route is exposed. OpenLux uses the fixed
`https://api.openlux.ai/v1` endpoint and reads `OPENLUX_API_KEY`.

## CLI

| Command | Purpose |
| --- | --- |
| `maptab-infer list` | List QA, planning, and probe tasks |
| `maptab-infer download` | Download the test-only or full Hugging Face package |
| `maptab-infer validate` | Resolve records, prompts, images, and tables |
| `maptab-infer inspect` | Render one exact model input without inference |
| `maptab-infer generate` | Run one task or task selector |
| `maptab-infer evaluate` | Evaluate one result file |
| `maptab-infer evaluate-dir` | Evaluate a result directory |

Download the inference package:

```bash
maptab-infer download \
  --repo-id szq-nju/MapTab \
  --revision main \
  --subset test \
  --data-root ./data
```

Use `--subset full` to include training shards and raw training records.

OpenLux generation:

```bash
export OPENLUX_API_KEY=your_key_here

maptab-infer generate \
  --data-root ./data \
  --domain metromap \
  --task all-qa \
  --provider openlux \
  --model gemini-3.5-flash
```

Local vLLM generation:

```bash
maptab-infer generate \
  --data-root ./data \
  --domain travelmap \
  --task all-planning \
  --split test \
  --provider vllm \
  --model Qwen/Qwen3.5-9B \
  --tensor-parallel-size 4 \
  --max-model-len 128000 \
  --gpu-memory-utilization 0.9
```

The vLLM adapter passes
`chat_template_kwargs={"enable_thinking": False}`. This makes the
Qwen3.5 route non-thinking; the Qwen3-VL route is the Instruct checkpoint.

## Bash environment

`generate_qa.sh` and `generate_rp.sh` use:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MAPTAB_DATA_ROOT` | required | Downloaded `szq-nju/MapTab` snapshot |
| `PROVIDER` | `openlux` | `openlux` or `vllm` |
| `MODEL_PATH` | `gemini-3.5-flash` | Supported route or matching local path |
| `DOMAIN` | `all` | `all`, `metromap`, or `travelmap` |
| `QA_TASKS` | `all-qa` | QA task or selector |
| `RP_TASKS` | `all-planning` | Planning task or selector |
| `SPLIT` | `test` | Planning split |
| `OUTPUT_DIR` | `results/response_generate` | Output directory |
| `TEMPERATURE` | `0` | Sampling temperature |
| `MAX_TOKENS` | `2048` | Maximum output tokens |
| `OFFSET` / `LIMIT` | unset | Optional bounded slice |
| `OVERWRITE` | `0` | Restart output when set to `1` |
| `RETRY_ERRORS` | `0` | Retry failed rows when set to `1` |
| `CONTINUE_ON_ERROR` | `0` | Retain row errors when set to `1` |
| `TENSOR_PARALLEL_SIZE` | `1` | vLLM tensor parallelism |
| `MAX_MODEL_LEN` | `128000` | vLLM maximum context |
| `GPU_MEMORY_UTILIZATION` | `0.9` | vLLM memory target |

## Python API

The stable components are:

- `maptab_infer.providers.OpenLuxProvider`
- `maptab_infer.providers.VLLMProvider`
- `maptab_infer.providers.validate_model`
- `maptab_infer.tasks.get_task` and `list_tasks`
- `maptab_infer.data.load_records` and `build_content`
- `maptab_infer.runner.run_generation`
- `maptab_infer.evaluate.evaluate_file`

OpenLux example:

```python
import os

from maptab_infer.providers import OpenLuxProvider
from maptab_infer.runner import run_generation
from maptab_infer.tasks import get_task

model = "gemini-3.5-flash"
provider = OpenLuxProvider(
    model=model,
    api_key=os.environ["OPENLUX_API_KEY"],
    temperature=0.0,
    max_tokens=2048,
    seed=42,
    max_pixels=10_000_000,
    max_retries=3,
    retry_backoff=2.0,
    timeout=120.0,
)

output = run_generation(
    provider=provider,
    model=model,
    data_root="./data",
    output_dir="results/response_generate",
    domain="metromap",
    spec=get_task("1_qa_only_pic_global"),
    split="test",
)
print(output)
```

Generation writes one JSON list per domain, task, and model. Every record keeps
the original example and adds provenance, `raw_response`, normalized
`response`, optional `reasoning_content`, and `error`.
