#!/bin/bash

export WORKSPACE_DIR="/path/to/your/workspace"  # TODO: Set your workspace directory
# export API_KEY="your-api-key"  # TODO: Set your API key
export CUDA_VISIBLE_DEVICES=0
SUBTASKS=(
    "shortest_path_csv_vertex2"
    "shortest_path_map_and_tab_with_constraint_1_2_3_4"
    "shortest_path_map_and_tab_no_constraint"
    "shortest_path_only_map"
    "shortest_path_only_tab"
)

MODEL_PATHS=(
    "Qwen/Qwen3-VL-8B-Instruct"
    "Qwen/Qwen3-VL-2B-Instruct"
)

TASKS=(
    "metromap"
    "travelmap"
)

RESULTS_DIR="$WORKSPACE_DIR/results/response_generate"
mkdir -p "$RESULTS_DIR"

echo "Starting evaluation..."

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_PATH")
    echo "=============================="
    echo "Running with model: $MODEL_PATH"
    echo "=============================="

    for TASK in "${TASKS[@]}"; do
        echo "--- Task: $TASK ---"

        for SUBTASK in "${SUBTASKS[@]}"; do
            OUTPUT_FILE="${RESULTS_DIR}/${TASK}_${SUBTASK}_${MODEL_NAME}_results.json"

            # 判断当前组合是否在 SKIP_FILES 中
            BASENAME=$(basename "$OUTPUT_FILE")
            SKIP=false
            for f in "${SKIP_FILES[@]}"; do
                if [[ "$BASENAME" == "$f" ]]; then
                    SKIP=true
                    break
                fi
            done

            if [ "$SKIP" = true ]; then
                echo "Skipping $TASK + $SUBTASK + $MODEL_NAME (already exists)"
                continue
            fi

            echo "Running subtask: $SUBTASK"
            python src/generate.py \
                --task "$TASK" \
                --subtask "$SUBTASK" \
                --model_path "$MODEL_PATH" \
                --api_key "$API_KEY"
        done
    done
done

echo "All tasks completed."