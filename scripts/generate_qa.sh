#!/bin/bash

export WORKSPACE_DIR="YOUR WORKSPACE DIRECTORY"
# export API_KEY="YOUR API KEY"
export CUDA_VISIBLE_DEVICES=0

SUBTASKS=(
    "1_qa_only_pic_global"
    "2_qa_only_pic_part"
    "3_qa_only_pic_spatial_judge"
    "4_qa_edge_tab_global"
    "5_qa_edge_tab_part"
    "6_qa_edge_tab_spatial_judge"
    "7_qa_vertex_tab_global"
    "8_qa_vertex_tab_part"
    "9_qa_vertex_tab_spatial_judge"
    "10_qa_pic_and_tab_global"
    "11_qa_pic_and_tab_part"
    "12_qa_pic_and_tab_spatial_judge"
)

MODEL_PATHS=(
    "Qwen/Qwen3-VL-8B-Instruct"
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
                --model_path "$MODEL_PATH"
        done
    done
done

echo "All tasks completed."
