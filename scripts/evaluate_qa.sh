#!/bin/bash

# 前半部分路径
export CUDA_VISIBLE_DEVICES=0
BASE_DIRECTORY="/path/to/your/workspace/results"

# 后半部分文件夹名称数组
FOLDERS=(
    "response_generate_qa_opensource_1_qwen3_8b"
)

# 遍历每个文件夹
for FOLDER in "${FOLDERS[@]}"; do
    # 拼接完整的文件夹路径
    DIRECTORY="${BASE_DIRECTORY}${FOLDER}"

    # 获取当前文件夹下所有的 .json 文件，并按名称顺序排序
    INPUT_FILES=$(find "$DIRECTORY" -type f -name "*.json" | sort)

    if [ -z "$INPUT_FILES" ]; then
        echo "No .json files found in folder: $DIRECTORY"
        continue
    fi

    # 运行评估
    echo "Starting evaluations for directory: $DIRECTORY"
    for INPUT_FILE in $INPUT_FILES; do
        echo "Processing file: $INPUT_FILE"
        python src/evaluate_qa.py --input_file "$INPUT_FILE"
    done
    echo "All evaluations completed for directory: $DIRECTORY"
done

echo "All evaluations completed."
