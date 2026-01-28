import base64
import os
import csv
import json as json_module
import pandas as pd
import time
import random
import numpy as np
from openai import OpenAI
from tqdm import tqdm


def set_random_seed(seed):
    """设置所有随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_client_model(model_path, api_key):
    """初始化 Yunwu 兼容的 OpenAI 客户端。"""
    assert api_key is not None, "API key is required for using Yunwu GPT"
    assert model_path is not None, "Model name is required for using Yunwu GPT"

    base_url = os.environ.get("YUNWU_BASE_URL", "https://yunwu.ai/v1/")
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model_path


def generate_response(queries, model, client=None, media_type="image/jpeg", api_key=None, random_baseline=False, task=None, subtask=None, seed=42):
    """与 azure_gpt 保持一致的多模态生成逻辑。
    
    Args:
        queries: 查询列表或字典
        model: 模型名称
        client: OpenAI客户端（可选）
        media_type: 图片媒体类型
        api_key: API密钥（可选）
        random_baseline: 是否使用随机基线
        task: 任务名称（可选）
        subtask: 子任务名称（可选）
        seed: 随机种子（默认为42）
    
    Returns:
        responses: 响应列表
    """
    # 设置随机种子确保可复现性
    set_random_seed(seed)
    print(f"Random seed set to: {seed}")
    
    if client is None and api_key is not None:
        client, _ = get_client_model(model, api_key)

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def identify_table_type(file_path):
        file_path_lower = file_path.lower()
        if 'edge' in file_path_lower:
            return 'edge'
        if 'vertex' in file_path_lower:
            return 'vertex'
        return 'edge'

    def process_csv_file(file_path, del_cols):
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            if del_cols:
                df = df.drop(columns=[c for c in del_cols if c in df.columns], errors='ignore')
            # 转换为 CSV 格式字符串（与 vllm_LLMengine.py 保持一致）
            return df.to_csv(index=False)
        except Exception as e:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                if del_cols:
                    filtered_rows = []
                    for row in rows:
                        filtered_rows.append({k: v for k, v in row.items() if k not in del_cols})
                    rows = filtered_rows
                return json_module.dumps(rows, ensure_ascii=False, indent=2)
            except Exception as e2:
                raise Exception(f"Error processing CSV file: {e2}") from e

    def process_json_file(file_path, del_cols):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json_module.load(f)
            if isinstance(data, list):
                filtered = []
                for item in data:
                    if isinstance(item, dict):
                        filtered.append({k: v for k, v in item.items() if k not in del_cols})
                    else:
                        filtered.append(item)
                data = filtered
            elif isinstance(data, dict):
                data = {k: v for k, v in data.items() if k not in del_cols}
            return json_module.dumps(data, ensure_ascii=False, indent=2)
        except Exception as e:
            raise Exception(f"Error processing JSON file: {e}")

    if isinstance(queries, dict):
        iterator = queries.keys()
    elif isinstance(queries, list):
        iterator = range(len(queries))
    else:
        raise ValueError("Queries must be a dict or list")

    # 初始化删除列的列表
    del_cols_edge = []
    del_cols_vertex = []
    
    if task == "metromap":
        if subtask == "shortest_path_only_tab":
            del_cols_edge = ["Time", "Price", "Comfort Level", "Reliability"]
            del_cols_vertex = []
        elif subtask == "shortest_path_only_map":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "shortest_path_map_and_tab_no_constraint":
            del_cols_edge = ["Time", "Price", "Comfort Level", "Reliability"]
            del_cols_vertex = []
        elif subtask == "demo":
            del_cols_edge = ["Time", "Price", "Comfort Level", "Reliability"]
            del_cols_vertex = []
        elif subtask == "shortest_path_map_and_tab_with_constraint_1":
            del_cols_edge = ["Price", "Comfort Level", "Reliability"]
            del_cols_vertex = ["Price", "Comfort Level", "Reliability"]
        elif subtask == "shortest_path_map_and_tab_with_constraint_2":
            del_cols_edge = ["Time", "Comfort Level", "Reliability"]
            del_cols_vertex = ["Time", "Comfort Level", "Reliability", "Transfer Time"]
        elif subtask == "shortest_path_map_and_tab_with_constraint_3":
            del_cols_edge = ["Time", "Price", "Reliability"]
            del_cols_vertex = ["Time", "Price", "Reliability", "Transfer Time"]
        elif subtask == "shortest_path_map_and_tab_with_constraint_4":
            del_cols_edge = ["Time", "Price", "Comfort Level"]
            del_cols_vertex = ["Time", "Price", "Comfort Level", "Transfer Time"]
        elif subtask == "shortest_path_map_and_tab_with_constraint_1_2_3_4":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "shortest_path_map_and_tab_with_constraint_1_2_4":
            del_cols_edge = ["Comfort Level"]
            del_cols_vertex = ["Comfort Level"]
        elif subtask == "shortest_path_map_and_tab_with_constraint_1_3_4":
            del_cols_edge = ["Price"]
            del_cols_vertex = ["Price"]
        elif subtask == "shortest_path_map_and_tab_with_constraint_2_3_4":
            del_cols_edge = ["Time"]
            del_cols_vertex = ["Time", "Transfer Time"]
        elif subtask == "only_vertex2":
            del_cols_edge = []
            del_cols_vertex = ["Line"]
        # CSV消融实验
        elif subtask == "4_csv_edge_global":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "5_csv_edge_part":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "6_csv_edge_spatial_judge":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "7_csv_vertex_global":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "8_csv_vertex_part":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "9_csv_vertex_spatial_judge":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "10_csv_and_pic_global":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "11_csv_and_pic_part":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "12_csv_and_pic_spatial_judge":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "shortest_path_csv_vertex2":
            del_cols_edge = []
            del_cols_vertex = ["Line"]
        elif subtask == "shortest_path_map_and_tab_csv_constraint_1_2_3_4":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "shortest_path_only_csv":
            del_cols_edge = ["Time", "Price", "Comfort Level", "Reliability"]
            del_cols_vertex = []
        elif subtask == "shortest_path_map_and_csv":
            del_cols_edge = ["Time", "Price", "Comfort Level", "Reliability"]
            del_cols_vertex = []
        elif subtask == "1_qa_only_pic_global":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "2_qa_only_pic_part":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "3_qa_only_pic_spatial_judge":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "4_qa_edge_tab_global":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "5_qa_edge_tab_part":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "6_qa_edge_tab_spatial_judge":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "7_qa_vertex_tab_global":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "8_qa_vertex_tab_part":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "9_qa_vertex_tab_spatial_judge":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "10_qa_pic_and_tab_global":
            del_cols_edge = []
            del_cols_vertex = ["Line"]
        elif subtask == "11_qa_pic_and_tab_part":
            del_cols_edge = []
            del_cols_vertex = ["Line"]
        elif subtask == "12_qa_pic_and_tab_spatial_judge":
            del_cols_edge = []
            del_cols_vertex = ["Line"]
        else:
            raise ValueError(f"Unknown subtask '{subtask}' for task 'metromap'")

    elif task == "travelmap":
        if subtask == "shortest_path_only_tab":
            del_cols_edge = ["Time", "Price", "Comfort Level", "Reliability"]
            del_cols_vertex = []
        elif subtask == "shortest_path_only_map":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "shortest_path_map_and_tab_no_constraint":
            del_cols_edge = ["Time", "Price", "Comfort Level", "Reliability"]
            del_cols_vertex = []
        elif subtask == "demo":
            del_cols_edge = ["Time", "Price", "Comfort Level", "Reliability"]
            del_cols_vertex = []
        elif subtask == "shortest_path_map_and_tab_with_constraint_1":
            del_cols_edge = ["Price", "Comfort Level", "Reliability"]
            del_cols_vertex = ["Price", "Comfort Level", "Reliability"]
        elif subtask == "shortest_path_map_and_tab_with_constraint_2":
            del_cols_edge = ["Comfort Level", "Reliability"]
            del_cols_vertex = ["Comfort Level", "Reliability"]
        elif subtask == "shortest_path_map_and_tab_with_constraint_3":
            del_cols_edge = ["Time", "Price", "Reliability"]
            del_cols_vertex = ["Time", "Price", "Reliability"]
        elif subtask == "shortest_path_map_and_tab_with_constraint_4":
            del_cols_edge = ["Time", "Price", "Comfort Level"]
            del_cols_vertex = ["Time", "Price", "Comfort Level"]
        elif subtask == "shortest_path_map_and_tab_with_constraint_1_2_3_4":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "shortest_path_map_and_tab_with_constraint_1_2_4":
            del_cols_edge = ["Comfort Level"]
            del_cols_vertex = ["Comfort Level"]
        elif subtask == "shortest_path_map_and_tab_with_constraint_1_3_4":
            del_cols_edge = ["Price"]
            del_cols_vertex = ["Price"]
        elif subtask == "shortest_path_map_and_tab_with_constraint_2_3_4":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "only_vertex2":
            del_cols_edge = []
            del_cols_vertex = []
        # CSV消融实验
        elif subtask == "4_csv_edge_global":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "5_csv_edge_part":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "6_csv_edge_spatial_judge":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "7_csv_vertex_global":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "8_csv_vertex_part":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "9_csv_vertex_spatial_judge":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "10_csv_and_pic_global":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "11_csv_and_pic_part":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "12_csv_and_pic_spatial_judge":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "shortest_path_csv_vertex2":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "shortest_path_map_and_tab_csv_constraint_1_2_3_4":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "shortest_path_only_csv":
            del_cols_edge = ["Time", "Price", "Comfort Level", "Reliability"]
            del_cols_vertex = []
        elif subtask == "shortest_path_map_and_csv":
            del_cols_edge = ["Time", "Price", "Comfort Level", "Reliability"]
            del_cols_vertex = []
        elif subtask == "1_qa_only_pic_global":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "2_qa_only_pic_part":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "3_qa_only_pic_spatial_judge":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "4_qa_edge_tab_global":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "5_qa_edge_tab_part":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "6_qa_edge_tab_spatial_judge":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "7_qa_vertex_tab_global":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "8_qa_vertex_tab_part":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "9_qa_vertex_tab_spatial_judge":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "10_qa_pic_and_tab_global":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "11_qa_pic_and_tab_part":
            del_cols_edge = []
            del_cols_vertex = []
        elif subtask == "12_qa_pic_and_tab_spatial_judge":
            del_cols_edge = []
            del_cols_vertex = []
        else:
            raise ValueError(f"Unknown subtask '{subtask}' for task 'travelmap'")

    responses = []
    for k in tqdm(iterator):
        item = queries[k]

        system_prompt = None
        user_content = []

        if isinstance(item, list):
            for type_, content in item:
                if type_ == 'system':
                    system_prompt = content
                elif type_ == 'text':
                    user_content.append({"type": "text", "text": content})
                elif type_ == 'image':
                    try:
                        base64_image = encode_image(content)
                        user_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{base64_image}"}
                        })
                    except Exception as e:
                        print(f"Error loading image {content}: {e}")
                        user_content.append({"type": "text", "text": f"[Error loading image: {content}]"})
                elif type_ == 'csv':
                    try:
                        # 识别是 edge_tab 还是 vertex_tab
                        table_type = identify_table_type(content)
                        
                        # 根据 table_type 选择要删除的列
                        if table_type == 'edge':
                            del_cols = del_cols_edge
                        elif table_type == 'vertex':
                            del_cols = del_cols_vertex
                        else:
                            del_cols = []
                        
                        # 处理 CSV 文件
                        file_content = process_csv_file(content, del_cols)
                        table_label = "Edge Table (CSV)" if table_type == 'edge' else "Vertex Table (CSV)"
                        user_content.append({"type": "text", "text": f"{table_label}:\n{file_content}"})
                    except Exception as e:
                        print(f"Error loading CSV file {content}: {e}")
                        user_content.append({"type": "text", "text": f"[Error loading CSV file: {content}]"})
                elif type_ == 'json':
                    try:
                        # 识别是 edge_tab 还是 vertex_tab
                        table_type = identify_table_type(content)
                        
                        # 根据 table_type 选择要删除的列
                        if table_type == 'edge':
                            del_cols = del_cols_edge
                        elif table_type == 'vertex':
                            del_cols = del_cols_vertex
                        else:
                            del_cols = []
                        
                        # 处理 JSON 文件
                        file_content = process_json_file(content, del_cols)
                        table_label = "Edge Table (JSON)" if table_type == 'edge' else "Vertex Table (JSON)"
                        user_content.append({"type": "text", "text": f"{table_label}:\n{file_content}"})
                    except Exception as e:
                        print(f"Error loading JSON file {content}: {e}")
                        user_content.append({"type": "text", "text": f"[Error loading JSON file: {content}]"})
        else:
            print(f"Warning: Item {k} is not a list, skipping. Type: {type(item)}")
            responses.append("Error: Invalid item format")
            continue

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if user_content:
            messages.append({"role": "user", "content": user_content})

        max_retries = 10
        sleep_time = 1
        result = None
        curr_retries = 0

        while curr_retries < max_retries and result is None:
            try:
                response = client.chat.completions.create(
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.0,
                    top_p=1.0,
                    model=model,
                    seed=seed,
                )
                result = response.choices[0].message.content
            except Exception as e:
                print(f"Error: {e}")
                print(f"Error {curr_retries}, sleeping for {sleep_time} seconds...")
                time.sleep(sleep_time)
                curr_retries += 1
                sleep_time *= 1.6

        if result is None:
            result = "Error in generating response."
            print(f"Error in generating response for {k}")

        responses.append(result)

    return responses


if __name__ == "__main__":
    """轻量自测：本地图片 + 文本，多模态调用 Yunwu GPT。"""
    model_name = os.environ.get("YUNWU_MODEL", "gemini-3-flash-preview")
    api_key = os.environ.get("YUNWU_TOKEN")
    if not api_key:
        raise SystemExit("请设置环境变量 YUNWU_TOKEN 后再运行测试")

    script_dir = os.path.abspath(os.path.dirname(__file__))
    sample_image = os.path.join(script_dir, "example", "dog.png")
    if not os.path.exists(sample_image):
        raise SystemExit(f"示例图片不存在: {sample_image}")

    queries = [
        [
            ("system", "You are a helpful assistant."),
            ("text", "这张图片里有什么？请详细描述。"),
            ("image", sample_image),
        ]
    ]

    client, _ = get_client_model(model_name, api_key)
    responses = generate_response(queries, model_name, client=client)
    print("\n--- Test Response ---")
    for resp in responses:
        print(resp)