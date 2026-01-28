import os
import base64
import csv
import json as json_module
import pandas as pd
import time
import random
import numpy as np
from volcenginesdkarkruntime import Ark
from tqdm import tqdm


def set_random_seed(seed):
    """设置所有随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_client_model(model_path, api_key):
    assert api_key is not None, "API key is required for using Doubao"
    assert model_path is not None, "Model name is required for using Doubao"
    client = Ark(
        base_url='https://ark.cn-beijing.volces.com/api/v3',
        api_key=api_key
    )
    model = model_path
    return client, model

# 工具函数：将图像编码为 base64
MAX_PIXELS=10000000
from PIL import Image
import io
def encode_file(path):
    with Image.open(path) as img:
        img=img.convert("RGB")
        w,h=img.size
        pixels=w*h
        if pixels>MAX_PIXELS:
            scale=(MAX_PIXELS/pixels)**0.5
            new_w=int(w*scale)
            new_h=int(h*scale)
            img=img.resize((new_w,new_h),Image.LANCZOS)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG",quality=95)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


def generate_response(queries, model, client=None, media_type="image/jpeg", api_key=None, random_baseline=False, task=None, subtask=None, seed=42):
    """与 gemini_yunwu 保持一致的多模态生成逻辑。
    
    Args:
        queries: 查询列表或字典
        model: 模型名称
        client: Ark客户端（可选）
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
    
    if client is None:
        if api_key is None:
            raise ValueError("Either client or api_key must be provided")
        client, _ = get_client_model(model, api_key)

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def identify_table_type(file_path):
        """根据文件路径判断是 edge_tab 还是 vertex_tab"""
        file_path_lower = file_path.lower()
        if 'edge' in file_path_lower:
            return 'edge'
        if 'vertex' in file_path_lower:
            return 'vertex'
        return 'edge'

    def process_csv_file(file_path, del_cols):
        """处理 CSV 文件，删除指定的列"""
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            if del_cols:
                df = df.drop(columns=[c for c in del_cols if c in df.columns], errors='ignore')
            # 转换为 CSV 格式字符串
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
        """处理 JSON 文件，删除指定的键"""
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

    # Determine if queries is a dict or list and get iterator
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
        
        # Initialize message components
        system_prompt = None
        user_content = []
        
        # Process the item (list of tuples)
        if isinstance(item, list):
            for type_, content in item:
                if type_ == 'system':
                    system_prompt = content
                elif type_ == 'text':
                    user_content.append({"type": "input_text", "text": content})
                elif type_ == 'image':
                    try:
                        base64_file = encode_file(content)
                        user_content.append({
                            "type": "input_image",
                            "image_url": f"data:{media_type};base64,{base64_file}"
                        })
                    except Exception as e:
                        print(f"Error loading image {content}: {e}")
                        user_content.append({"type": "input_text", "text": f"[Error loading image: {content}]"})
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
                        user_content.append({"type": "input_text", "text": f"{table_label}:\n{file_content}"})
                    except Exception as e:
                        print(f"Error loading CSV file {content}: {e}")
                        user_content.append({"type": "input_text", "text": f"[Error loading CSV file: {content}]"})
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
                        user_content.append({"type": "input_text", "text": f"{table_label}:\n{file_content}"})
                    except Exception as e:
                        print(f"Error loading JSON file {content}: {e}")
                        user_content.append({"type": "input_text", "text": f"[Error loading JSON file: {content}]"})
        else:
            print(f"Warning: Item {k} is not a list, skipping. Type: {type(item)}")
            responses.append("Error: Invalid item format")
            continue

        # Construct messages for Doubao
        doubao_messages = []
        if system_prompt:
            doubao_messages.append({"role": "system", "content": system_prompt})
        
        if user_content:
            doubao_messages.append({"role": "user", "content": user_content})

        # Retry logic
        max_retries = 10
        sleep_time = 1
        result = None
        curr_retries = 0
        
        while curr_retries < max_retries and result is None:
            try:
                response = client.responses.create(
                    model=model,
                    input=doubao_messages,
                    thinking={"type":"enabled"}
                )
                # print(response)
                # Extract response text
                answer = str(response)
                if hasattr(response, 'output') and response.output:
                    for resp_item in response.output:
                        if hasattr(resp_item, 'type') and resp_item.type == 'message':
                            if hasattr(resp_item, 'content') and resp_item.content:
                                for content_item in resp_item.content:
                                    if hasattr(content_item, 'text') and content_item.text:
                                        answer = content_item.text
                                        break
                result = answer
                
            except Exception as e:
                print(f"Error: {e}")
                print(f"Error {curr_retries}, sleeping for {sleep_time} seconds...")
                time.sleep(sleep_time)
                curr_retries += 1
                sleep_time *= 1.6
        
        if result is None:
            result = "Error in generating response."
            print(f"Error in generating response for {k}")
        # print(result)
        responses.append(result)
        
        
    return responses


if __name__ == "__main__":
    import os
    
    # 测试代码：使用 example/dog.png 测试模型描述图片
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "example", "dog.png")
    
    # 设置模型路径和 API Key（可以根据实际情况修改）
    model_path = "doubao-seed-1-6-251015"
    api_key = os.getenv('ARK_API_KEY')  # 从环境变量获取，或直接设置
    if api_key is None:
        api_key = "106e3245-4f9f-4308-853c-b76be1a8fc20"  # 请替换为您的 API Key
    
    # 构造测试查询 (新格式：列表的列表)
    queries = [
        [
            ('system', 'You are a helpful assistant that can describe images.'),
            ('text', 'Please describe this image in detail.'),
            ('image', image_path)
        ]
    ]
    
    print(f"正在测试模型: {model_path}")
    print(f"图片路径: {image_path}")
    print(f"查询内容: {queries[0]}")
    print("-" * 50)
    
    # 初始化客户端和模型
    client, model = get_client_model(model_path, api_key)
    
    # 调用生成函数
    try:
        responses = generate_response(queries, model, api_key=api_key, client=client)
        print("\n模型回答:")
        for i, response in enumerate(responses):
            print(f"Response {i+1}: {response}")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

