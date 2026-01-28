import base64
import os
import time
import csv
import re
import json as json_module
import pandas as pd
import random
import numpy as np
import torch
from openai import AzureOpenAI
from tqdm import tqdm

def set_random_seed(seed):
    """设置所有随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_client_model(model_path, api_key):
    assert api_key is not None, "API key is required for using Azure GPT"
    assert model_path is not None, "Model name is required for using Azure GPT"
    
    # Azure OpenAI configuration - endpoint should be set via environment variable
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    deployment = model_path  # Use model_path as deployment name
    api_version = os.environ.get("AZURE_API_VERSION")
    
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
    )
    
    return client, model_path

def generate_response(queries, model, client=None, media_type="image/jpeg", api_key=None, random_baseline=False, task=None, subtask=None, skip_indices=None, on_response_callback=None, seed=42):
    if client is None and api_key is not None:
        client, _ = get_client_model(model, api_key)
    
    # 设置随机种子确保可复现性
    set_random_seed(seed)
    print(f"Random seed set to: {seed}")

    if skip_indices is None:
        skip_indices = set()
        
    # 可以根据 task 和 subtask 调整参数
    if task and subtask:
        print(f"Processing task: {task}, subtask: {subtask}")

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # 工具函数：根据文件路径判断是 edge_tab 还是 vertex_tab
    def identify_table_type(file_path):
        """根据文件路径判断是 edge_tab 还是 vertex_tab"""
        file_path_lower = file_path.lower()
        if 'edge' in file_path_lower:
            return 'edge'
        elif 'vertex' in file_path_lower:
            return 'vertex'
        else:
            # 默认返回 edge，或者可以根据其他规则判断
            return 'edge'
    
    # 工具函数：处理 CSV 文件，根据列名过滤
    def process_csv_file(file_path, del_cols):
        """处理 CSV 文件，删除指定的列"""
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            # 删除指定的列
            if del_cols:
                df = df.drop(columns=[col for col in del_cols if col in df.columns], errors='ignore')
            # 转换为 CSV 格式字符串
            return df.to_csv(index=False)
        except Exception as e:
            # 如果 pandas 读取失败，尝试使用 csv 模块
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                # 过滤列
                if del_cols:
                    filtered_rows = []
                    for row in rows:
                        filtered_row = {k: v for k, v in row.items() if k not in del_cols}
                        filtered_rows.append(filtered_row)
                    rows = filtered_rows
                return json_module.dumps(rows, ensure_ascii=False, indent=2)
            except Exception as e2:
                raise Exception(f"Error processing CSV file: {e2}")
    
    # 工具函数：处理 JSON 文件，根据键名过滤
    def process_json_file(file_path, del_cols):
        """处理 JSON 文件，删除指定的键"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json_module.load(f)
            
            # 如果是列表，处理每个元素
            if isinstance(data, list):
                filtered_data = []
                for item in data:
                    if isinstance(item, dict):
                        filtered_item = {k: v for k, v in item.items() if k not in del_cols}
                        filtered_data.append(filtered_item)
                    else:
                        filtered_data.append(item)
                data = filtered_data
            # 如果是字典，直接过滤
            elif isinstance(data, dict):
                data = {k: v for k, v in data.items() if k not in del_cols}
            
            return json_module.dumps(data, ensure_ascii=False, indent=2)
        except Exception as e:
            raise Exception(f"Error processing JSON file: {e}")

    def extract_between_tags(text, tag):
        pattern = f"<{tag}_begin>(.*?)<{tag}_end>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

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
            del_cols_edge=["Time","Price","Comfort Level","Reliability"]
            del_cols_vertex=[]
        elif subtask == "shortest_path_only_map":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "shortest_path_map_and_tab_no_constraint":
            del_cols_edge=["Time","Price","Comfort Level","Reliability"]
            del_cols_vertex=[]
        elif subtask == "demo":
            del_cols_edge=["Time","Price","Comfort Level","Reliability"]
            del_cols_vertex=[]
        elif subtask == "shortest_path_map_and_tab_with_constraint_1":
            del_cols_edge=["Price","Comfort Level","Reliability"]
            del_cols_vertex=["Price","Comfort Level","Reliability"]
        elif subtask == "shortest_path_map_and_tab_with_constraint_2":
            del_cols_edge=["Time","Comfort Level","Reliability"]
            del_cols_vertex=["Time","Comfort Level","Reliability","Transfer Time"]
        elif subtask == "shortest_path_map_and_tab_with_constraint_3":
            del_cols_edge=["Time","Price","Reliability"]
            del_cols_vertex=["Time","Price","Reliability","Transfer Time"]
        elif subtask == "shortest_path_map_and_tab_with_constraint_4":
            del_cols_edge=["Time","Price","Comfort Level"]
            del_cols_vertex=["Time","Price","Comfort Level","Transfer Time"]
        elif subtask == "shortest_path_map_and_tab_with_constraint_1_2_3_4":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "shortest_path_map_and_tab_with_constraint_1_2_4":
            del_cols_edge=["Comfort Level"]
            del_cols_vertex=["Comfort Level"]
        elif subtask == "shortest_path_map_and_tab_with_constraint_1_3_4":
            del_cols_edge=["Price"]
            del_cols_vertex=["Price"]
        elif subtask == "shortest_path_map_and_tab_with_constraint_2_3_4":
            del_cols_edge=["Time"]
            del_cols_vertex=["Time","Transfer Time"]
        elif subtask == "only_vertex2":
            del_cols_edge=[]
            del_cols_vertex=["Line"]

        # CSV消融实验
        elif subtask == "4_csv_edge_global":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "5_csv_edge_part":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "6_csv_edge_spatial_judge":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "7_csv_vertex_global":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "8_csv_vertex_part":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "9_csv_vertex_spatial_judge":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "10_csv_and_pic_global":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "11_csv_and_pic_part":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "12_csv_and_pic_spatial_judge":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "shortest_path_csv_vertex2":
            del_cols_edge=[]
            del_cols_vertex=["Line"]
        elif subtask == "shortest_path_map_and_tab_csv_constraint_1_2_3_4":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "shortest_path_only_csv":
            del_cols_edge=["Time","Price","Comfort Level","Reliability"]
            del_cols_vertex=[]
        elif subtask == "shortest_path_map_and_csv":
            del_cols_edge=["Time","Price","Comfort Level","Reliability"]
            del_cols_vertex=[]

        elif subtask == "1_qa_only_pic_global":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "2_qa_only_pic_part":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "3_qa_only_pic_spatial_judge":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "4_qa_edge_tab_global":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "5_qa_edge_tab_part":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "6_qa_edge_tab_spatial_judge":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "7_qa_vertex_tab_global":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "8_qa_vertex_tab_part":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "9_qa_vertex_tab_spatial_judge":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "10_qa_pic_and_tab_global":
            del_cols_edge=[]
            del_cols_vertex=["Line"]
        elif subtask == "11_qa_pic_and_tab_part":
            del_cols_edge=[]
            del_cols_vertex=["Line"]
        elif subtask == "12_qa_pic_and_tab_spatial_judge":
            del_cols_edge=[]
            del_cols_vertex=["Line"]
        elif subtask in [ "shortest_path_with_qa_and_constraint_1_2_3_4",
                        "shortest_path_with_qa_and_constraint_1_2_4",
                        "shortest_path_with_qa_and_constraint_1_3_4",
                        "shortest_path_with_qa_and_constraint_2_3_4",
                        "shortest_path_with_qa_and_constraint_1",
                        "shortest_path_with_qa_and_constraint_2",
                        "shortest_path_with_qa_and_constraint_3",
                        "shortest_path_with_qa_and_constraint_4"]:
            del_cols_edge=[]
            del_cols_vertex=[]
        else:
            raise ValueError(f"Unknown subtask '{subtask}' for task 'metromap'")

    elif task == "travelmap":
        if subtask == "shortest_path_only_tab":
            del_cols_edge=["Time","Price","Comfort Level","Reliability"]
            del_cols_vertex=[]
        elif subtask == "shortest_path_only_map":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "shortest_path_map_and_tab_no_constraint":
            del_cols_edge=["Time","Price","Comfort Level","Reliability"]
            del_cols_vertex=[]
        elif subtask == "demo":
            del_cols_edge=["Time","Price","Comfort Level","Reliability"]
            del_cols_vertex=[]
        elif subtask == "shortest_path_map_and_tab_with_constraint_1":
            del_cols_edge=["Price","Comfort Level","Reliability"]
            del_cols_vertex=["Price","Comfort Level","Reliability"]
        elif subtask == "shortest_path_map_and_tab_with_constraint_2":
            del_cols_edge=["Comfort Level","Reliability"]
            del_cols_vertex=["Comfort Level","Reliability"]
        elif subtask == "shortest_path_map_and_tab_with_constraint_3":
            del_cols_edge=["Time","Price","Reliability"]
            del_cols_vertex=["Time","Price","Reliability"]
        elif subtask == "shortest_path_map_and_tab_with_constraint_4":
            del_cols_edge=["Time","Price","Comfort Level"]
            del_cols_vertex=["Time","Price","Comfort Level"]
        elif subtask == "shortest_path_map_and_tab_with_constraint_1_2_3_4":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "shortest_path_map_and_tab_with_constraint_1_2_4":
            del_cols_edge=["Comfort Level"]
            del_cols_vertex=["Comfort Level"]
        elif subtask == "shortest_path_map_and_tab_with_constraint_1_3_4":
            del_cols_edge=["Price"]
            del_cols_vertex=["Price"]
        elif subtask == "shortest_path_map_and_tab_with_constraint_2_3_4":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "only_vertex2":
            del_cols_edge=[]
            del_cols_vertex=[]

        # CSV消融实验
        elif subtask == "4_csv_edge_global":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "5_csv_edge_part":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "6_csv_edge_spatial_judge":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "7_csv_vertex_global":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "8_csv_vertex_part":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "9_csv_vertex_spatial_judge":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "10_csv_and_pic_global":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "11_csv_and_pic_part":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "12_csv_and_pic_spatial_judge":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "shortest_path_csv_vertex2":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "shortest_path_map_and_tab_csv_constraint_1_2_3_4":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "shortest_path_only_csv":
            del_cols_edge=["Time","Price","Comfort Level","Reliability"]
            del_cols_vertex=[]
        elif subtask == "shortest_path_map_and_csv":
            del_cols_edge=["Time","Price","Comfort Level","Reliability"]
            del_cols_vertex=[]

        elif subtask == "1_qa_only_pic_global":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "2_qa_only_pic_part":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "3_qa_only_pic_spatial_judge":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "4_qa_edge_tab_global":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "5_qa_edge_tab_part":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "6_qa_edge_tab_spatial_judge":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "7_qa_vertex_tab_global":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "8_qa_vertex_tab_part":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "9_qa_vertex_tab_spatial_judge":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "10_qa_pic_and_tab_global":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "11_qa_pic_and_tab_part":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask == "12_qa_pic_and_tab_spatial_judge":
            del_cols_edge=[]
            del_cols_vertex=[]
        elif subtask in [ "shortest_path_with_qa_and_constraint_1_2_3_4",
                        "shortest_path_with_qa_and_constraint_1_2_4",
                        "shortest_path_with_qa_and_constraint_1_3_4",
                        "shortest_path_with_qa_and_constraint_2_3_4",
                        "shortest_path_with_qa_and_constraint_1",
                        "shortest_path_with_qa_and_constraint_2",
                        "shortest_path_with_qa_and_constraint_3",
                        "shortest_path_with_qa_and_constraint_4"]:
            del_cols_edge=[]
            del_cols_vertex=[]
        else:
            raise ValueError(f"Unknown subtask '{subtask}' for task 'travelmap'")

    responses = []
    for k in tqdm(iterator):
        # 跳过已完成的索引
        if k in skip_indices:
            responses.append(None)  # 占位，保持索引一致
            continue
        
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

        # Construct messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if user_content:
            messages.append({"role": "user", "content": user_content})

        # Retry logic
        max_retries = 10
        sleep_time = 1
        result = None
        curr_retries = 0
        
        while curr_retries < max_retries and result is None:
            try:
                response = client.chat.completions.create(
                    messages=messages,
                    max_completion_tokens=1000,
                    temperature=0,
                    top_p=1.0,
                    seed=seed,
                    model=model  # This should be the deployment name
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
        
        # 调用回调函数保存结果
        if on_response_callback is not None:
            on_response_callback(k, result)
        
    return responses

def generate_all_results_response(queries, model, client=None, media_type="image/jpeg", api_key=None, task=None, subtask=None, seed=42):
    """
    生成响应并提取所有结果（route, Total_Time, Total_Price, Average_Comfort Level, Average_Reliability）
    
    Args:
        queries: 查询列表或字典
        model: 模型名称
        client: AzureOpenAI客户端（可选）
        media_type: 媒体类型（默认为"image/jpeg"）
        api_key: API密钥（可选）
        task: 任务名称（可选）
        subtask: 子任务名称（可选）
        seed: 随机种子（默认为42）
    
    Returns:
        responses: 包含结构化结果的字典列表
    """
    if client is None and api_key is not None:
        client, _ = get_client_model(model, api_key)
    
    # 设置随机种子确保可复现性
    set_random_seed(seed)
    print(f"Random seed set to: {seed}")

    # 可以根据 task 和 subtask 调整参数
    if task and subtask:
        print(f"Processing task: {task}, subtask: {subtask}")

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # 工具函数：根据文件路径判断是 edge_tab 还是 vertex_tab
    def identify_table_type(file_path):
        """根据文件路径判断是 edge_tab 还是 vertex_tab"""
        file_path_lower = file_path.lower()
        if 'edge' in file_path_lower:
            return 'edge'
        elif 'vertex' in file_path_lower:
            return 'vertex'
        else:
            # 默认返回 edge，或者可以根据其他规则判断
            return 'edge'
    
    # 工具函数：处理 CSV 文件，根据列名过滤
    def process_csv_file(file_path, del_cols):
        """处理 CSV 文件，删除指定的列"""
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            # 删除指定的列
            if del_cols:
                df = df.drop(columns=[col for col in del_cols if col in df.columns], errors='ignore')
            # 转换为 JSON 格式字符串（更易读）
            return df.to_json(orient='records', force_ascii=False, indent=2)
        except Exception as e:
            # 如果 pandas 读取失败，尝试使用 csv 模块
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                # 过滤列
                if del_cols:
                    filtered_rows = []
                    for row in rows:
                        filtered_row = {k: v for k, v in row.items() if k not in del_cols}
                        filtered_rows.append(filtered_row)
                    rows = filtered_rows
                return json_module.dumps(rows, ensure_ascii=False, indent=2)
            except Exception as e2:
                raise Exception(f"Error processing CSV file: {e2}")
    
    # 工具函数：处理 JSON 文件，根据键名过滤
    def process_json_file(file_path, del_cols):
        """处理 JSON 文件，删除指定的键"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json_module.load(f)
            
            # 如果是列表，处理每个元素
            if isinstance(data, list):
                filtered_data = []
                for item in data:
                    if isinstance(item, dict):
                        filtered_item = {k: v for k, v in item.items() if k not in del_cols}
                        filtered_data.append(filtered_item)
                    else:
                        filtered_data.append(item)
                data = filtered_data
            # 如果是字典，直接过滤
            elif isinstance(data, dict):
                data = {k: v for k, v in data.items() if k not in del_cols}
            
            return json_module.dumps(data, ensure_ascii=False, indent=2)
        except Exception as e:
            raise Exception(f"Error processing JSON file: {e}")

    def extract_between_tags(text, tag):
        pattern = f"<{tag}_begin>(.*?)<{tag}_end>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    # Determine if queries is a dict or list and get iterator
    if isinstance(queries, dict):
        iterator = queries.keys()
    elif isinstance(queries, list):
        iterator = range(len(queries))
    else:
        raise ValueError("Queries must be a dict or list")

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
            # 添加空结果并继续
            responses.append({
                "response": None,
                "Answer_Total_Time": None,
                "Answer_Total_Price": None,
                "Answer_Average_Comfort_Level": None,
                "Answer_Average_Reliability": None
            })
            continue

        # Construct messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if user_content:
            messages.append({"role": "user", "content": user_content})

        # Retry logic
        max_retries = 10
        sleep_time = 1
        result = None
        curr_retries = 0
        
        while curr_retries < max_retries and result is None:
            try:
                response = client.chat.completions.create(
                    messages=messages,
                    max_completion_tokens=1000,
                    temperature=0,
                    top_p=1.0,
                    seed=seed,
                    model=model  # This should be the deployment name
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
        
        # 提取标签内容
        route = extract_between_tags(result, "route")
        total_time = extract_between_tags(result, "Total_Time")
        total_price = extract_between_tags(result, "Total_Price")
        avg_comfort = extract_between_tags(result, "Average_Comfort Level")
        avg_reliability = extract_between_tags(result, "Average_Reliability")
        
        # 组织成字典存储
        responses.append({
            "response": route,
            "Answer_Total_Time": total_time,
            "Answer_Total_Price": total_price,
            "Answer_Average_Comfort_Level": avg_comfort,
            "Answer_Average_Reliability": avg_reliability
        })
        # print(f"Processed response for query {k}: {responses[-1]}")
    
    return responses
