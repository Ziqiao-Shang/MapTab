# Adapted from qwen3_vl_8b_api_map_and_tab.py
# This has support for the Qwen3-VL model using vLLM

import base64
import os
import re
import csv
import json as json_module
import pandas as pd
import random
import numpy as np
import torch
from vllm import LLM, SamplingParams
from tqdm import tqdm
from PIL import Image
import io
import math


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

def generate_response(queries, model_path, task=None, subtask=None, seed=42):
    """
    生成响应函数
    
    Args:
        queries: 查询列表或字典
        model_path: 模型路径
        task: 任务名称（可选）
        subtask: 子任务名称（可选）
        seed: 随机种子（默认为42）
    
    Returns:
        responses: 响应列表
    """
    # 设置随机种子确保可复现性
    set_random_seed(seed)
    print(f"Random seed set to: {seed}")
    
    # 初始化模型
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype="auto",
        # max_model_len=262144,
        max_model_len=128000,
        gpu_memory_utilization=0.9,
        seed=seed
    )

    # 设置采样参数
    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=0.0
    )
    
    # 可以根据 task 和 subtask 调整参数
    if task and subtask:
        print(f"Processing task: {task}, subtask: {subtask}")

    # 工具函数：将图像编码为 base64
    def encode_image_to_base64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    
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
            # return df.to_json(orient='records', force_ascii=False, indent=2)
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
        else:
            raise ValueError(f"Unknown subtask '{subtask}' for task 'travelmap'")

    responses=[]
    for k in tqdm(iterator):
        item = queries[k]
        
        # Initialize message components
        system_prompt = None
        user_content = []
        
        # Process the item (list of tuples)
        # Example item: [('system', '...'), ('text', '...'), ('image', 'path'), ('csv', 'path')]
        if isinstance(item, list):
            for type_, content in item:
                if type_ == 'system':
                    system_prompt = content
                elif type_ == 'text':
                    user_content.append({"type": "text", "text": content})
                elif type_ == 'image':
                    try:
                        base64_image = encode_image_to_base64(content)
                        user_content.append({
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
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
                        #print(file_content)
                        #print(type(file_content))
                        table_label = "Edge Table (JSON)" if table_type == 'edge' else "Vertex Table (JSON)"
                        #print( f"{table_label}:\n{file_content}")
                        user_content.append({"type": "text", "text": f"{table_label}:\n{file_content}"})
                        #print(user_content)
                    except Exception as e:
                        print(f"Error loading JSON file {content}: {e}")
                        user_content.append({"type": "text", "text": f"[Error loading JSON file: {content}]"})
        else:
            print(f"Warning: Item {k} is not a list, skipping. Type: {type(item)}")
            continue

        # Construct messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if user_content:
            messages.append({"role": "user", "content": user_content})
        #print(messages)
        output = llm.chat(messages, sampling_params=sampling_params)
        answer = output[0].outputs[0].text.strip()
        responses.append(answer)
    return responses

if __name__ == "__main__":
    import os
    # 测试代码：使用 example/dog.png 测试模型描述图片
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "example", "dog.png")
    
    # 构造测试查询 (新格式：列表的列表)
    # 每一项是一个列表，包含多个元组 (type, content)
    queries = [
        [
            ('system', 'You are a helpful assistant that can describe images.'),
            ('text', 'Please describe this image in detail.'),
            ('image', image_path)
        ]
    ]
    
    # 设置模型路径（可以根据实际情况修改）
    model_path = "Qwen/Qwen3-VL-8B-Instruct"
    
    print(f"正在测试模型: {model_path}")
    print(f"图片路径: {image_path}")
    print(f"查询内容: {queries[0]}")
    print("-" * 50)
    
    # 调用生成函数
    try:
        responses = generate_response(queries, model_path, task="test", subtask="test")
        
        # 打印结果
        print("\n模型回答:")
        for i, response in enumerate(responses):
            print(f"Response {i+1}: {response}")
            
    except Exception as e:
        print(f"Error during test: {e}")