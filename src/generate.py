import argparse
import sys
import os
import json

# 添加 generate_lib 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'generate_lib'))

from generate_lib.utils import get_generate_fn, get_client_fn, generate_response_remote_wrapper
from metromap_utils import build_metromap_queries   
from travelmap_utils import build_travelmap_queries
def main():
    parser = argparse.ArgumentParser(description='Generate responses using vision-language models')
    
    parser.add_argument('--task', type=str, required=True,
                        help='Task name (e.g., metromap_shortest_path_query)')
    parser.add_argument('--subtask', type=str, required=True,
                        help='Subtask name (e.g., map_and_tab_no_constraint, only_map, only_tab)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Model path or model name')
    parser.add_argument('--api_key', type=str, default=None,
                        help='API key for API-based models (optional for local models)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    task = args.task
    subtask = args.subtask
    model_path = args.model_path
    api_key = args.api_key
    seed = args.seed
    
    print(f"Task: {task}")
    print(f"Subtask: {subtask}")
    print(f"Model: {model_path}")
    print(f"API Key: {'Provided' if api_key else 'Not provided'}")
    print(f"Random Seed: {seed}")
    print("-" * 50)
    
    # 构建数据文件路径并加载 queries
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if task == "metromap":
        queries, datas = build_metromap_queries(task, subtask, base_dir)
    elif task == "travelmap":
        queries, datas = build_travelmap_queries(task, subtask, base_dir)
    else:
        raise ValueError(f"Unknown task '{task}'")
    
    # 获取生成函数
    generate_fn = get_generate_fn(model_path)
    
    # 根据模型类型选择调用方式
    if api_key:
        # API 模型：需要 client
        client_fn = get_client_fn(model_path)
        client, model = client_fn(model_path, api_key)
        # generate_response_remote_wrapper 支持 task 和 subtask 参数
        responses=generate_response_remote_wrapper(generate_fn, queries, model_path, api_key, client, task=task, subtask=subtask)
    else:
        # 本地模型：直接调用
        responses=generate_fn(queries, model_path, task=task, subtask=subtask, seed=seed)
    


    # 将 responses 写入 datas
    if len(responses) != len(datas):
        print(f"Warning: Number of responses ({len(responses)}) does not match number of data items ({len(datas)})")
    
    for i, response in enumerate(responses):
        if i < len(datas):
            if type(response) is tuple:
                datas[i]['response'] = response[1]
                datas[i]['reasoning content'] = response[0]
            else:
                datas[i]['response'] = response
            
    # 保存结果
    results_dir = os.path.join(base_dir, "results", "response_generate")
    os.makedirs(results_dir, exist_ok=True)
    
    # 构建文件名: task_subtask_modelname_results.json
    # model_path 可能是路径，取最后一部分作为 modelname
    model_name = os.path.basename(model_path)
    # 移除可能的扩展名
    if model_name.endswith('.pt') or model_name.endswith('.pth') or model_name.endswith('.bin'):
         model_name = os.path.splitext(model_name)[0]
         
    output_filename = f"{task}_{subtask}_{model_name}_results.json"
    output_file = os.path.join(results_dir, output_filename)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(datas, f, ensure_ascii=False, indent=4)
        
    print(f"Results saved to: {output_file}")
    
    print("Generation completed.")

if __name__ == "__main__":
    main()

