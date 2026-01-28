import time
from tqdm import tqdm

def generate_response_remote_wrapper(generate_fn, 
        queries, model_path, api_key, client, task=None, subtask=None, seed=42, init_sleep=1, 
        max_retries=10, sleep_factor=1.6):
    """
    包装生成函数，处理远程API调用
    
    Args:
        generate_fn: 生成函数，应该接受 (queries, model_path, ...) 参数
        queries: 查询列表或字典
        model_path: 模型路径
        api_key: API密钥
        client: API客户端
        task: 任务名称
        subtask: 子任务名称
        seed: 随机种子（默认为42）
        init_sleep: 初始重试等待时间
        max_retries: 最大重试次数
        sleep_factor: 重试等待时间倍增因子
    """
    # 检查 queries 的类型
    if isinstance(queries, list):
        # 列表格式：直接调用 generate_fn，它应该能够处理列表格式
        # generate_fn 的签名通常是: generate_response(queries, model, client=None, api_key=None, ...)
        responses = generate_fn(queries, model_path, client=client, api_key=api_key, task=task, subtask=subtask, seed=seed)
        return responses
    elif isinstance(queries, dict):
        # 字典格式：保持原有逻辑（向后兼容）
        responses = []
        for k in tqdm(queries):
            sleep_time = init_sleep
            query = queries[k]['question']
            image = queries[k]["figure_path"]
            curr_retries = 0
            result = None
            while curr_retries < max_retries and result is None:
                try:
                    result = generate_fn(image, query, model_path, 
                        api_key=api_key, client=client, random_baseline=False, task=task, subtask=subtask)
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"Error {curr_retries}, sleeping for {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    curr_retries += 1
                    sleep_time *= sleep_factor
            if result is None:
                result = "Error in generating response."
                print(f"Error in generating response for {k}")
            queries[k]['response'] = result
            responses.append(result)
        return responses
    else:
        raise ValueError(f"queries must be a list or dict, got {type(queries)}")

def get_client_fn(model_path):
    # gemini
    model_name = model_path.split('/')[-1]
    if model_name in [
                        'qwen3-vl-32b-instruct',
                        'qwen3-vl-8b-instruct',
                        'qwen3-vl-30b-a3b-instruct',
                        # "qwen3-vl-plus"
                        ]:
        from .qwen_api import get_client_model
    elif model_name in [
                        'qwen3-vl-32b-thinking',
                        'qwen3-vl-8b-thinking',
                        "qwen3-vl-plus",
                        'qwen3-max']:
        from .qwen_thinking_api import get_client_model
    elif model_path in ['gpt-4.1', 'gpt-5', 'gpt-4o']:
        from .azure_gpt import get_client_model
    elif model_path in ['doubao-seed-1-6-251015']:
        from .doubao import get_client_model
    elif model_name in ['gemini-3-flash-preview']:
        from .gemini_yunwu import get_client_model
    else:
        raise ValueError(f"Model {model_path} not supported")
    return get_client_model

def get_generate_fn(model_path):
    model_name = model_path.split('/')[-1]
    # vLLM 支持的模型
    if model_name in [
                        # 暂定测评模型
                        'Qwen3-VL-8B-Instruct',
                        'Qwen3-VL-8B-Thinking',
                        'Qwen3-VL-30B-A3B-Thinking',
                        'Qwen2.5-VL-7B-Instruct',
                        'Kimi-VL-A3B-Thinking-2506',
                        'Kimi-VL-A3B-Instruct',
                        'Phi-4-multimodal-instruct',
                        'Phi-3.5-vision-instruct',
                        'Glyph',

                        # 备选测评模型
                        'Qwen3-VL-2B-Instruct',
                        'llava-v1.6-mistral-7b-hf',
                        'InternVL3_5-30B-A3B',
                        'InternVL3_5-8B',
                        'Ovis2.5-9B'
                    ]:
        from .vllm_LLMengine import generate_response
    elif model_name in [
                        'qwen3-vl-32b-instruct',
                        'qwen3-vl-8b-instruct',
                        'qwen3-vl-30b-a3b-instruct',
                        # "qwen3-vl-plus"
                        ]:
        from .qwen_api import generate_response
    elif model_name in [
                        'qwen3-vl-32b-thinking',
                        'qwen3-vl-8b-thinking',
                        "qwen3-vl-plus",
                        'qwen3-max']:
        from .qwen_thinking_api import generate_response
    elif model_name in ['gpt-4.1', 'gpt-5','gpt-4o']:
        from .azure_gpt import generate_response
    elif model_name in ['doubao-seed-1-6-251015']:
        from .doubao import generate_response
    elif model_name in ['gemini-3-flash-preview']:
        from .gemini_yunwu import generate_response
    else:
        raise ValueError(f"Model {model_name} not supported")
    return generate_response
