import os
import json
import argparse
import re
import numpy as np

parser = argparse.ArgumentParser(description="Evaluate numeric answers in JSON files")
parser.add_argument("--input_file", type=str, default="travelmap_shortest_path_only_map_Qwen3-VL-8B-Instruct_results.json")
args = parser.parse_args()


# -------------------------------------------------------
# 工具函数：提取 response 数字
# -------------------------------------------------------
def extract_answer_from_response(response):
    """从 <answer_begin>...</answer_end> 中提取数字，如果不存在或无法转换为 float，则返回 None"""
    match = re.search(r"<answer_begin>(.*?)<answer_end>", response)
    if match:
        try:
            return round(float(match.group(1)), 2)
        except:
            return None
    return None


def extract_answer(answer):
    """将 answer 转为数字，两位小数，如果无法转换则返回 None"""
    try:
        return round(float(answer), 2)
    except:
        return None


# -------------------------------------------------------
# evaluate_item
# -------------------------------------------------------
def evaluate_item(item):
    response_num = extract_answer_from_response(item.get("response", ""))
    answer_num = extract_answer(item.get("answer", ""))

    # 异常情况直接判为错误
    if response_num is None or answer_num is None:
        correct = 0
    else:
        correct = 1 if response_num == answer_num else 0

    # 将结果写入 item
    item["response_num"] = response_num
    item["answer_num"] = answer_num
    item["correct"] = correct

    return correct


# -------------------------------------------------------
# 主程序
# -------------------------------------------------------
def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 脚本所在的上上级目录
    input_file = os.path.basename(args.input_file)  # 输入文件名称
    folder_name = os.path.basename(os.path.dirname(args.input_file))  # 输入文件所在文件夹名称
    input_path = args.input_file

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        datas = json.load(f)

    correct_list = []

    for item in datas:
        correct = evaluate_item(item)
        correct_list.append(correct)

    # -------------------------------------------------------
    # 输出文件路径（在 ../results_evaluate/ 下）
    # -------------------------------------------------------
    results_dir = os.path.join(base_dir, "results_evaluate", f"evaluate_qa_{folder_name}")
    os.makedirs(results_dir, exist_ok=True)

    input_name = input_file.replace(".json", "")
    output_filename = f"evaluate_qa_{input_name}.json"
    output_file = os.path.join(results_dir, output_filename)

    # 写入结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(datas, f, ensure_ascii=False, indent=4)

    # 统计平均准确率
    avg_acc = np.mean(correct_list)

    print(f"Saved evaluated results to: {output_file}\n")
    print(f"Average accuracy = {avg_acc:.4f}")

    # -------------------------------------------------------
    # 写入 summary 文件
    # -------------------------------------------------------
    summary_file = os.path.join(results_dir, f"evaluate_qa_{folder_name}.txt")
    print(f"Summary file will be saved to: {summary_file}")
    with open(summary_file, 'a', encoding='utf-8') as f:
        f.write(f"input_file: {input_file}\n")
        f.write(f"average_accuracy: {avg_acc:.4f}\n")
        f.write("-" * 50 + "\n")

    print("Evaluation completed and results saved.")


if __name__ == "__main__":
    main()
