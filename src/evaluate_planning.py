import argparse
import os
import json
import difflib
import numpy as np

parser = argparse.ArgumentParser(description="Evaluate MLLM route accuracy without LLM")
parser.add_argument("--input_file", type=str, default="travelmap_shortest_path_only_map_Qwen3-VL-8B-Instruct_results.json")
args = parser.parse_args()

# -------------------------------------------------------
# 工具函数：语义相似度（简易版，处理错别字）
# -------------------------------------------------------
def semantic_similarity(a, b):
    a = a.strip().lower()
    b = b.strip().lower()
    return difflib.SequenceMatcher(None, a, b).ratio()

def is_same_station(a, b):
    return semantic_similarity(a, b) >= 0.5


# -------------------------------------------------------
# 计算 part_acc
# -------------------------------------------------------
def calc_part_acc(std_route, mllm_route):
    match_len = 0
    for s1, s2 in zip(std_route, mllm_route):
        if is_same_station(s1, s2):
            match_len += 1
        else:
            break
    return match_len / len(std_route)


# -------------------------------------------------------
# 计算 all_acc
# -------------------------------------------------------
def calc_all_acc(std_route, mllm_route):
    if len(std_route) != len(mllm_route):
        return 0

    for s1, s2 in zip(std_route, mllm_route):
        if "(transfer)" in s1 or "(transfer)" in s2:
            if s1 != s2:
                return 0
        else:
            if not is_same_station(s1, s2):
                return 0

    return 1


# -------------------------------------------------------
# Difficulty score 映射（新增）
# -------------------------------------------------------
DIFFICULTY_SCORE_MAP = {
    ("easy", "easy"): 1,
    ("easy", "medium"): 2,
    ("easy", "hard"): 3,
    ("medium", "easy"): 4,
    ("medium", "medium"): 5,
    ("medium", "hard"): 6,
    ("hard", "easy"): 7,
    ("hard", "medium"): 8,
    ("hard", "hard"): 9,
}

def calc_difficulty_score(map_diff, query_diff):
    return DIFFICULTY_SCORE_MAP.get(
        (map_diff.lower(), query_diff.lower()), 0
    )


# -------------------------------------------------------
# evaluate_item
# -------------------------------------------------------
def evaluate_item(item):
    mllm_route = item["response"].split("-")
    std_routes = [r.split("-") for r in item["routes"]]

    all_acc = 0
    highest_part_acc = 0.0

    for std in std_routes:
        if calc_all_acc(std, mllm_route) == 1:
            all_acc = 1

        highest_part_acc = max(highest_part_acc, calc_part_acc(std, mllm_route))

    return all_acc, round(highest_part_acc, 4)


# -------------------------------------------------------
# 主程序
# -------------------------------------------------------
def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.basename(args.input_file)
    folder_name = os.path.basename(os.path.dirname(args.input_file))

    input_path = args.input_file

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        datas = json.load(f)

    all_acc_list = []
    part_acc_list = []
    difficulty_total_score = 0  # 新增：总分

    for item in datas:
        all_acc, part_acc = evaluate_item(item)
        all_acc_list.append(all_acc)
        part_acc_list.append(part_acc)

        # Difficulty score（新增）
        diff_score = calc_difficulty_score(
            item["Map_Difficulty"],
            item["Query_Difficulty"]
        )

        # 只有完全正确才加分
        if all_acc == 1:
            difficulty_total_score += diff_score

        # 写回 item
        item["all_acc"] = all_acc
        item["part_acc"] = float(f"{part_acc:.4f}")
        item["Difficulty_score"] = diff_score

    # -------------------------------------------------------
    # 输出文件
    # -------------------------------------------------------
    results_dir = os.path.join(base_dir, "results_evaluate", f"evaluate_planning_{folder_name}")
    os.makedirs(results_dir, exist_ok=True)

    input_name = input_file.replace(".json", "")
    output_filename = f"evaluate_planning_{input_name}.json"
    output_file = os.path.join(results_dir, output_filename)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(datas, f, ensure_ascii=False, indent=4)

    avg_all = np.mean(all_acc_list)
    avg_part = np.mean(part_acc_list)

    print(f"Saved evaluated results to: {output_file}\n")
    print(f"Average all_acc = {avg_all:.4f}")
    print(f"Average part_acc = {avg_part:.4f}")
    print(f"Difficulty_score_total = {difficulty_total_score}")

    # -------------------------------------------------------
    # summary 文件
    # -------------------------------------------------------
    summary_file = os.path.join(results_dir, f"evaluate_planning_{folder_name}.txt")
    print(f"Summary file will be saved to: {summary_file}")

    with open(summary_file, 'a', encoding='utf-8') as f:
        f.write(f"input_file: {input_file}\n")
        f.write(f"all_acc: {avg_all:.4f}\n")
        f.write(f"part_acc: {avg_part:.4f}\n")
        f.write(f"Difficulty_score_total: {difficulty_total_score}\n")
        f.write("-" * 50 + "\n")

    print("Evaluation completed and results saved.")

if __name__ == "__main__":
    main()
