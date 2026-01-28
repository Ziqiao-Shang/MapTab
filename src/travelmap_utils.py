import os
import json
WORKSPACE_DIR = os.getenv('WORKSPACE_DIR')

def build_travelmap_queries(task, subtask, base_dir):
    """
    根据 task 和 subtask 构建数据文件路径并加载 queries
    
    Args:
        task: 任务名称 ('metromap' 或 'travelmap')
        subtask: 子任务名称
        base_dir: 基础目录路径
    
    Returns:
        queries: 加载的查询数据（字典格式）
    """
    data_file = None

    # travelmap 的 subtask 选项
    if subtask == "shortest_path_only_tab":
        data_file = os.path.join(base_dir, "travelmap", "data", "test_set", "travelmap_shortest_path_query_only_tab_test_set.json")
    elif subtask == "shortest_path_only_map":
        data_file = os.path.join(base_dir, "travelmap", "data", "test_set", "travelmap_shortest_path_query_only_map_test_set.json")
    elif subtask == "shortest_path_map_and_tab_no_constraint":
        data_file = os.path.join(base_dir, "travelmap", "data", "test_set", "travelmap_shortest_path_query_map_and_tab_test_set.json")
    elif subtask == "shortest_path_map_and_tab_with_constraint_1":
        data_file = os.path.join(base_dir, "travelmap", "data", "test_set", "travelmap_shortest_path_query_map_and_tab_with_constraint_1_test_set.json")
    elif subtask == "shortest_path_map_and_tab_with_constraint_2":
        data_file = os.path.join(base_dir, "travelmap", "data", "test_set", "travelmap_shortest_path_query_map_and_tab_with_constraint_2_test_set.json")
    elif subtask == "shortest_path_map_and_tab_with_constraint_3":
        data_file = os.path.join(base_dir, "travelmap", "data", "test_set", "travelmap_shortest_path_query_map_and_tab_with_constraint_3_test_set.json")
    elif subtask == "shortest_path_map_and_tab_with_constraint_4":
        data_file = os.path.join(base_dir, "travelmap", "data", "test_set", "travelmap_shortest_path_query_map_and_tab_with_constraint_4_test_set.json")
    elif subtask == "shortest_path_map_and_tab_with_constraint_1_2_3_4":
        data_file = os.path.join(base_dir, "travelmap", "data", "test_set", "travelmap_shortest_path_query_map_and_tab_with_constraint_1_2_3_4_test_set.json")
    elif subtask == "shortest_path_map_and_tab_with_constraint_1_2_4":
        data_file = os.path.join(base_dir, "travelmap", "data", "test_set", "travelmap_shortest_path_query_map_and_tab_with_constraint_1_2_4_test_set.json")
    elif subtask == "shortest_path_map_and_tab_with_constraint_1_3_4":
        data_file = os.path.join(base_dir, "travelmap", "data", "test_set", "travelmap_shortest_path_query_map_and_tab_with_constraint_1_3_4_test_set.json")
    elif subtask == "shortest_path_map_and_tab_with_constraint_2_3_4":
        data_file = os.path.join(base_dir, "travelmap", "data", "test_set", "travelmap_shortest_path_query_map_and_tab_with_constraint_2_3_4_test_set.json")
    elif subtask == "only_vertex2":
        data_file = os.path.join(base_dir, "travelmap", "data", "test_set", "travelmap_shortest_path_query_map_and_tab_with_constraint_1_2_3_4_only_vertex2_test_set.json")

    # csv消融实验
    elif subtask == "shortest_path_only_csv":
        data_file = os.path.join(base_dir, "travelmap", "data", "test_set", "travelmap_shortest_path_query_only_tab_test_set.json")
    elif subtask == "shortest_path_map_and_csv":
        data_file = os.path.join(base_dir, "travelmap", "data", "test_set", "travelmap_shortest_path_query_map_and_tab_test_set.json")
    elif subtask == "shortest_path_csv_vertex2":
        data_file = os.path.join(base_dir, "travelmap", "data", "test_set", "travelmap_shortest_path_query_map_and_tab_with_constraint_1_2_3_4_only_vertex2_test_set.json")
    elif subtask == "shortest_path_map_and_tab_csv_constraint_1_2_3_4":
        data_file = os.path.join(base_dir, "travelmap", "data", "test_set", "travelmap_shortest_path_query_map_and_tab_with_constraint_1_2_3_4_test_set.json")
    elif subtask == "4_csv_edge_global":
        data_file = os.path.join(base_dir, "travelmap", "qa_data", "travelmap_4_qa_edge_tab_global.json")
    elif subtask == "5_csv_edge_part":
        data_file = os.path.join(base_dir, "travelmap", "qa_data", "travelmap_5_qa_edge_tab_part.json")
    elif subtask == "6_csv_edge_spatial_judge":
        data_file = os.path.join(base_dir, "travelmap", "qa_data", "travelmap_6_qa_edge_tab_spatial_judge.json")
    elif subtask == "7_csv_vertex_global":
        data_file = os.path.join(base_dir, "travelmap", "qa_data", "travelmap_7_qa_vertex_tab_global.json")
    elif subtask == "8_csv_vertex_part":
        data_file = os.path.join(base_dir, "travelmap", "qa_data", "travelmap_8_qa_vertex_tab_part.json")
    elif subtask == "9_csv_vertex_spatial_judge":
        data_file = os.path.join(base_dir, "travelmap", "qa_data", "travelmap_9_qa_vertex_tab_spatial_judge.json")
    elif subtask == "10_csv_and_pic_global":
        data_file = os.path.join(base_dir, "travelmap", "qa_data", "travelmap_10_qa_pic_and_tab_global.json")
    elif subtask == "11_csv_and_pic_part":
        data_file = os.path.join(base_dir, "travelmap", "qa_data", "travelmap_11_qa_pic_and_tab_part.json")
    elif subtask == "12_csv_and_pic_spatial_judge":
        data_file = os.path.join(base_dir, "travelmap", "qa_data", "travelmap_12_qa_pic_and_tab_spatial_judge.json")

    elif subtask == "demo":
        data_file = os.path.join(base_dir, "travelmap", "data", "test_set", "demo_travelmap_shortest_path_query_map_and_tab_with_constraint_1_2_3_4_test_set.json")
    elif subtask == "shortest_path_with_qa_and_constraint_1":
        data_file = os.path.join(base_dir, "travelmap", "data", "test_set", "travelmap_shortest_path_query_map_and_tab_with_constraint_1_test_set.json")
    elif subtask == "shortest_path_with_qa_and_constraint_2":
        data_file = os.path.join(base_dir, "travelmap", "data", "test_set", "travelmap_shortest_path_query_map_and_tab_with_constraint_2_test_set.json")
    elif subtask == "shortest_path_with_qa_and_constraint_3":
        data_file = os.path.join(base_dir, "travelmap", "data", "test_set", "travelmap_shortest_path_query_map_and_tab_with_constraint_3_test_set.json")
    elif subtask == "shortest_path_with_qa_and_constraint_4":
        data_file = os.path.join(base_dir, "travelmap", "data", "test_set", "travelmap_shortest_path_query_map_and_tab_with_constraint_4_test_set.json")
    elif subtask == "shortest_path_with_qa_and_constraint_1_2_3_4":
        data_file = os.path.join(base_dir, "travelmap", "data", "test_set", "travelmap_shortest_path_query_map_and_tab_with_constraint_1_2_3_4_test_set.json")
    elif subtask == "shortest_path_with_qa_and_constraint_1_2_4":
        data_file = os.path.join(base_dir, "travelmap", "data", "test_set", "travelmap_shortest_path_query_map_and_tab_with_constraint_1_2_4_test_set.json")
    elif subtask == "shortest_path_with_qa_and_constraint_1_3_4":
        data_file = os.path.join(base_dir, "travelmap", "data", "test_set", "travelmap_shortest_path_query_map_and_tab_with_constraint_1_3_4_test_set.json")
    elif subtask == "shortest_path_with_qa_and_constraint_2_3_4":
        data_file = os.path.join(base_dir, "travelmap", "data", "test_set", "travelmap_shortest_path_query_map_and_tab_with_constraint_2_3_4_test_set.json")
    
    elif subtask == "1_qa_only_pic_global":
        data_file = os.path.join(base_dir, "travelmap", "qa_data", "travelmap_1_qa_only_pic_global.json")
    elif subtask == "2_qa_only_pic_part":
        data_file = os.path.join(base_dir, "travelmap", "qa_data", "travelmap_2_qa_only_pic_part.json")
    elif subtask == "3_qa_only_pic_spatial_judge":
        data_file = os.path.join(base_dir, "travelmap", "qa_data", "travelmap_3_qa_only_pic_spatial_judge.json")
    elif subtask == "4_qa_edge_tab_global":
        data_file = os.path.join(base_dir, "travelmap", "qa_data", "travelmap_4_qa_edge_tab_global.json")
    elif subtask == "5_qa_edge_tab_part":
        data_file = os.path.join(base_dir, "travelmap", "qa_data", "travelmap_5_qa_edge_tab_part.json")
    elif subtask == "6_qa_edge_tab_spatial_judge":
        data_file = os.path.join(base_dir, "travelmap", "qa_data", "travelmap_6_qa_edge_tab_spatial_judge.json")
    elif subtask == "7_qa_vertex_tab_global":
        data_file = os.path.join(base_dir, "travelmap", "qa_data", "travelmap_7_qa_vertex_tab_global.json")
    elif subtask == "8_qa_vertex_tab_part":
        data_file = os.path.join(base_dir, "travelmap", "qa_data", "travelmap_8_qa_vertex_tab_part.json")
    elif subtask == "9_qa_vertex_tab_spatial_judge":
        data_file = os.path.join(base_dir, "travelmap", "qa_data", "travelmap_9_qa_vertex_tab_spatial_judge.json")
    elif subtask == "10_qa_pic_and_tab_global":
        data_file = os.path.join(base_dir, "travelmap", "qa_data", "travelmap_10_qa_pic_and_tab_global.json")
    elif subtask == "11_qa_pic_and_tab_part":
        data_file = os.path.join(base_dir, "travelmap", "qa_data", "travelmap_11_qa_pic_and_tab_part.json")
    elif subtask == "12_qa_pic_and_tab_spatial_judge":
        data_file = os.path.join(base_dir, "travelmap", "qa_data", "travelmap_12_qa_pic_and_tab_spatial_judge.json")
    
    else:
        raise ValueError(f"Unknown subtask '{subtask}' for task 'travelmap'")
    
    # Determine prompt file based on subtask
    if "with_constraint" in subtask:
        # Extract the constraint part
        constraint_part = subtask.split("with_constraint_")[-1]
        prompt_filename = f"travelmap_shortest_path_with_constraint_{constraint_part}.txt"
    elif "qa_and_constraint" in subtask:
        # Extract the constraint part
        constraint_part = subtask.split("qa_and_constraint_")[-1]
        prompt_filename = f"travelmap_shortest_path_with_qa_and_constraint_{constraint_part}.txt"
    elif "only_tab" in subtask or "only_csv" in subtask:
        prompt_filename = "travelmap_shortest_path_only_tab.txt"
    elif "only_map" in subtask:
        prompt_filename = "travelmap_shortest_path_only_map.txt"
    elif "no_constraint" in subtask or "map_and_csv" in subtask:
        prompt_filename = "travelmap_shortest_path_map_and_tab.txt"
    elif "only_vertex2" in subtask:
        prompt_filename = "travelmap_shortest_path_with_constraint_1_2_3_4_only_vertex2.txt"

    elif "csv_vertex2" in subtask:
        prompt_filename = "travelmap_shortest_path_with_constraint_1_2_3_4_only_vertex2.txt"
    elif "csv_constraint" in subtask:
        prompt_filename = "travelmap_shortest_path_with_constraint_1_2_3_4.txt"
    elif "4_csv_edge_global" in subtask:
        prompt_filename = "travelmap_4_qa_edge_tab_global.txt"
    elif "5_csv_edge_part" in subtask:
        prompt_filename = "travelmap_5_qa_edge_tab_part.txt"
    elif "6_csv_edge_spatial_judge" in subtask:
        prompt_filename = "travelmap_6_qa_edge_tab_spatial_judge.txt"
    elif "7_csv_vertex_global" in subtask:
        prompt_filename = "travelmap_7_qa_vertex_tab_global.txt"
    elif "8_csv_vertex_part" in subtask:
        prompt_filename = "travelmap_8_qa_vertex_tab_part.txt"
    elif "9_csv_vertex_spatial_judge" in subtask:
        prompt_filename = "travelmap_9_qa_vertex_tab_spatial_judge.txt"
    elif "10_csv_and_pic_global" in subtask:
        prompt_filename = "travelmap_10_qa_pic_and_tab_global.txt"
    elif "11_csv_and_pic_part" in subtask:
        prompt_filename = "travelmap_11_qa_pic_and_tab_part.txt"
    elif "12_csv_and_pic_spatial_judge" in subtask:
        prompt_filename = "travelmap_12_qa_pic_and_tab_spatial_judge.txt"

    elif "demo" in subtask:
        prompt_filename = "demo.txt"

    elif "1_qa_only_pic_global" in subtask:
        prompt_filename = "travelmap_1_qa_only_pic_global.txt"
    elif "2_qa_only_pic_part" in subtask:
        prompt_filename = "travelmap_2_qa_only_pic_part.txt"
    elif "3_qa_only_pic_spatial_judge" in subtask:
        prompt_filename = "travelmap_3_qa_only_pic_spatial_judge.txt"
    elif "4_qa_edge_tab_global" in subtask:
        prompt_filename = "travelmap_4_qa_edge_tab_global.txt"
    elif "5_qa_edge_tab_part" in subtask:
        prompt_filename = "travelmap_5_qa_edge_tab_part.txt"
    elif "6_qa_edge_tab_spatial_judge" in subtask:
        prompt_filename = "travelmap_6_qa_edge_tab_spatial_judge.txt"
    elif "7_qa_vertex_tab_global" in subtask:
        prompt_filename = "travelmap_7_qa_vertex_tab_global.txt"
    elif "8_qa_vertex_tab_part" in subtask:
        prompt_filename = "travelmap_8_qa_vertex_tab_part.txt"
    elif "9_qa_vertex_tab_spatial_judge" in subtask:
        prompt_filename = "travelmap_9_qa_vertex_tab_spatial_judge.txt"
    elif "10_qa_pic_and_tab_global" in subtask:
        prompt_filename = "travelmap_10_qa_pic_and_tab_global.txt"
    elif "11_qa_pic_and_tab_part" in subtask:
        prompt_filename = "travelmap_11_qa_pic_and_tab_part.txt"
    elif "12_qa_pic_and_tab_spatial_judge" in subtask:
        prompt_filename = "travelmap_12_qa_pic_and_tab_spatial_judge.txt"

    elif "no_constraint" in subtask:
        prompt_filename = "travelmap_shortest_path_map_and_tab.txt"    
    else:
        raise ValueError(f"Cannot determine prompt file for subtask '{subtask}'")
    
    prompt_file = os.path.join(base_dir, "travelmap", "prompts", prompt_filename)
    
    # 检查数据文件是否存在
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    print(f"Loading data from: {data_file}")
    
    # 加载数据
    with open(data_file, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    
    # Load prompt content and update queries
    prompt_content=""
    if os.path.exists(prompt_file):
        print(f"Loading prompt from: {prompt_file}")
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_content = f.read().strip()
    else:
        print(f"Warning: Prompt file not found: {prompt_file}")
        assert(0)
    #print(prompt_content)
    queries=[]
    # travelmap 的 subtask 选项
    if subtask == "shortest_path_only_tab":
        for item in datas:
            queries.append([
                ("text",prompt_content.format(question=item["question"])),
                ("text","This is an edge table of a scenic area planning map."),
                ("json",WORKSPACE_DIR+"/"+item['edge_tab']), 
            ]
            )

    elif subtask == "shortest_path_only_map":
        for item in datas:
            queries.append([
                ("text",prompt_content.format(question=item["question"])),
                ("text","This is the scenic area planning map image."),
                ("image",WORKSPACE_DIR+"/"+item['figure']),
            ]
            )

    elif subtask == "shortest_path_map_and_tab_no_constraint":
        for item in datas:
            queries.append([
                ("text",prompt_content.format(question=item["question"])),
                ("text","This is an edge table of a scenic area planning map."),
                ("json",WORKSPACE_DIR+"/"+item['edge_tab']),
                ("text","This is the scenic area planning map image."),
                ("image",WORKSPACE_DIR+"/"+item['figure']),
            ]
            )

    elif subtask == "only_vertex2":
        for item in datas:
            question = item["question"]
            w1, w2, w3, w4= item["weights"]
            queries.append([
                ("text",prompt_content.format(question=question, w1=w1, w2=w2, w3=w3, w4=w4)),
                ("text","This is a vertex table of a scenic area planning map."),
                ("json",WORKSPACE_DIR+"/"+item['vertex_tab']),
                ("text","This is the scenic area planning map image."),
                ("image",WORKSPACE_DIR+"/"+item['figure']),
            ]
            )



    # csv消融实验
    elif "only_csv" in subtask:
        for item in datas:
            edge_csv_path = item["edge_tab"].replace(".json", ".csv")
            queries.append([
                ("text",prompt_content.format(question=item["question"])),
                ("text","This is an edge table of a scenic area planning map."),
                ("csv",WORKSPACE_DIR+"/"+edge_csv_path), 
            ]
            )
    elif "map_and_csv" in subtask:
        for item in datas:
            edge_csv_path = item["edge_tab"].replace(".json", ".csv")
            queries.append([
                ("text",prompt_content.format(question=item["question"])),
                ("text","This is an edge table of a scenic area planning map."),
                ("csv",WORKSPACE_DIR+"/"+edge_csv_path),
                ("text","This is the scenic area planning map image."),
                ("image",WORKSPACE_DIR+"/"+item['figure']),
            ]
            )
    elif "csv_vertex2" in subtask:
        for item in datas:
            question = item["question"]
            w1, w2, w3, w4= item["weights"]
            vertex_csv_path = item["vertex_tab"].replace(".json", ".csv")
            queries.append([
                ("text",prompt_content.format(question=question, w1=w1, w2=w2, w3=w3, w4=w4)),
                ("text","This is a vertex table of a scenic area planning map."),
                ("csv",WORKSPACE_DIR+"/"+vertex_csv_path),
                ("text","This is the scenic area planning map image."),
                ("image",WORKSPACE_DIR+"/"+item['figure']),
            ]
            )
    elif "csv_constraint" in subtask:
        for item in datas:
            question = item["question"]
            w1, w2, w3, w4= item["weights"]
            edge_csv_path = item["edge_tab"].replace(".json", ".csv")
            vertex_csv_path = item["vertex_tab"].replace(".json", ".csv")
            queries.append([
                ("text",prompt_content.format(question=question, w1=w1, w2=w2, w3=w3, w4=w4)),
                ("text","This is an edge table of a scenic area planning map."),
                ("csv",WORKSPACE_DIR+"/"+edge_csv_path),
                ("text","This is a vertex table of a scenic area planning map."),
                ("csv",WORKSPACE_DIR+"/"+vertex_csv_path),
                ("text","This is the scenic area planning map image."),
                ("image",WORKSPACE_DIR+"/"+item['figure']),
            ]
            )
    elif "csv_edge" in subtask:
        for item in datas:
            question = item["question"]
            edge_csv_path = item["edge_tab"].replace(".json", ".csv")
            queries.append([
                ("text",prompt_content.format(question=question)),
                ("text","This is an edge table of a scenic area planning map."),
                ("csv",WORKSPACE_DIR+"/"+edge_csv_path),
            ]
            )
    elif "csv_vertex" in subtask:
        for item in datas:
            question = item["question"]
            vertex_csv_path = item["vertex_tab"].replace(".json", ".csv")
            queries.append([
                ("text",prompt_content.format(question=question)),
                ("text","This is a vertex table of a scenic area planning map."),
                ("csv",WORKSPACE_DIR+"/"+vertex_csv_path),
            ]
            )
    elif "csv_and_pic" in subtask:
        for item in datas:
            question = item["question"]
            vertex_csv_path = item["vertex_tab"].replace(".json", ".csv")
            queries.append([
                ("text",prompt_content.format(question=question)),
                ("text","This is a vertex table of a scenic area planning map."),
                ("csv",WORKSPACE_DIR+"/"+vertex_csv_path),
                ("text","This is the scenic area planning map image."),
                ("image",WORKSPACE_DIR+"/"+item['figure']),
            ]
            )



    elif subtask == "demo":
        for item in datas:
            question = item["question"]
            w1, w2, w3, w4= item["weights"]
            queries.append([
                ("text",prompt_content.format(question=question, w1=w1, w2=w2, w3=w3, w4=w4)),
                ("text","This is an edge table of a scenic area planning map."),
                ("json",WORKSPACE_DIR+"/"+item['edge_tab']),
                ("text","This is a vertex table of a scenic area planning map."),
                ("json",WORKSPACE_DIR+"/"+item['vertex_tab']),
                ("text","This is the scenic area planning map image."),
                ("image",WORKSPACE_DIR+"/"+item['figure']),
            ]
            )

    elif "only_pic" in subtask:
        for item in datas:
            question = item["question"]
            queries.append([
                ("text",prompt_content.format(question=question)),
                ("text","This is the scenic area planning map image."),
                ("image",WORKSPACE_DIR+"/"+item['figure']),
            ]
            )

    elif "edge_tab" in subtask:
        for item in datas:
            question = item["question"]
            queries.append([
                ("text",prompt_content.format(question=question)),
                ("text","This is an edge table of a scenic area planning map."),
                ("json",WORKSPACE_DIR+"/"+item['edge_tab']),
            ]
            )

    elif "vertex_tab" in subtask:
        for item in datas:
            question = item["question"]
            queries.append([
                ("text",prompt_content.format(question=question)),
                ("text","This is a vertex table of a scenic area planning map."),
                ("json",WORKSPACE_DIR+"/"+item['vertex_tab']),
            ]
            )

    elif "pic_and_tab" in subtask:
        for item in datas:
            question = item["question"]
            queries.append([
                ("text",prompt_content.format(question=question)),
                ("text","This is a vertex table of a scenic area planning map."),
                ("json",WORKSPACE_DIR+"/"+item['vertex_tab']),
                ("text","This is the scenic area planning map image."),
                ("image",WORKSPACE_DIR+"/"+item['figure']),
            ]
            )
    
    elif "qa_and_constraint" in subtask:
        for item in datas:
            question = item["question"]
            w1, w2, w3, w4= item["weights"]
            if subtask =="shortest_path_with_qa_and_constraint_1_2_3_4":
                queries.append([
                    ("text",prompt_content.format(question=question, w1=w1, w2=w2, w3=w3,w4=w4)),
                    ("text","This is an edge table of a scenic area planning map."),
                    ("json",WORKSPACE_DIR+"/"+item['edge_tab']),
                    ("text","This is a vertex table of a scenic area planning map."),
                    ("json",WORKSPACE_DIR+"/"+item['vertex_tab']),
                    ("text","This is the scenic area planning map image."),
                    ("image",WORKSPACE_DIR+"/"+item['figure']),
                ]
                )

            elif subtask=="shortest_path_with_qa_and_constraint_1_2_4":
                queries.append([
                    ("text",prompt_content.format(question=question, w1=w1, w2=w2, w4=w4)),
                    ("text","This is an edge table of a scenic area planning map."),
                    ("json",WORKSPACE_DIR+"/"+item['edge_tab']),
                    ("text","This is a vertex table of a scenic area planning map."),
                    ("json",WORKSPACE_DIR+"/"+item['vertex_tab']),
                    ("text","This is the scenic area planning map image."),
                    ("image",WORKSPACE_DIR+"/"+item['figure']),
                ]
                )

            elif subtask=="shortest_path_with_qa_and_constraint_1_3_4":
                queries.append([
                    ("text",prompt_content.format(question=question, w1=w1, w3=w3, w4=w4)),
                    ("text","This is an edge table of a scenic area planning map."),
                    ("json",WORKSPACE_DIR+"/"+item['edge_tab']),
                    ("text","This is a vertex table of a scenic area planning map."),
                    ("json",WORKSPACE_DIR+"/"+item['vertex_tab']),
                    ("text","This is the scenic area planning map image."),
                    ("image",WORKSPACE_DIR+"/"+item['figure']),
                ]
                )

            elif subtask=="shortest_path_with_qa_and_constraint_2_3_4":
                queries.append([
                    ("text",prompt_content.format(question=question, w2=w2, w3=w3, w4=w4)),
                    ("text","This is an edge table of a scenic area planning map."),
                    ("json",WORKSPACE_DIR+"/"+item['edge_tab']),
                    ("text","This is a vertex table of a scenic area planning map."),
                    ("json",WORKSPACE_DIR+"/"+item['vertex_tab']),
                    ("text","This is the scenic area planning map image."),
                    ("image",WORKSPACE_DIR+"/"+item['figure']),
                ]
                )

            else:
                queries.append([
                    ("text",prompt_content.format(question=question)),
                    ("text","This is an edge table of a scenic area planning map."),
                    ("json",WORKSPACE_DIR+"/"+item['edge_tab']),
                    ("text","This is a vertex table of a scenic area planning map."),
                    ("json",WORKSPACE_DIR+"/"+item['vertex_tab']),
                    ("text","This is the scenic area planning map image."),
                    ("image",WORKSPACE_DIR+"/"+item['figure']),
                ]
                )
    else:
        for item in datas:
            question = item["question"]
            w1, w2, w3, w4= item["weights"]
            if subtask =="shortest_path_map_and_tab_with_constraint_1_2_3_4":
                queries.append([
                    ("text",prompt_content.format(question=question, w1=w1, w2=w2, w3=w3,w4=w4)),
                    ("text","This is an edge table of a scenic area planning map."),
                    ("json",WORKSPACE_DIR+"/"+item['edge_tab']),
                    ("text","This is a vertex table of a scenic area planning map."),
                    ("json",WORKSPACE_DIR+"/"+item['vertex_tab']),
                    ("text","This is the scenic area planning map image."),
                    ("image",WORKSPACE_DIR+"/"+item['figure']),
                ]
                )

            elif subtask=="shortest_path_map_and_tab_with_constraint_1_2_4":
                queries.append([
                    ("text",prompt_content.format(question=question, w1=w1, w2=w2, w4=w4)),
                    ("text","This is an edge table of a scenic area planning map."),
                    ("json",WORKSPACE_DIR+"/"+item['edge_tab']),
                    ("text","This is a vertex table of a scenic area planning map."),
                    ("json",WORKSPACE_DIR+"/"+item['vertex_tab']),
                    ("text","This is the scenic area planning map image."),
                    ("image",WORKSPACE_DIR+"/"+item['figure']),
                ]
                )

            elif subtask=="shortest_path_map_and_tab_with_constraint_1_3_4":
                queries.append([
                    ("text",prompt_content.format(question=question, w1=w1, w3=w3, w4=w4)),
                    ("text","This is an edge table of a scenic area planning map."),
                    ("json",WORKSPACE_DIR+"/"+item['edge_tab']),
                    ("text","This is a vertex table of a scenic area planning map."),
                    ("json",WORKSPACE_DIR+"/"+item['vertex_tab']),
                    ("text","This is the scenic area planning map image."),
                    ("image",WORKSPACE_DIR+"/"+item['figure']),
                ]
                )

            elif subtask=="shortest_path_map_and_tab_with_constraint_2_3_4":
                queries.append([
                    ("text",prompt_content.format(question=question, w2=w2, w3=w3, w4=w4)),
                    ("text","This is an edge table of a scenic area planning map."),
                    ("json",WORKSPACE_DIR+"/"+item['edge_tab']),
                    ("text","This is a vertex table of a scenic area planning map."),
                    ("json",WORKSPACE_DIR+"/"+item['vertex_tab']),
                    ("text","This is the scenic area planning map image."),
                    ("image",WORKSPACE_DIR+"/"+item['figure']),
                ]
                )

            else:
                queries.append([
                    ("text",prompt_content.format(question=question)),
                    ("text","This is an edge table of a scenic area planning map."),
                    ("json",WORKSPACE_DIR+"/"+item['edge_tab']),
                    ("text","This is a vertex table of a scenic area planning map."),
                    ("json",WORKSPACE_DIR+"/"+item['vertex_tab']),
                    ("text","This is the scenic area planning map image."),
                    ("image",WORKSPACE_DIR+"/"+item['figure']),
                ]
                )
            #print(queries[0])
            #assert(0)

    print(f"Loaded {len(datas)} queries")
    
    return queries, datas


def test_build_travelmap_queries():
    """测试 build_travelmap_queries 函数"""
    print("=" * 60)
    print("测试 build_travelmap_queries 函数")
    print("=" * 60)
    
    # 获取当前脚本所在目录的上级目录作为 base_dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    
    print(f"Base directory: {base_dir}")
    print()
    
    # 测试用例列表
    test_cases = [
        ("travelmap", "shortest_path_only_tab"),
    ]
    
    for task, subtask in test_cases:
        print(f"测试: task='{task}', subtask='{subtask}'")
        queries, datas = build_travelmap_queries(task, subtask, base_dir)
        print(queries[0])
    
    print("测试完成!")


if __name__ == "__main__":
    test_build_travelmap_queries()

