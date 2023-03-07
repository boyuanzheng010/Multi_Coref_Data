import pickle as pkl
import csv
import json
import os
import jsonlines
from utils import merge_head_sharing_np, extract_scene_constituency, extract_scene_constituency_chinese_align

source_data = {}
input_dir = "data/parsed_data/zh"
for file_name in os.listdir(input_dir):
    file_path = os.path.join(input_dir, file_name)
    with open(file_path, 'rb') as f:
        temp = pkl.load(f)
        scene_id = file_name.strip().split(".")[0]
        # print(scene_id)
        source_data[scene_id] = temp

output = []
for scene_id in source_data:
    # if scene_id not in ["s09e09c13t", "s01e14c07f", "s09e03c12t"]:
    # if scene_id not in ["s09e09c13t"]:
    #     continue
    if scene_id not in ["s01e13c07f"]:
        continue
    print(scene_id)
    parsed_scene = source_data[scene_id]
    output.append(extract_scene_constituency_chinese_align(parsed_scene, scene_id))

with open("pilot_zh_test_mutli_coref_input_align_s01e13c07f.csv", "w", encoding="utf-8") as csv_fh:
    fieldnames = ['json_data']
    writer = csv.DictWriter(csv_fh, fieldnames, lineterminator='\n')
    writer.writeheader()
    for line in output:
        writer.writerow({'json_data': json.dumps(line)})


















