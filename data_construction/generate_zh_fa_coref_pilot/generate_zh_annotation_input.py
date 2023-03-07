import pickle as pkl
from utils import merge_head_sharing_np, extract_scene_constituency
import csv
import json
import os


source_data = {}
input_dir = "data/parsed_data/zh"
for file_name in os.listdir(input_dir):
    file_path = os.path.join(input_dir, file_name)
    with open(file_path, 'rb') as f:
        temp = pkl.load(f)
        scene_id = file_name.strip().split(".")[0]
        source_data[scene_id] = temp

output = []
for scene_id in source_data:
    print(scene_id)
    parsed_scene = source_data[scene_id]
    output.append(extract_scene_constituency(parsed_scene, scene_id))

# with open("all_zh_test_mutli_coref_input.csv", "w", encoding="utf-8") as csv_fh:
#     fieldnames = ['json_data']
#     writer = csv.DictWriter(csv_fh, fieldnames, lineterminator='\n')
#     writer.writeheader()
#     for line in output:
#         writer.writerow({'json_data': json.dumps(line)})

# for item in output[0]:
#     print(item)
#     print(output[0][item])

















