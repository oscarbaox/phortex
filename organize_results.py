import re
import numpy as np
import os
import json

pattern = re.compile(r'\"height\": [0-9]*')
pattern2 = re.compile(r'(?<=simulations\/).*(?=\/params)')
root_dir = '.'

sim_list = []
for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        filepath = os.path.join(dirpath, filename)
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if pattern.search(line):
                        print(filepath)
                        sim_name = pattern2.search(filepath)
                        if sim_name:
                            sim_list.append(sim_name.group())
                        break  # match found, move to next file
        except (OSError, UnicodeDecodeError):
            continue  # skip unreadable files
print(sim_list)
print(len(sim_list))

count = 0
results = {}

for name in sim_list:
    simulations_path = f"./output/simulations/{name}/"
    planning_path = f"./output/planning/{name}_metaloop1/"
    metrics_path = planning_path + f"metrics/"
    with open(simulations_path+f"params.json","r",encoding="utf-8") as f:
        params = json.load(f)
    results[name] = {}
    results[name]["params"] = params
    chains_idx = [0,1,2]
    chain_metrics = []
    for idx in chains_idx:
        with open(metrics_path+f"metric_results_chain{idx}_.json","r",encoding="utf-8") as f:
            chain_metrics.append(json.load(f))
    results[name]["metrics_list"] = chain_metrics
    count += 1
    print(count)

with open("./output/collected_sim_result.json","w",encoding='utf-8') as f_out:
    json.dump(results,f_out,indent=4)