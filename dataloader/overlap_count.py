import os
import numpy as np
import json
import sys
sys.path.append('../')
from config import cfg

overlap_path = cfg.patch_overlap
count_range = np.arange(10)/10
for each_json in os.listdir(overlap_path)[:1]:
    json_path = os.path.join(overlap_path,each_json)
    with open(json_path, 'r') as f:
        data = np.array(json.load(f))
        for each_range in count_range[-2:]:
            index = np.array(np.where(data[:,2] > each_range)).squeeze(0)
            print(each_json,index)
            print(data[index])