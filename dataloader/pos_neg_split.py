import numpy as np
import pandas as pd
import os
import shutil
import sys
sys.path.append('../')
from config import cfg
data_path = cfg.data_append_path
xslx_path = cfg.label_path


df = pd.read_excel(xslx_path)
labels = df.values
m = {}
for val in labels:
    m[val[0]] = val[1]
tifs = [each for each in os.listdir(data_path) if 'pos' not in each
        and 'neg' not in each]


print(tifs)
for idx in tifs:
    src = os.path.join(data_path, idx)
    if m[int(idx)] == 'Positive':
        dst = cfg.data_pos_path
    else :
        dst = cfg.data_neg_path
    shutil.move(src,dst)
    print(idx, m[int(idx)], src, dst)



