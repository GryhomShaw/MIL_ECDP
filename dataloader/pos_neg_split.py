import numpy as np
import pandas as pd
import os
import shutil
import sys
sys.path.append('../')
from config import cfg
data_path = cfg.data_appen_path
xslx_path = cfg.label_path

output_path = {}
for each in ['pos','neg']:
    output_path[each] = os.path.join(data_path,each)
    if not os.path.isdir(output_path[each]):
        os.makedirs(output_path[each])

df = pd.read_excel(xslx_path)
labels = df.values

tifs = [each for each in os.listdir(data_path) if 'pos' not in each
        and 'neg' not in each]


print(tifs)
for idx in tifs:
    src = os.path.join(data_path, idx)
    if labels[int(idx)-1][1] == 'Positive':
        dst = os.path.join(output_path['pos'])
    else :
        dst = os.path.join(output_path['neg'])
    shutil.move(src,dst)
    print(idx, labels[int(idx)-1][1], src, dst)



