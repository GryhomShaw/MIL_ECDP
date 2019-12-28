import  numpy as np
import os
import sys
import PIL
from PIL import Image
import json
sys.path.append('../')
from config import  cfg

PIL.Image.MAX_IMAGE_PIXELS = 933120000
with open(cfg.data_split,'r') as f:
    data = json.load(f)
img_path = []
for each_img in data['train_pos'] + data['train_neg']:
    img_path.extend( [ os.path.join(each_img,each_patch)for each_patch in os.listdir(each_img)])
print(len(img_path))
e = []
e2 = []
pix = []

for idx, each_img_path in enumerate(img_path):
    img = Image.open(each_img_path)
    print(idx,img.size,each_img_path)

    img = np.array(img)
    img = img / 255 if (np.max(img) > 1) else img
    pix += np.prod(img.shape[:2])
    e.append(img.astype('float32').reshape(-1,3).sum(axis=0))
    e2.append((img.astype('float32')**2).reshape(-1,3).sum(axis=0))

e = np.array(e).sum(axis=0) / pix
e2 = np.array(e2).sum(axis=0) / pix

mean = e
std = np.sqrt(np.array(e2) - np.array(e) ** 2)
print(mean, std)