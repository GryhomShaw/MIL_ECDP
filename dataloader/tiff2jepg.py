import os
import openslide  as opsl
import numpy as np
import math
import cv2
import threadpool
import sys
sys.path.append('../')
from config import cfg
from eic_utils import cp
import warnings
warnings.filterwarnings("ignore")
def work(args):
    img_path,size,scale = args
    output_path = img_path[:-4] + '_mask.jpg'
    try:
        slide = opsl.OpenSlide(img_path)
    except:
        print("processing:\t{}".format(img_path))
        pass
    [n, m] = slide.dimensions
    blocks_pre_col = math.ceil(m / size)
    blocks_pre_row = math.ceil(n / size)
    row_cache = []
    img_cache = []
    for i in range(blocks_pre_col):
        for j in range(blocks_pre_row):
            x = i * size
            y = j * size
            height = min(x + size, m) - x
            width = min(y + size, n) - y
            img = np.array(slide.read_region((y, x), 0, (width, height)))
            img = cv2.resize(img, (width // scale, height // scale))
            row_cache.append(img)
        img_cache.append(np.concatenate(row_cache, axis=1))
        row_cache = []
    img = np.concatenate(img_cache, axis=0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    mask = 255 - th1
    cv2.imwrite(output_path, mask)
    cp('(#g)save_path:{}(#)'.format(output_path))



def tiff2jpeg(args):
    index=0
    params=[]
    for root, dirs, filenames in os.walk(cfg.data_append_path):
        for each_tiff in filenames:
            if '.tif' in each_tiff and '.enp' not in each_tiff :
                img_path = os.path.join(root, each_tiff)
                params.append([img_path, args.size, args.scale])
                index += 1
    cp('(#b)total_img:\t{}(#)'.format(index))
    pool = threadpool.ThreadPool(args.poolsize)
    requests = threadpool.makeRequests(work, params)
    [pool.putRequest(req) for req in requests]
    pool.wait()


