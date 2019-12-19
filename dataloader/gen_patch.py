import argparse
import os
import numpy as np
import random
import  sys
sys.path.append('../')
from config import cfg
from eic_utils import cp,procedure
import openslide as opsl
import threadpool
import pandas as pd
import math
import cv2
import json
def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--size', type=int, default=20000, help="size of cut")
    parse.add_argument('--patch_size', type=int, default=2048, help="size of patch")
    parse.add_argument('--scale', type=int, default=4, help="scale of resize")
    parse.add_argument('--poolsize', type=int, default=16, help="size of pool")
    parse.add_argument('--threshold', type=float, default=0.5, help="size of patch")
    return parse.parse_args()

def rename():
    for root, dirs, filenames in os.walk(cfg.data_append_path):
        for each_tiff in filenames:
            if '.tif' in each_tiff and '.enp' not in each_tiff :
                img_path = os.path.join(root, each_tiff)
                new_path = os.path.join(root, each_tiff.split('_')[0].replace('.tif','')+'.tif')

                if img_path == new_path:
                    continue
                os.rename(img_path, new_path)
                cp('(#r){}(#)\t(#g){}(#)'.format(img_path, new_path))

def work(args):
    img_path, size, scale, output_patch_path, patch_size, threshold = args
    '''
    img_path: path of tif  (e.g. ./data_append/1/1.tif)
    size: size of patch (from tiff to jpeg) (e.g. 20000)
    scale: scale (riff2jpeg)  (e.g. 4)
    output_patch_path: path of patch (e.g. ./Patch/pos/1)
    patch_size: during cut_image (2048)
    '''
    output_mask_path = img_path[:-4] + '_mask.jpg'
    try:
        slide = opsl.OpenSlide(img_path)
    except:
        pass
    with procedure('Tiff2jpeg'):
        if not os.path.isfile(output_mask_path):
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
            cv2.imwrite(output_mask_path, mask)
            cp('(#g)save_mask_path:{}(#)'.format(output_mask_path))
    with procedure('Cut image'):
        mask = cv2.imread(output_mask_path,0)
        assert  len(mask.shape) == 2 ,print('size error')
        mask_patch_size = patch_size // scale
        step = mask_patch_size // 2
        try:
            os.makedirs(output_patch_path)
        except:
            pass
        data = {}
        data['roi'] = []
        h, w =mask.shape[0], mask.shape[1]
        for i in range(0, h, step):
            for j in range(0, w, step):
                si = min(i, h - mask_patch_size)
                sj = min(j, w - mask_patch_size)
                si = max(0, si)  # 有可能h比size还要小
                sj = max(0, sj)

                sub_img = mask[si: si + mask_patch_size, sj: sj + mask_patch_size]
                if np.sum(sub_img) // 255 > sub_img.shape[0] * sub_img.shape[1] * threshold:
                    # slide = opsl.OpenSlide(tiff_path)
                    [n, m] = slide.dimensions
                    x = min(scale * si, m - patch_size)
                    y = min(scale * sj, n - patch_size)
                    #print( x, y)
                    data['roi'].append([x, y])
                    patch = np.array(slide.read_region((y, x), 0, (patch_size, patch_size)).convert('RGB'))
                    patch_name = "{}_{}.jpg".format(x, y)
                    patch_path = os.path.join(output_patch_path, patch_name)
                    cv2.imwrite(patch_path, patch)
                    cp('(#y)save_path:\t{}(#)'.format(patch_path))
                if sj != j:
                    break

            if si != i:
                break
        json_path = output_mask_path[:-9] + '_mask.json'
        data['id'] = img_path.split('/')[-2]
        with open(json_path, 'w') as f:
            json.dump(data, f)
        cp('(#g)save_josn:\t{}(#)'.format(json_path))


if __name__ == '__main__':
    args = get_args()
    rename()
    df = pd.read_excel(cfg.label_path)
    labels = df.values
    m = {}
    for val in labels:
        m[val[0]] = val[1]
    params = []
    idx = 0
    for root, dirs, filenames in os.walk(cfg.data_neg_path):
        for each_tif in filenames:
            if '.tif' in each_tif:
                name = each_tif.split('.')[0]
                flag = 'pos' if m[int(name)] == 'Positive' else 'neg'
                path = os.path.join(root, each_tif)  # ./EDCP/data_append/1/1.tif
                out_patch_path = os.path.join(cfg.patch_data, flag, name)  # ./EDCP_PATCH/pos/1/
                idx += 1
                params.append([path, args.size, args.scale, out_patch_path, args.patch_size, args.threshold])

    # print(idx)
    cp('(#b)total_img:\t{}(#)'.format(idx))
    pool = threadpool.ThreadPool(args.poolsize)
    requests = threadpool.makeRequests(work, params)
    [pool.putRequest(req) for req in requests]
    pool.wait()








