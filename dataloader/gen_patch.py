from tiff2jepg import tiff2jpeg
from cut_image import cut
import argparse
import os
import numpy as np
import random
import  sys
sys.path.append('../')
from config import cfg
from eic_utils import cp,procedure

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--size', type=int, default=20000, help="size of cut")
    parse.add_argument('--patchsize', type=int, default=2048, help="size of patch")
    parse.add_argument('--scale', type=int, default=4, help="scale of resize")
    parse.add_argument('--poolsize', type=int, default=32, help="size of pool")
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


if __name__ == '__main__':
    args = get_args()
    
    # with procedure("tiff2jpeg... ") as p3:
    with procedure ('rename tif'):
        rename()
    with procedure ('convert tif to mask'):
        tiff2jpeg(args)
    with procedure ('cut images'):
        cut(args)
    with procedure ('move and split data '):
        pass





