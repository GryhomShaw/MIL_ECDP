import cv2, os, threadpool
import numpy as np
import openslide as opsl
import json
import sys
sys.path.append('../')
from config import cfg
from eic_utils import cp
import warnings
warnings.filterwarnings("ignore")
def work(p):# {{{
    img_path, out_path, idx, size, scale, threshold = p

    tiff_path = img_path[:-9]+'.tif' #1.tif
    mask_size = size // 4

    step = mask_size // 2

    img = cv2.imread(img_path,0)
    h, w = img.shape[:2]


    try:
        os.makedirs(out_path)
    except:
        pass
    data={}
    data['roi']=[]
    for i in range(0, h, step):
        for j in range(0, w, step):
            si = min(i, h - mask_size)
            sj = min(j, w - mask_size)
            si = max(0,si) #有可能h比size还要小
            sj = max(0,sj)

            sub_img = img[si: si+mask_size, sj: sj+mask_size]
            # print(tiff_path)
            if np.sum(sub_img) // 255 > sub_img.shape[0] * sub_img.shape[1] * threshold:
                slide = opsl.OpenSlide(tiff_path)
                [n,m] = slide.dimensions
                x = min(scale*si, m - size)
                y = min(scale*sj, n - size)
                #print(x,y)
                data['roi'].append([x,y])
                patch = np.array(slide.read_region((y,x),0,(size,size)).convert('RGB'))
                patch_name = "{}_{}.jpg".format(x,y)
                out_patch_path =  os.path.join(out_path,patch_name)
                cv2.imwrite(out_patch_path,patch)
                cp('(#y)save_img:\t{}(#)'.format(out_patch_path))
            if sj != j:
                break

        if si != i:
            break
    json_path = img_path[:-9]+'_mask.json'
    data['id'] = img_path.split('/')[-2]
    with open(json_path, 'w') as f :
        json.dump(data, f)
    cp('(#g)save_josn:\t{}(#)'.format(json_path))

# }}}


def cut(args):
    params = []
    idx = 0
    for root, dirs, filenames in os.walk(cfg.data_append_path):
        for each_mask in filenames:
            if '_mask' in each_mask:
                path = os.path.join(root,each_mask) #./EDCP/data_append/1/1_mask.jpg
                out_path = os.path.join(cfg.patch_data,each_mask.split('_')[0]) #./EDCP_PATCH/1/
                idx+=1
                params.append([path,out_path,idx,args.patchsize,args.scale,args.threshold])

    #print(idx)
    cp('(#b)total_img:\t{}(#)'.format(idx))
    pool = threadpool.ThreadPool(args.poolsize)
    requests = threadpool.makeRequests(work, params)
    [pool.putRequest(req) for req in requests]
    pool.wait()


