import sys
import os
import numpy as np
import argparse
import random
import openslide
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from config import cfg
from Dataset import MILdataset
import shutil
import json
import cv2
from eic_utils import cp, procedure
from utils.summary import TensorboardSummary

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str, default='./train_output/checkpoint_best.pth', help='path to pretrained model')
    parser.add_argument('--batch_size', type=int, default=100, help='how many images to sample per slide (default: 100)')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--patch_size', default=1024, type=int, help='size of patch')
    parser.add_argument('--k', default=5, type=int, help='topk of patch')
    parser.add_argument('--resume',default=True, type=bool, help="load model")
    return parser.parse_args()
args = get_args()
def main():

    #load model
    with procedure('load model'):
        model = models.resnet34(True)
        model.fc = nn.Linear(model.fc.in_features, cfg.n_classes)
        model = torch.nn.DataParallel(model.cuda())
        if args.resume:
            ch = torch.load(args.model)
            model.load_state_dict(ch['state_dict'])
        cudnn.benchmark = True


    with procedure('prepare dataset'):
        # normalization
        normalize = transforms.Normalize(mean=cfg.mean,std=cfg.std)
        trans = transforms.Compose([transforms.ToTensor(),normalize])
        #load data
        with open(cfg.data_split) as f:
            data = json.load(f)
        dset = MILdataset( data ['train_pos'][:2], args.patch_size, trans)
        loader = torch.utils.data.DataLoader(
            dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    dset.setmode(1)
    probs = inference(loader, model)
    maxs,index = group_topk(np.array(dset.slideIDX), probs, args.k)
    if not os.path.isdir(cfg.color_img_path):
        os.makedirs(cfg.color_img_path)
    for name, target, probs, idxs in zip(dset.slidenames, dset.targets, maxs, index):
        assert  len(dset.slidenames) == len(maxs) ,print("length, error")
        flag = 'pos' if target == 1 else 'neg'
        orign_img_path = os.path.join(cfg.data_path,flag,name,name+'_orign.jpg')
        color_img_path = os.path.join(cfg.color_img_path,name+'.jpg')
        #print("orign_img_path: ",orign_img_path)
        patch_names = []
        orign_img = cv2.imread(orign_img_path)
        for i in range(args.k):
            idx = idxs[i]
            src = dset.grid[idx]
            dst = os.path.join(cfg.patch_predict, flag, src.split('/')[-2])
            if not os.path.isdir(dst):
                os.makedirs(dst)
            shutil.copy(src,dst)
            cp('(#r){}(#)\t(#g){}(#)'.format(src,dst))
            patch_names.append(src.split('/')[-1])

        plot_label(orign_img, patch_names, probs, color_img_path)



def get_coord(patch_name):
    h, w = patch_name.split('_')[0],patch_name.split('_')[1]
    w = w.split('.')[0]
    return int(h)//4, int(w)//4

def plot_label(img, patch_names, probs, save_path):
    assert len(patch_names) == len(probs), print('length not match')
    for idx in range(len(patch_names)):
        h, w =get_coord(patch_names[idx])
        start = (w,h)
        end = (min(int(w+args.patch_size//4),img.shape[1]), min(int(h+args.patch_size//4),img.shape[0]))
        print(img.shape,start,end)
        cv2.rectangle(img, start, end, (0,255,0), 15)
        cv2.putText(img, str(round(probs[idx],2)), (w+25,h+60),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,255),5)
    cv2.imwrite(save_path, img)

def inference(loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, input in enumerate(loader):
            print('Batch: [{}/{}]'.format(i+1, len(loader)))
            input = input.cuda()
            output = F.softmax(model(input), dim=1)
            probs[i*args.batch_size:i*args.batch_size+input.size(0)] = output.detach()[:,1].clone()
    return probs.cpu().numpy()

def group_topk(groups, data, k=1):
    out = [] #topk prob
    out_index = [] #topk index
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-k:] = True
    index[:-k] = groups[k:] != groups[:-k]
    data = data[index]
    res_order = order[index]
    assert len(data) % k == 0, cp('(#r) topk lenth error(#): {}'.format(len(data)))
    assert len(data) == len(res_order), cp('(#r)prob and index not match(#)')
    for i in range(0,len(data),k):
        out.append(list(data[i:i+k]))
        out_index.append(list(res_order[i:i+k]))
    return out,out_index

if __name__ == '__main__':
    main()
