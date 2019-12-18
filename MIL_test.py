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
from dataset import MILdataset
import shutil
import json
from eic_utils import cp
def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output', type=str, default='.', help='name of output directory')
    parser.add_argument('--model', type=str, default='./train_output/checkpoint_best.pth', help='path to pretrained model')
    parser.add_argument('--batch_size', type=int, default=100, help='how many images to sample per slide (default: 100)')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--patch_size', default=1024, type=int, help='size of patch')
    parser.add_argument('--k', default=1, type=int, help='topk of patch')
    return parser.parse_args()
args = get_args()
def main():

    #load model
    model = models.resnet34(True)
    model.fc = nn.Linear(model.fc.in_features, cfg.n_classes)
    ch = torch.load(args.model)
    model = torch.nn.DataParallel(model.cuda())
    model.load_state_dict(ch['state_dict'])
    cudnn.benchmark = True

    #normalization
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([transforms.ToTensor(),normalize])

    #load data
    with open(cfg.data_split) as f:
        data = json.load(f)

    dset = MILdataset(data['val_neg']+data['val_pos']+data['train_neg']+data['train_pos'], args.patch_size, trans)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    dset.setmode(1)
    probs = inference(loader, model)
    maxs,index = group_topk(np.array(dset.slideIDX), probs, args.k)
    #print(index)
    #print(len(maxs))
    #print(np.max(probs))
    fp = open(os.path.join(args.output, 'predictions.csv'), 'w')
    fp.write('file,target,prediction,probability\n')
    for name, target, probs, idxs in zip(dset.slidenames, dset.targets, maxs, index):
        for i in range(args.k):
            idx = idxs[i]
            src = dset.grid[idx]
            flag = 'pos' if target == 1 else 'neg'
            dst = os.path.join(cfg.patch_predict, flag, src.split('/')[-2])
            if not os.path.isdir(dst):
                os.makedirs(dst)
            shutil.copy(src,dst)
            cp('(#r){}(#)\t(#g){}(#)'.format(src,dst))
        #cp('(#b)probs: {}(#)'.format(probs))
        fp.write('{},{},{},{}\n'.format(name, target, int(probs[args.k-1]>=0.5), probs[args.k-1]))

    fp.close()

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
    for i in range(0,len(data),2):
        out.append(list(data[i:i+2]))
        out_index.append(list(res_order[i:i+2]))
    return out,out_index

if __name__ == '__main__':
    main()
