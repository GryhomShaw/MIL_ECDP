import os
import sys
import time
import numpy as np
import argparse
import random
import openslide
import json
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from Dataset import MILdataset
from eic_utils import procedure,cp
from config import cfg
from utils.summary import TensorboardSummary
from utils.average import AverageMeter, ProgressMeter
os.environ["CUDA_VISIBLE_DEVICES"] ="0,1"

def get_args():
    parser = argparse.ArgumentParser(description='ECDP_NCIC')
    parser.add_argument('--val', type=bool, default=True, help="val or not")
    parser.add_argument('--patch_size', type=int, default=512, help="size of patch")
    parser.add_argument('--output', type=str, default='./train_output', help='name of output file')
    parser.add_argument('--batch_size', type=int, default=512, help='mini-batch size (default: 512)')
    parser.add_argument('--nepochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--workers', default=16, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--test_every', default=10, type=int, help='test on val every (default: 10)')
    parser.add_argument('--k', default=10, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')
    parser.add_argument('--dis_slide', default=2, type=int, help='display slides for one step on the tensorboradX')
    return parser.parse_args()
best_acc = 0
def main():
    global args, best_acc
    args = get_args()

    #cnn
    with procedure('init model'):
        model = models.resnet34(True)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model = torch.nn.parallel.DataParallel(model.cuda())

    with procedure('loss and optimizer'):
        if cfg.weights==0.5:
            criterion = nn.CrossEntropyLoss().cuda()
        else:
            w = torch.Tensor([1-cfg.weights,cfg.weights])
            criterion = nn.CrossEntropyLoss(w).cuda()
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    cudnn.benchmark = True

    #normalization
    normalize = transforms.Normalize(mean=cfg.mean,std=cfg.std)
    trans = transforms.Compose([transforms.ToTensor(), normalize])


    with procedure('prepare dataset'):
        #load data
        with open(cfg.data_split) as f :   #
            data = json.load(f)
        train_dset = MILdataset(data['train_neg'][:14] + data['train_pos'],  args.patch_size, trans)
        train_loader = torch.utils.data.DataLoader(
            train_dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        if args.val:
            val_dset = MILdataset(data['val_pos']+data['val_neg'], args.patch_size, trans)
            val_loader = torch.utils.data.DataLoader(
                val_dset,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
    with procedure('init tensorboardX'):
        tensorboard_path = os.path.join(args.output,'tensorboard')
        if not os.path.isdir(tensorboard_path):
            os.makedirs(tensorboard_path)
        summary = TensorboardSummary(tensorboard_path,args.dis_slide)
        writer = summary.create_writer()
    
    #open output file
    fconv = open(os.path.join(args.output,'convergence.csv'), 'w')
    fconv.write('epoch,metric,value\n')
    fconv.close()

    #loop throuh epochs
    for epoch in range(args.nepochs):
        train_dset.setmode(1)
        probs = inference(epoch, train_loader, model)
        topk = group_argtopk(np.array(train_dset.slideIDX), probs, args.k)
        images, names, labels = train_dset.getpatchinfo(topk)
        summary.plot_calsses_pred(writer,images,names,labels,np.array([probs[k] for k in topk ]),args.k,epoch)
        slidenames, length =train_dset.getslideinfo()
        summary.plot_histogram(writer,slidenames, probs, length, epoch)
        #print([probs[k] for k in topk ])
        train_dset.maketraindata(topk)
        train_dset.shuffletraindata()
        train_dset.setmode(2)
        loss = train(epoch, train_loader, model, criterion, optimizer, writer)
        cp('(#r)Training(#)\t(#b)Epoch: [{}/{}](#)\t(#g)Loss:{}(#)'.format(epoch+1, args.nepochs, loss))
        fconv = open(os.path.join(args.output, 'convergence.csv'), 'a')
        fconv.write('{},loss,{}\n'.format(epoch+1,loss))
        fconv.close()

        #Validation
        if args.val and (epoch+1) % args.test_every == 0:
            val_dset.setmode(1)
            probs = inference(epoch, val_loader, model)
            maxs = group_max(np.array(val_dset.slideIDX), probs, len(val_dset.targets))
            pred = [1 if x >= 0.5 else 0 for x in maxs]
            err,fpr,fnr = calc_err(pred, val_dset.targets)
            #print('Validation\tEpoch: [{}/{}]\tError: {}\tFPR: {}\tFNR: {}'.format(epoch+1, args.nepochs, err, fpr, fnr))
            cp('(#y)Vaildation\t(#)(#b)Epoch: [{}/{}]\t(#)(#g)Error: {}\tFPR: {}\tFNR: {}(#)'.format(epoch+1, args.nepochs, err, fpr, fnr))
            fconv = open(os.path.join(args.output, 'convergence.csv'), 'a')
            fconv.write('{},error,{}\n'.format(epoch+1, err))
            fconv.write('{},fpr,{}\n'.format(epoch+1, fpr))
            fconv.write('{},fnr,{}\n'.format(epoch+1, fnr))
            fconv.close()
            #Save best model
            err = (fpr+fnr)/2.
            if 1-err >= best_acc:
                best_acc = 1-err
                obj = {
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict()
                }
                torch.save(obj, os.path.join(args.output,'checkpoint_best.pth'))

def inference(run, loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, input in enumerate(loader):
            #print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run+1, args.nepochs, i+1, len(loader)))
            if (i+1) % 50 == 0:
                cp('(#y)Inference\t(#)(#b)Epoch:[{}/{}]\t(#)(#g)Batch: [{}/{}](#)'.format(run+1, args.nepochs, i+1, len(loader)))
            input = input.cuda()
            output = F.softmax(model(input), dim=1)
            probs[i*args.batch_size:i*args.batch_size+input.size(0)] = output.detach()[:,1].clone()
    return probs.cpu().numpy()

def train(run, loader, model, criterion, optimizer, writer):
    losses = AverageMeter('Loss', ':.4f')
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(len(loader), [batch_time, data_time, losses],
                             prefix=cp.trans("(#b)[TRN](#) Epoch: [{}]".format(run)))
    model.train()
    running_loss = 0.
    end = time.time()
    for i, (input, target) in enumerate(loader):
        data_time.update(time.time() - end)
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        losses.update(loss.item(),input.size(0))
        running_loss += loss.item()*input.size(0)
        progress.display(i)
        writer.add_scalar('train/loss',loss.item(),run * len(loader) + i)
    return running_loss/len(loader.dataset)

def calc_err(pred,real):
    pred = np.array(pred)
    real = np.array(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum())/pred.shape[0]
    fpr = float(np.logical_and(pred==1,neq).sum())/(real==0).sum()
    fnr = float(np.logical_and(pred==0,neq).sum())/(real==1).sum()
    return err, fpr, fnr

def group_argtopk(groups, data,k=1):
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-k:] = True
    index[:-k] = groups[k:] != groups[:-k]
    return list(order[index])

def group_max(groups, data, nmax):
    out = np.empty(nmax)
    out[:] = np.nan
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    out[groups[index]] = data[index]
    return out

if __name__ == '__main__':
    main()
