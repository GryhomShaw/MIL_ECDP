import os
import sys
import random
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from Dataset import Rnndata
from config import cfg
from eic_utils import procedure,cp

def get_args():
    parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 RNN aggregator training script')
    parser.add_argument('--output', type=str, default='.', help='name of output file')
    parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size (default: 128)')
    parser.add_argument('--nepochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--workers', default=16, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--s', default=2, type=int, help='how many top k tiles to consider (default: 10)')
    parser.add_argument('--ndims', default=128, type=int, help='length of hidden representation (default: 128)')
    parser.add_argument('--model', type=str, help='path to trained model checkpoint')
    parser.add_argument('--shuffle', action='store_true', help='to shuffle order of sequence')
    parser.add_argument('--patch_size', type=int, default=2048, help='size of patch')
    return parser.parse_args()

best_acc = 0
args = get_args()
def main():
    global  best_acc
    #load libraries
    normalize = transforms.Normalize(mean=cfg.mean,std=cfg.std)
    trans = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    with procedure('Prepare Dataset'):
        with open(cfg.rnn_data_split) as f:
            data = json.load(f)
        shuffle ={'train': True,'val':False}
        dataset = {x : Rnndata(data[x+'_pos']+data[x+'_neg'], args.s,args.patch_size, False, trans) for x in ['train', 'val']}
        dataloader = {x : torch.utils.data.DataLoader(dataset[x], batch_size = args.batch_size, shuffle = shuffle[x],
                                                        num_workers = args.workers,pin_memory = True)
                                                        for x in ['train','val']}
    #make model
    with procedure('Init Model'):
        embedder = ResNetEncoder(args.model)
        for param in embedder.parameters():
            param.requires_grad = False
        embedder = torch.nn.parallel.DataParallel(embedder.cuda())
        embedder.eval()

        rnn = rnn_single(args.ndims)
        rnn = torch.nn.parallel.DataParallel(rnn.cuda())
    
    #optimization
    with procedure('Optimization and Criterion'):
        if cfg.weights==0.5:
            criterion = nn.CrossEntropyLoss().cuda()
        else:
            w = torch.Tensor([1-cfg.weights,cfg.weights])
            criterion = nn.CrossEntropyLoss(w).cuda()
        optimizer = optim.SGD(rnn.parameters(), 0.1, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)
        cudnn.benchmark = True

    fconv = open(os.path.join(args.output, 'convergence.csv'), 'w')
    fconv.write('epoch,train.loss,train.fpr,train.fnr,val.loss,val.fpr,val.fnr\n')
    fconv.close()

    #
    for epoch in range(args.nepochs):

        train_loss, train_fpr, train_fnr = train_single(epoch, embedder, rnn, dataloader['train'], criterion, optimizer)
        val_loss, val_fpr, val_fnr = test_single(epoch, embedder, rnn, dataloader['val'], criterion)

        fconv = open(os.path.join(args.output,'convergence.csv'), 'a')
        fconv.write('{},{},{},{},{},{},{}\n'.format(epoch+1, train_loss, train_fpr, train_fnr, val_loss, val_fpr, val_fnr))
        fconv.close()

        val_err = (val_fpr + val_fnr)/2
        if 1-val_err >= best_acc:
            best_acc = 1-val_err
            obj = {
                'epoch': epoch+1,
                'state_dict': rnn.state_dict()
            }
            torch.save(obj, os.path.join(args.output,'rnn_checkpoint_best.pth'))

def train_single(epoch, embedder, rnn, loader, criterion, optimizer):
    rnn.train()
    running_loss = 0.
    running_fps = 0.
    running_fns = 0.

    for i,(inputs,target) in enumerate(loader):
        cp('(#y)Training - Epoch: [{}/{}](#)\t(#g)Batch: [{}/{}](#)'.format(epoch+1, args.nepochs, i+1, len(loader)))
        print(inputs[0].size(),len(inputs),len(target))
        batch_size = inputs[0].size(0)
        rnn.zero_grad()

        state = rnn.module.init_hidden(batch_size).cuda()
        for s in range(len(inputs)):
            input = inputs[s].cuda()
            _, input = embedder(input)
            output, state = rnn(input, state)

        target = target.cuda()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()*target.size(0)
        fps, fns = errors(output.detach(), target.cpu())
        running_fps += fps
        running_fns += fns

    running_loss = running_loss/len(loader.dataset)
    running_fps = running_fps/(np.array(loader.dataset.targets)==0).sum()
    running_fns = running_fns/(np.array(loader.dataset.targets)==1).sum()
    cp('(#y)Training - Epoch: [{}/{}](#)\t(#r)Loss: {}(#)\t(#g)FPR: {}(#)\t(#b)FNR: {}(#)'
       .format(epoch+1, args.nepochs, running_loss, running_fps, running_fns))
    return running_loss, running_fps, running_fns

def test_single(epoch, embedder, rnn, loader, criterion):
    rnn.eval()
    running_loss = 0.
    running_fps = 0.
    running_fns = 0.

    with torch.no_grad():
        for i,(inputs,target) in enumerate(loader):
            cp('(#y)Validating - Epoch: [{}/{}](#)\t(#b)Batch: [{}/{}](#)'.format(epoch+1,args.nepochs,i+1,len(loader)))
            batch_size = inputs[0].size(0)
            
            state = rnn.module.init_hidden(batch_size).cuda()
            for s in range(len(inputs)):
                input = inputs[s].cuda()
                _, input = embedder(input)
                output, state = rnn(input, state)
            
            target = target.cuda()
            loss = criterion(output,target)
            
            running_loss += loss.item()*target.size(0)
            fps, fns = errors(output.detach(), target.cpu())
            running_fps += fps
            running_fns += fns
            
    running_loss = running_loss/len(loader.dataset)
    running_fps = running_fps/(np.array(loader.dataset.targets)==0).sum()
    running_fns = running_fns/(np.array(loader.dataset.targets)==1).sum()
    cp('(#y)Validating - Epoch: [{}/{}](#)\t(#r)Loss: {}(#)\t(#g)FPR: {}(#)\t(#b)FNR: {}(#)'
       .format(epoch+1, args.nepochs, running_loss, running_fps, running_fns))
    return running_loss, running_fps, running_fns

def errors(output, target):
    _, pred = output.topk(1, 1, True, True)
    pred = pred.squeeze().cpu().numpy()
    real = target.numpy()
    neq = pred!=real
    fps = float(np.logical_and(pred==1,neq).sum())
    fns = float(np.logical_and(pred==0,neq).sum())
    return fps,fns

class ResNetEncoder(nn.Module):

    def __init__(self,path):
        super(ResNetEncoder, self).__init__()
        temp = models.resnet34()
        temp.fc = nn.Linear(temp.fc.in_features, 2)
        temp.load_state_dict({k.replace('module.',''):v for k,v in torch.load(path)['state_dict'].items()})
        self.features = nn.Sequential(*list(temp.children())[:-1])
        self.fc = temp.fc

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        return self.fc(x), x

class rnn_single(nn.Module):

    def __init__(self, ndims):
        super(rnn_single, self).__init__()
        self.ndims = ndims

        self.fc1 = nn.Linear(512, ndims)
        self.fc2 = nn.Linear(ndims, ndims)

        self.fc3 = nn.Linear(ndims, 2)

        self.activation = nn.ReLU()

    def forward(self, input, state):
        input = self.fc1(input)
        state = self.fc2(state)
        state = self.activation(state+input)
        output = self.fc3(state)
        return output, state

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.ndims)



if __name__ == '__main__':
    main()
