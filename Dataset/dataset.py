import os
import json
import random

import cv2
import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
import numpy as np
from eic_utils import cp


class MILdataset(data.Dataset):
    def __init__(self, data_path,  size, transform=None):
        #Flatten grid
        grid = []   # path for per patch
        slideIDX = []
        slidenames = []
        targets = []
        slideLen = [0]
        idx = 0
        for each_file in data_path:
            slidenames.append(each_file.split('/')[-1])
            if 'pos' in each_file:
                targets.append(1)
            else:
                targets.append(0)
            slideLen.append(slideLen[-1] + len(os.listdir(each_file)))
            for each_patch in os.listdir(each_file):
                img_path = os.path.join(each_file,each_patch)
                grid.append(img_path)
                slideIDX.append(idx)

            idx += 1
            cp('(#g)index: {}(#)\t(#r)name: {}(#)\t(#y)len: {}(#)'.format(idx,each_file.split('/')[-1],len(os.listdir(each_file))))
        cp('(#g)total: {}(#)'.format(len(grid)))
        print(slideLen)

        assert len(targets) == len(slidenames) , print("targets and names not match")
        assert len(slideIDX) == len(grid), print("idx and mask not match")


        self.slidenames = slidenames
        self.targets = targets
        self.grid = grid
        self.slideIDX = slideIDX
        self.transform = transform
        self.mode = None
        self.slideLen = slideLen#   patches for each slide

        # self.mult = lib['mult']
        # self.size = int(np.round(224*lib['mult']))
        self.size = size
        self.level = 0
    def setmode(self,mode):
        self.mode = mode
    def maketraindata(self, idxs):
        self.t_data = [(self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]]) for x in idxs]
    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))
    def getpatchinfo(self,idxs):
        images = []
        names = []
        targets = []
        for each in idxs:
            images.append(cv2.resize(cv2.imread(self.grid[each])[:,:,::-1],(512,512)))
            names.append(os.path.join(*self.grid[each].split('/')[-2:]).replace('.jpg',''))
            targets.append(self.targets[self.slideIDX[each]])
        return np.array(images), np.array(names), np.array(targets)
    def getslideinfo(self):
        return self.slidenames, self.slideLen

    def __getitem__(self,index):
        if self.mode == 1:
            img_path = self.grid[index]
            img = cv2.imread(img_path)
            img = img[:,:,::-1]
            #orign_img = F.to_tensor(img)
            if self.size != 224:
                img = cv2.resize(img,(224,224))
            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.mode == 2:
            slideIDX, img_path, target = self.t_data[index]
            img = cv2.imread(img_path)
            img = img[:, :, ::-1]
            if self.size != 224:
                img = cv2.resize(img, (224, 224))
            if self.transform is not None:
                img = self.transform(img)
            return img, target
    def __len__(self):
        if self.mode == 1:
            return len(self.grid)
        elif self.mode == 2:
            return len(self.t_data)
