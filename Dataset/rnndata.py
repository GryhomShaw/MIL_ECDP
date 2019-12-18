import os
import random

import cv2
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from eic_utils import cp

class Rnndata(data.Dataset):
    def __init__(self, path, s, size, shuffle=False, transform=None):
        # path : ./Predict/10
        slidenames = []
        targets = []
        grid = []  #path of patch
        temp_grid = []
        for each_slide in path:
            if 'pos' in each_slide:
                targets.append(1)
            elif 'neg' in each_slide:
                targets.append(0)
            slidenames.append(each_slide.split('/')[-1])
            for each_patch in os.listdir(each_slide):
                temp_grid.append(os.path.join(each_slide,each_patch))
            grid.append(temp_grid)
            temp_grid = []
        cp('(#g)Total length: {}(#) (#y)grid: {}(#)'.format(len(targets),len(grid)))
        print(grid[0])
        assert len(targets) == len(slidenames) , cp('(#r) targets and slidenames not match (#)')
        self.s = s
        self.transform = transform
        self.slidenames = slidenames
        self.targets = targets
        self.grid = grid
        self.level = 0
        self.size = size
        self.shuffle = shuffle


    def __getitem__(self, index):

        grid = self.grid[index]
        if self.shuffle:
            grid = random.sample(grid, len(grid))

        out = []
        s = min(self.s, len(grid))
        #print(s,len(grid))
        for i in range(s):
            img = cv2.imread(grid[i])
            img = img[:,:,::-1]
            if self.size != 224:
                img = cv2.resize(img,(224, 224))
            if self.transform is not None:
                img = self.transform(img)
            out.append(img)
        return out, self.targets[index]

    def __len__(self):

        return len(self.targets)
