import os
import math
import torch
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TensorboardSummary(object):
    def __init__(self, path, slides_per_step):
        self.path = path
        self.slides_per_step = slides_per_step #    每次显示几张图的结果

    def create_writer(self):
        return SummaryWriter(log_dir = os.path.join(self.path))

    def plot_calsses_pred(self, writer, images, names, targets, probs, K, step):
        """
        :param writer:
        :param images: K * H * W * C numpy
        :param names: image names (K)
        :param targets: K (0/1)
        :param probs:  K (0~1)
        :param K: topk
        :param step
        :return: none
        """
        assert images.shape[0] == names.shape[0] == targets.shape[0] == probs.shape[0], print('shape error')
        deta = self.slides_per_step * K
        total = math.ceil(images.shape[0] / deta)
        cur_step = (step % total) * deta
        images = images[cur_step:cur_step + deta, :, :, :]
        names = names[cur_step:cur_step + deta]
        targets = targets[cur_step: cur_step + deta]
        probs = probs[cur_step:cur_step + deta]
        fig = plt.figure(figsize= (200,20))
        per_row = images.shape[0] // K
        print(images.shape[0], per_row, K)
        for idx in range(per_row):
            for idy in range(K):
                index = idx*K + idy
                ax = fig.add_subplot(per_row,K,index+1,xticks=[],yticks=[])
                plt.imshow(images[index])
                ax.set_title("{0}：{1:.1f}\nlabel: {2}".format(names[index],probs[index]*100.0,targets[index]),
                             color = ("green" if(int(probs[index] >= 0.5) == targets[index]) else "red"),fontsize = 7)
        writer.add_figure('predicitions vs targets', fig, step)
    def plot_histogram(self, writer, names, probs, length, step):
        assert len(probs) == length[-1] ,print("shape error")
        for i in range(len(length)-1):
            cur_start = length[i]
            cur_end =  length[i + 1]
            writer.add_histogram(names[i],np.array(probs[cur_start:cur_end]),step)



