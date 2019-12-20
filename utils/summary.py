import os
import math
import torch
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import numpy as np


class TensorboardSummary(object):
    def __init__(self, path):
        self.path = path

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
        deta = 2 * K
        total = math.ceil(images.shape[0] / deta)
        cur_step = (step % total) * deta
        images = images[cur_step:cur_step + deta, :, :, :]
        names = names[cur_step:cur_step + deta]
        targets = targets[cur_step: cur_step + deta]
        probs = probs[cur_step:cur_step + deta]
        fig = plt.figure()
        per_row = images.shape[0] // K
        print(images.shape[0], per_row, K)
        for idx in range(per_row):
            for idy in range(K):
                index = idx*K + idy
                ax = fig.add_subplot(per_row,K,index+1,xticks=[],yticks=[])
                plt.imshow(images[index])
                ax.set_title("{0}：{1:.1f}\nlabel: {2}".format(names[index],probs[index]*100.0,targets[index]),
                             color = ("green" if(int(probs[index] >= 0.5) == targets[index]) else "red"),fontsize = 5)
        writer.add_figure('predicitions vs targets', fig, step)



