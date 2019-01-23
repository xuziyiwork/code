from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        loss = np.sum(np.square(input-target))/len(target)
        return loss

    def backward(self, input, target):
        '''Your codes here'''
        loss_back = np.abs(input-target)/len(target)
        return loss_back
 