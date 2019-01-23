from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Using your codes in Homework 1'''
        loss = np.sum((input-target)**2) / 2*len(target)
        return loss

    def backward(self, input, target):
        '''Using your codes in Homework 1'''
        loss_back = np.abs(input-target)
        return loss_back


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        loss = -np.sum(target*np.log(input)+(1-target)*np.log(1-input))
        return loss

    def backward(self, input, target):
        '''Your codes here'''
        loss_back = target/input + (1-target)/(1-input)
        return -loss_back