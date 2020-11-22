'''
Author: your name
Date: 2020-11-21 20:30:33
LastEditTime: 2020-11-22 01:23:21
LastEditors: Liu Chen
Description: 
FilePath: \BACN\networks.py
  
'''


import torch
import torchvision
from torch.nn.modules import LSTM
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import copy
import numpy as np
from matplotlib import pyplot as plt


class CNN_Layer(nn.Module):
    """
    bn: if Ture, use batch normalization
    pooling: tuple - (type, kernel_size, strides, padding)
                type: 'max', 'avg';
                kernel_size, strides: int or tuple
                padding: int
                pooling=None: don not use pooling layer
    act: choose an activation: 'relu' ... , None do not use activation
    """
    def __init__(self, inc, outc,
                 kernel, strides=1, padding=0,
                 bn=True, pooling=None, act='relu'):
       
        super(CNN_Layer, self).__init__()
        activations = {'relu': nn.ReLU}
        pool_methods = {'max': nn.MaxPool2d}

        cnn = []
        cnn.append(nn.Conv2d(inc, outc, kernel, strides, padding))
        if bn is True:
            cnn.append(nn.BatchNorm2d(outc))
        if act is not None:
            cnn.append(activations[act]())
        if pooling is not None:
            cnn.append(pool_methods[pooling[0]](pooling[1],
                                                pooling[2], pooling[3]))

        CNN_BN_ReLu = nn.Sequential(*cnn)

        self.cnn_layer = CNN_BN_ReLu

    def forward(self, x):
        return self.cnn_layer(x)


class MatchNet(nn.Module):
    """
    OutPut: stacked features: shape [2, batchsize, feat_dim],
             A_feats and B_feats are in it
    """

    def __init__(self):
        self.name = "MatchNet"
        super(MatchNet, self).__init__()
        FCs = [nn.Linear(256, 512),
               nn.ReLU(),
               nn.Linear(512, 50)]
        self.header = nn.Sequential(*FCs)
        CNNs = []
        CNNs.append(CNN_Layer(3, 24, 7, padding=3, pooling=('max', 2, 2, 0)))
        CNNs.append(CNN_Layer(24, 64, 5, padding=2, pooling=('max', 3, 2, 0)))
        CNNs.append(CNN_Layer(64, 96, 3, padding=0))
        CNNs.append(CNN_Layer(96, 96, 1, padding=0))
        CNNs.append(CNN_Layer(96, 64, 3, strides=2,
                              padding=0, pooling=('max', 3, 2, 0)))
        self.FeatureNetwork = nn.Sequential(*CNNs)

    def forward(self, x):
        f = self.FeatureNetwork(x)
        f = torch.flatten(f, start_dim=1)
        f = self.header(f)
        out = f / f.detach().pow(2).sum(1, keepdim=True).sqrt()
    
        return out




class BACN(nn.Module):
    # 输入必须是 64x64
    def __init__(self):
        self.name = "BACN"
        super(BACN, self).__init__()
        # A_bot = [CNN_Layer(3, 32, 3, padding=1),
        #          CNN_Layer(32, 3, 3, padding=1)]
        # B_bot = A_bot.copy()
        # self.CNNA = nn.Sequential(*A_bot)
        # self.CNNB = nn.Sequential(*B_bot)
        CNNs = []
        CNNs.append(CNN_Layer(6, 24, 7, padding=3, pooling=('max', 2, 2, 0)))
        CNNs.append(CNN_Layer(24, 64, 5, padding=2, pooling=('max', 3, 2, 0)))
        CNNs.append(CNN_Layer(64, 96, 3, padding=0))
        CNNs.append(CNN_Layer(96, 96, 1, padding=0))
        CNNs.append(CNN_Layer(96, 64, 3, strides=2,
                              padding=0, pooling=('max', 3, 2, 0)))
        self.FeatureNetwork = nn.Sequential(*CNNs)
        FCs = [nn.Linear(256, 1024),
               nn.ReLU(),
               nn.Linear(1024, 2),
               nn.PReLU()
               ]
        self.fc = nn.Sequential(*FCs)

    def forward(self, xa, xb):
        # a = self.CNNA(xa)
        # b = self.CNNB(xb)
        x = torch.cat([xa, xb], dim=1)
        f = self.FeatureNetwork(x)
        f = torch.flatten(f, start_dim=1)
        out = self.fc(f)
        return out