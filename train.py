'''
Author: your name
Date: 2020-11-21 20:29:41
LastEditTime: 2020-11-22 10:50:38
LastEditors: Liu Chen
Description: 
FilePath: \BACN\train.py
  
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataset import Train_Collection
from networks import MatchNet, BACN
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from config import *
from test import Verification_test_BACN, Verification_test_MatchNet

writer = SummaryWriter(log_dir='vislog')

Train_dataset = Train_Collection()
DS = DataLoader(Train_dataset, hyper.batchsize)


class Balance_loss(nn.Module):
    def __init__(self):
        super(Balance_loss, self).__init__()
        self.margin = 0.7
        self.eps = 1e-9

    def forward(self, predsA, predsB, target):
        distances = (predsA - predsB).pow(2).sum(1)
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() *
                        F.relu(self.margin - (distances + self.eps).sqrt()).pow(2)
                        )
        return losses.sum()


net = MatchNet()
# net = BACN()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
model_save_path = 'weights/{}_weight.pth'.format(net.name)
# net.load_state_dict(torch.load(model_save_path))
lossfunc = Balance_loss()
# lossfunc = torch.nn.CrossEntropyLoss()

itt = 0
globa_ver = float('inf')
for epoch in range(4):
    for a, b, labels in DS:
        # ======== MatchNet =====
        x = net(a)
        xplus = net(b)
        loss = lossfunc(x, xplus, labels)

        # ======== BACN  =====
        # out = net(a, b)
        # loss = lossfunc(out, labels.squeeze().long())

        loss.backward()
        optimizer.step()
        print(float(loss.data), end=' ')
        writer.add_scalar('scalar/loss', float(loss.data), itt)
        itt += 1
        if itt % 10 == 0:
            ver_rate = Verification_test_MatchNet(net)
            if ver_rate <= globa_ver:
                globa_ver = ver_rate
                torch.save(net.state_dict(), model_save_path)
                print('best: ', globa_ver)
            

writer.close()
