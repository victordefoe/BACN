'''
Author: your name
Date: 2020-11-22 00:11:09
LastEditTime: 2020-11-22 01:49:00
LastEditors: Liu Chen
Description: 
FilePath: \BACN\test.py
  
'''

import torch, torchvision 
import numpy as np 
from dataset import Verific_Collection
from torch.utils.data import Dataset, DataLoader
from config import *
from networks import BACN, MatchNet

def Verification_test_BACN(net=None):
    if net is None:
        model = BACN()
        model.load_state_dict(torch.load('weights/{}_weight.pth'.format(model.name)))
    else:
        model = net
    data = Verific_Collection()
    DS = DataLoader(data, 1)
    TN, TP, FN, FP = 0, 0, 0, 0
    for index, dd in enumerate(DS):
        imgA, imgB, label = dd
        # out_probs,_,_ = self.model(item[0])
        label = label.float().squeeze()
        
        out_probs = model(imgA, imgB)
        diss = out_probs[:,0] - out_probs[:,1]  # [notsimilar_prob, similar_prob]
        
        thed = 0
        TN += ((diss > thed).float() * (1-label)).sum()  # 实际为负，预测为负
        TP += ((diss < thed).float() * (label)).sum()  # 实际为正，预测为正
        FN += ((diss > thed).float() * (label)).sum()  # 实际为正，预测为负
        FP += ((diss < thed).float() * (1-label)).sum()  # 实际为负，预测为正
        
        if index > 666:
            print(out_probs, label)
            break
    sens = TP/(TP + FN)
    spec = TN/(TN + FP)
    F_lessmiss = min(sens, spec)
    print('sens:{} = {}/{} , spec:{} = {}/{}'.format(sens,TP,(TP+FN),spec,TN,(TN+FP)))
    return round(F_lessmiss.item(), 4)  

def Verification_test_MatchNet(net=None):
    if net is None:
        model = MatchNet()
        model.load_state_dict(torch.load('weights/{}_weight.pth'.format(model.name)))
    else:
        model = net
    data = Verific_Collection()
    DS = DataLoader(data, 1)
    TN, TP, FN, FP = 0, 0, 0, 0
    for index, dd in enumerate(DS):
        imgA, imgB, label = dd
        # out_probs,_,_ = self.model(item[0])
        label = label.float().squeeze()
        pA =  model(imgA)
        pB = model(imgB)
        diss = (pA - pB).pow(2).sum(1).t()
        
        thed = 0.35
        TN += ((diss > thed).float() * (1-label)).sum()  # 实际为负，预测为负
        TP += ((diss < thed).float() * (label)).sum()  # 实际为正，预测为正
        FN += ((diss > thed).float() * (label)).sum()  # 实际为正，预测为负
        FP += ((diss < thed).float() * (1-label)).sum()  # 实际为负，预测为正
        
        if index > 666:
            break
    sens = TP/(TP + FN)
    spec = TN/(TN + FP)
    F_lessmiss = min(sens, spec)
    print('\nsens:{} = {}/{} , spec:{} = {}/{}'.format(sens,TP,(TP+FN),spec,TN,(TN+FP)))
    return round(F_lessmiss.item(), 4)  

if __name__ == '__main__':
    Verification_test_BACN()