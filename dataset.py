'''
Author: your name
Date: 2020-11-21 20:28:56
LastEditTime: 2020-11-21 23:52:29
LastEditors: Liu Chen
Description: 
FilePath: \BACN\dataset.py
  
'''
import os
import json
import random
import pandas as pd
import numpy as np
from os.path import join as opj
import torch
import torchvision
import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageFile, ImageEnhance, ImageFilter
import skimage
from skimage import util

# modify the config.py in different enviornment for path configuration
from config import *
training_dataset_root = TRAIN_DATA_ROOT
train_gt = TRAIN_GT_PATH


# training data
class Train_Collection(Dataset):
    def __init__(self, img_size=(64,64)):
        sourceA = opj(training_dataset_root,'patch')
        sourceB = opj(training_dataset_root,'seps','18')
        dists_path = opj(training_dataset_root,'dists_forms')
        gts = json.load(open(train_gt))
        self.A_imgs = [opj(sourceA, i) for i in gts.keys() if i[-3:] == 'bmp']
        self.B_imgs = [opj(sourceB, i)
                       for i in gts.values() if i[-3:] == 'jpg']

        prepro = []
        prepro.append(transforms.Resize(size=img_size))
        prepro.append(transforms.ToTensor())
        self.trans = transforms.Compose(prepro)
    
    def __len__(self):
        return len(self.A_imgs)
            
    def __getitem__(self, item):
        pn = random.randint(0,1)
        if pn == 0:
            imgA = Image.open(self.A_imgs[item])
            imgAname = os.path.split(self.A_imgs[item])[1]

            imgB = Image.open(self.B_imgs[item])
            imgBname = os.path.split(self.B_imgs[item])[1]
            label = 1
        # negative condition
        else:
            imgA = Image.open(self.A_imgs[item])
            imgAname = os.path.split(self.A_imgs[item])[1]
            while True:
                choice = random.randint(0, len(self.B_imgs)-1)
                if choice != item:
                    break
            imgB = Image.open(self.B_imgs[choice])
            imgBname = os.path.split(self.B_imgs[choice])[1]
            label = 0
        
        label = torch.Tensor([label])
        img_A = self.trans(imgA)
        img_B = self.trans(imgB)

        return img_A, img_B, label



ver_dataset_root = VER_DATA_ROOT
ver_gt = VER_GT_PATH

class Verific_Collection(Dataset):
    def __init__(self, img_size=(64,64)):
        sourceA = opj(ver_dataset_root,'patch')
        sourceB = opj(ver_dataset_root,'seps','18')
        gts = json.load(open(ver_gt))

        self.A_imgs = [opj(sourceA, i) for i in gts.keys() if i[-3:] == 'bmp']
        self.B_imgs = [opj(sourceB, i)
                       for i in gts.values() if i[-3:] == 'jpg']

        prepro = []
        prepro.append(transforms.Resize(size=img_size))
        prepro.append(transforms.ToTensor())
        self.trans = transforms.Compose(prepro)

    def __len__(self):
        return len(self.A_imgs)

    
    def __getitem__(self, item):
        pn = random.randint(0,1)
        if pn == 0:
            imgA = Image.open(self.A_imgs[item])
            imgAname = os.path.split(self.A_imgs[item])[1]
            imgB = Image.open(self.B_imgs[item])
            imgBname = os.path.split(self.B_imgs[item])[1]
            label = 1
        # negative condition
        else:
            imgA = Image.open(self.A_imgs[item])
            imgAname = os.path.split(self.A_imgs[item])[1]
            while True:
                choice = random.randint(0, len(self.B_imgs)-1)
                if choice != item:
                    break
            imgB = Image.open(self.B_imgs[choice])
            imgBname = os.path.split(self.B_imgs[choice])[1]
            label = 0
        
        label = torch.Tensor([label])
        img_A = self.trans(imgA)
        img_B = self.trans(imgB)

        return img_A, img_B, label
        
            