import os
import random
from typing import Tuple, Sequence, Callable
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms


class DatasetMNIST(torch.utils.data.Dataset):
    def __init__(self,
                 dir_path,
                 meta_df,
                 mode='train',
                 augmentations=None):
        
        self.dir_path = dir_path 
        self.meta_df = meta_df 
        self.mode = mode
        self.train_mode = transforms.Compose([
                        transforms.RandomRotation(180, expand=False),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                            ])

        self.test_mode = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                            ])
        
    def __len__(self):
        return len(self.meta_df)
    
    def __getitem__(self, index):
        image = Image.open(self.dir_path + str(self.meta_df.iloc[index,0]).zfill(5) + '.png').convert('RGB')

        label = self.meta_df.iloc[index, 1:].values.astype('float')
        sample = {'image': image, 'label': label}

        if self.mode == 'train':
            sample['image'] = self.train_mode(sample['image'])

        elif self.mode == 'test' or self.mode == 'valid':
            sample['image'] = self.test_mode(sample['image'])

        sample['label'] = torch.FloatTensor(sample['label'])
        return sample

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


import matplotlib.pyplot as plt

def tensor2img(img):
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    a = unorm(img).numpy()
    a = a.transpose(1, 2, 0)
    return a