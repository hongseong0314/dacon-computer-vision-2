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
    """
    dir_path : 데이터폴더 경로
    meta_df : 가져올 데이터 csv
    mode : 불러 올 데이터(train or test)
    mix_up : mix_up augmentation 유무
    reverse : reverse augmentation 유무
    sub_data : emnist 로 만든 sub data 사용 유무
    """
    def __init__(self,
                 dir_path,
                 meta_df,
                 mode='train',
                 mix_up=False,
                 reverse=False,
                 augmentations=None,
                 sub_data=False):
        
        self.dir_path = dir_path 
        self.meta_df = meta_df 
        self.mode = mode
        self.mix = mix_up
        self.sub_data = sub_data
        self.reverse = reverse
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
        
        # subdata True시 30%로 가져오기
        if self.sub_data and np.random.rand() >= 0.7:
            sub_df = pd.read_csv(r"./sub_data.csv")
            image = Image.open(str(sub_df.iloc[index,0])).convert('RGB')
            label = sub_df.iloc[index, 1:].values.astype('float')
        
        # mix up
        if np.random.rand() >= 0.8 and self.mix == True and self.mode=='train':
            image, label = self.mixup(image, label)

        sample = {'image': image, 'label': label}

        # train mode transform
        if self.mode == 'train':
            sample['image'] = self.train_mode(sample['image'])

        # test mode transform
        elif self.mode == 'test' or self.mode == 'valid':
            sample['image'] = self.test_mode(sample['image'])

        # reverse
        if np.random.rand() >= 0.8 and self.mode=='train' and self.reverse:
            sample['image'] = self.diagonal_reverse(sample['image'])

        sample['label'] = torch.FloatTensor(sample['label'])
        return sample
    
    def mixup(self, image, label):
        """
        랜덤하게 4개의 이미지를 붙인다.
        """
        idxs = np.random.randint(1, len(self.meta_df), 3)
        h, w = image.size
        
        images = [Image.open(self.dir_path + str(self.meta_df.iloc[index,0]).zfill(5) + '.png').convert('RGB') for index in idxs]
        images.append(image)
        labels = [self.meta_df.iloc[index, 1:].values.astype('float') for index in idxs]
        labels.append(label)
        
        expand_img = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        expand_img[:h, :w, :] = np.array(images[0])
        expand_img[:h, w:int(w*2), :] = np.array(images[1])
        expand_img[h:int(h*2), 0:w, :] = np.array(images[2])
        expand_img[h:int(h*2), w:int(w*2), :] = np.array(images[3])
        
        del images
        return Image.fromarray(expand_img).resize((h,w)), np.clip(np.sum(np.array(labels), axis=0), 0, 1)
    
    def diagonal_reverse(self, img):
        """
        하나의 이미지를 4등분하여 재배열 한다.
        """
        transformed_img = img.clone()
        center = img.shape[1] // 2
        transformed_img[:, 0:center, 0:center] = img[:, center:center + center, center:center + center]
        transformed_img[:, 0:center, center:center + center]  = img[:, center:center*2, 0:center]
        transformed_img[:, center:center + center, 0:center] = img[:, 0:center, center:center*2]
        transformed_img[:, center:center + center, center:center + center] = img[:, 0:center, 0:center]
        
        return transformed_img

class UnNormalize(object):
    """
    정규화 inverse 한다.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


import matplotlib.pyplot as plt

def tensor2img(img):
    """
    tensor 형태에서 0~255 진행 후 numpy 배열로 바꿔서 반환
    """
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    a = unorm(img).numpy()
    a = a.transpose(1, 2, 0)
    return a