import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from PIL import Image

import torch
import torchvision.transforms as T

from dataloader import DatasetMNIST, tensor2img
from trainer import train_model
from resnet50 import Resnet

# device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

# path setup
root = os.path.join(os.getcwd(), 'dirty_mnist')
train_path = os.path.join(root, 'train')
dirty_mnist_answer = pd.read_csv(os.path.join(root, "dirty_mnist_2nd_answer.csv"))

BAGGING_NUM = 4
BATCH_SIZE = 32
flod_num = 4

# 모델을 학습하고, 최종 모델을 기반으로 테스트 데이터에 대한 예측 결과물을 저장하는 도구 함수이다
def train_and_predict(cfg_dict):
    cfg = cfg_dict.copy()
    cfg['bagging_num'] = BAGGING_NUM
    cfg['fold_num'] = flod_num
    print("training ")
    # 모델을 학습
    train_model(**cfg)

# ResNet 모델 학습 설정값
res_config = {
    'model_class': Resnet,
    'is_1d': False,
    'reshape_size': None,
    'BATCH_SIZE': BATCH_SIZE,
    'epochs': 30,
    'lr' : 1e-3,
    'CODER': 'resnet',
    'DatasetMNIST' : DatasetMNIST,
    'dirty_mnist_answer' : dirty_mnist_answer,
}

if __name__ == '__main__':
    print("train resnet.........")
    train_and_predict(res_config)