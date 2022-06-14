import sys
import os
sys.path.append('mnist_classes')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from PIL import Image
import random
import torch
import torchvision.transforms as T

from dataloader import DatasetMNIST, tensor2img
from trainer import train_model
from model.efficientNet import mnistEfficient

# path setup
root = os.path.join(os.getcwd(), 'dirty_mnist')
train_path = os.path.join(root, 'train')
dirty_mnist_answer = pd.read_csv("./dirty_mnist_2nd_answer.csv")

BAGGING_NUM = 4
BATCH_SIZE = 256
flod_num = 4
save_path = os.getcwd()
# 모델을 학습하고, 최종 모델을 기반으로 테스트 데이터에 대한 예측 결과물을 저장하는 도구 함수이다
def train_and_predict(cfg_dict):
    cfg = cfg_dict.copy()
    cfg['bagging_num'] = BAGGING_NUM
    cfg['fold_num'] = flod_num
    print("training ")
    # 모델을 학습
    train_model(**cfg)

# Eff 모델 학습 설정값
eff_config = {
    'model_class': mnistEfficient,
    'is_1d': False,
    'reshape_size': None,
    'BATCH_SIZE': BATCH_SIZE,
    'epochs': 30,
    'lr' : 1e-3,
    'CODER': 'efficient',
    'DatasetMNIST' : DatasetMNIST,
    'dirty_mnist_answer' : dirty_mnist_answer,
}

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    seed_everything(np.random.randint(1, 5000))
    print("train efficientNet.........")
    train_and_predict(eff_config)
