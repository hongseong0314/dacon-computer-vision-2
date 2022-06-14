import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import zipfile
from PIL import Image
from glob import glob

import torch
import torchvision.transforms as T
import ttach as tta
from torch.utils.data import Dataset, DataLoader

from dataloader import DatasetMNIST
from model.efficientNet  import mnistEfficient

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#
sample_submission = pd.read_csv("./sample_submission.csv")
test_dataset = DatasetMNIST("test_dirty_mnist/test/", sample_submission)
batch_size = 128
test_data_loader = DataLoader(
    test_dataset,
    batch_size = batch_size,
    shuffle = False,
    num_workers = 8,
    drop_last = False
)

tta_transforms = tta.Compose([tta.Rotate90(angles=[0, 90, 180, 270]),])

def inference(model_class, thr=0.5, m=False, path=None, score=None, tta_chech=False):
    """
    model_class : 사용할 모델
    thr : 임계값
    m : 멀티 GPU사용
    path : 모델 파라미터 저장 경로
    score : greedy 앙상블 시 각 모델의 정확도 값
    tta_chech : TTA 사용 유무
    """
    def get_model(model=model_class, m=m, pretrained=False):
            # multi-GPU일 경우, Data Parallelism
        mdl = torch.nn.DataParallel(model()) if m else model()
        if not pretrained:
            return mdl
        else:
            print("load pretrained model here...")
            # 기학습된 torch.load()로 모델을 불러온다
            mdl.load_state_dict(torch.load(pretrained))
            return mdl

    model = get_model()
    model.to(device)
    
    if not score:
        for e, m in enumerate(path):
            model.load_state_dict(torch.load(m))
            model.eval()
            pred_scores = []
            
            for batch_idx, batch_data in enumerate(test_data_loader):
                with torch.no_grad():
                    # 추론
                    images = batch_data['image']
                    images = images.to(device)
                    
                    if tta_chech:
                        tta_model = tta.ClassificationTTAWrapper(model, tta_transforms)
                        probs  = tta_model(images)
                    else:
                        probs  = model(images)
                        
                    probs = probs.cpu().detach().numpy()
                pred_scores.append(probs)

            # 앙상블 0
            if e == 0:
                final_pred = np.vstack(pred_scores)
                # final_test_fnames = test_fnames
            else:
                final_pred += np.vstack(pred_scores)
                # assert final_test_fnames == test_fnames
        final_pred /= len(path)

        # threshold     
        return (final_pred >= thr) * 1
    else:
        def softmax(x):
            return np.exp(x) / np.sum(np.exp(x))

        model_acc_weight = softmax(score)
        for e, m in enumerate(path):
            model.load_state_dict(torch.load(m))
            model.eval()
            pred_scores = []
         
            for batch_idx, batch_data in enumerate(test_data_loader):
                with torch.no_grad():
                    # 추론
                    images = batch_data['image']
                    images = images.to(device)
                    
                    if tta_chech:
                        tta_model = tta.ClassificationTTAWrapper(model, tta_transforms)
                        probs  = tta_model(images)
                    else:
                        probs  = model(images)

                    probs = probs.cpu().detach().numpy()
                pred_scores.append(probs)

            
            # 앙상블 0
            if e == 0:
                final_pred = np.vstack(pred_scores) * model_acc_weight[e]
                # final_test_fnames = test_fnames
            else:
                final_pred += np.vstack(pred_scores) * model_acc_weight[e]
        
        return (final_pred >= thr) * 1


if __name__ == '__main__':
    model_weight = glob(os.path.join(os.getcwd(), 'full/*.pth'))
    print(f"bagging num : {len(model_weight)}")

    # score = np.array([0.8030923077, 0.8081230769, 0.8103230769, 0.8091846154])
    
    pred = inference(mnistEfficient, thr=0.499, m=True, path=model_weight, tta_chech=True)
    sample_submission = pd.read_csv("./sample_submission.csv")
    sample_submission.iloc[:,1:] = np.vstack(pred)
    sample_submission.to_csv("effb0_full.csv", index = False)
    
  







