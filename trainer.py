# trainer 함수
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
from time import time
from torch.nn import Softmax
import numpy as np
import pandas as pd
import os
from random import choice
from sklearn.model_selection import KFold
from torchvision import transforms
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def train_model(model_class, DatasetMNIST, dirty_mnist_answer, BATCH_SIZE, epochs, lr, is_1d = None, 
                reshape_size = None, fold_num=4, CODER=None, 
                bagging_num=1):
    # bagging_num 만큼 모델 학습을 반복 수행한다
    for b in range(bagging_num):
        print("bagging num : ", b)

        # 교차 검증
        kfold = KFold(n_splits=fold_num, shuffle=True)
        best_models = [] 
        
        previse_name = ''
        best_model_name = ''
        for fold_index, (trn_idx, val_idx) in enumerate(kfold.split(dirty_mnist_answer),1):
            print(f'[fold: {fold_index}]')
            torch.cuda.empty_cache()
            
            # kfold dataset 구성
            train_answer = dirty_mnist_answer.iloc[trn_idx]
            test_answer  = dirty_mnist_answer.iloc[val_idx]

            #Dataset 정의
            train_dataset = DatasetMNIST("dirty_mnist/train/", train_answer, 'train')
            valid_dataset = DatasetMNIST("dirty_mnist/train/", test_answer, 'test')

            #DataLoader 정의
            train_data_loader = DataLoader(
                train_dataset,
                batch_size = BATCH_SIZE,
                shuffle = False,
                num_workers = 8,
            )
            valid_data_loader = DataLoader(
                valid_dataset,
                batch_size = int(BATCH_SIZE / 2),
                shuffle = False,
                num_workers = 4,
            )

            # model setup
            model = model_class()
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(),lr = lr)
            # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
            #                                             step_size = 5,
            #                                             gamma = 0.75)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, 
                                                                            eta_min=0.001, last_epoch=-1)                                       
            criterion = torch.nn.MultiLabelSoftMarginLoss()

            valid_acc_max = 0

            # train 시작
            for epoch in range(epochs):
                train_acc_list = []
                model.train()
                print("-" * 50)
                # trainloader를 통해 batch_size 만큼의 훈련 데이터를 읽어온다
                with tqdm(train_data_loader,total=train_data_loader.__len__(), unit="batch") as train_bar:
                    for batch_idx, batch_data in enumerate(train_bar):
                        train_bar.set_description(f"Train Epoch {epoch}")
                        images, labels = batch_data['image'], batch_data['label']
                        #images, labels = images.to(device), labels.to(device)
                        images, labels = Variable(images.cuda()), Variable(labels.cuda())
                        optimizer.zero_grad()
                        with torch.set_grad_enabled(True):
                            # 모델 예측
                            probs  = model(images)
                            # loss 계산
                            loss = criterion(probs, labels)
                            # 중간 노드의 gradient로
                            # backpropagation을 적용하여
                            # gradient 계산
                            loss.backward()
                            # weight 갱신
                            optimizer.step()

                            # train accuracy 계산
                            probs  = probs.cpu().detach().numpy()
                            labels = labels.cpu().detach().numpy()
                            preds = probs > 0.5
                            batch_acc = (labels == preds).mean()
                            train_acc_list.append(batch_acc)
                            train_acc = np.mean(train_acc_list)

                        # 현재 progress bar에 현재 미니배치의 loss 결과 출력
                        train_bar.set_postfix(train_loss= loss.item(), train_acc = train_acc)
                                                                                                   
                # epoch마다 valid 계산
                valid_acc_list = []
                model.eval()
                with tqdm(valid_data_loader,total=valid_data_loader.__len__(), unit="batch") as valid_bar:
                    for batch_idx, batch_data in enumerate(valid_bar):
                        valid_bar.set_description(f"Valid Epoch {epoch}")
                        images, labels = batch_data['image'], batch_data['label']
                        images, labels = Variable(images.cuda()), Variable(labels.cuda())
                        with torch.no_grad():
                            probs  = model(images)
                            valid_loss = criterion(probs, labels)

                            probs  = probs.cpu().detach().numpy()
                            labels = labels.cpu().detach().numpy()
                            preds = probs > 0.5
                            batch_acc = (labels == preds).mean()
                            valid_acc_list.append(batch_acc)

                        valid_acc = np.mean(valid_acc_list)
                        valid_bar.set_postfix(valid_loss = valid_loss.item(), valid_acc = valid_acc)
                
                # Learning rate 조절
                lr_scheduler.step()  

                # 모델 저장
                if valid_acc_max < valid_acc:
                    valid_acc_max = valid_acc
                    best_model = model
                    create_directory("model")
                    # model_name_bagging_kfold_bestmodel_valid loss로 이름 지정
                    best_model_name = "model/model_{}_{}_{}_{:.4f}.pth".format(CODER, b, fold_index, valid_loss.item())
                    torch.save(model.state_dict(), best_model_name)
                    
                    if os.path.isfile(previse_name):
                        os.remove(previse_name)

                    # 갱신
                    previse_name = best_model_name
            # 폴드별로 가장 좋은 모델 저장
            #best_models.append(best_model)


def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)