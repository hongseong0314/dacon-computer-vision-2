import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob

import torch

def evaluation(model_class, test_data_loader=None, m=False, path=None, score=None, device=None):
    """
    model_class : 사용할 모델
    test_data_loader : 데이터 로더
    m : 멀티 GPU사용
    path : 모델 파라미터 저장 경로
    score : greedy 앙상블 시 각 모델의 정확도 값
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
            pred_scores = []
            
            for batch_idx, batch_data in enumerate(test_data_loader):
                with torch.no_grad():
                    # 추론
                    model.eval()
                    images = batch_data['image']
                    images = images.to(device)
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
        return final_pred
    else:
        def softmax(x):
            return np.exp(x) / np.sum(np.exp(x))

        model_acc_weight = softmax(score)
        for e, m in enumerate(path):
            model.load_state_dict(torch.load(m))
            pred_scores = []
         
            for batch_idx, batch_data in enumerate(test_data_loader):
                with torch.no_grad():
                    # 추론
                    model.eval()
                    images = batch_data['image']
                    images = images.to(device)
                    probs  = model(images)
                    probs = probs.cpu().detach().numpy()
                pred_scores.append(probs)

            
            # 앙상블 0
            if e == 0:
                final_pred = np.vstack(pred_scores) * model_acc_weight[e]
                # final_test_fnames = test_fnames
            else:
                final_pred += np.vstack(pred_scores) * model_acc_weight[e]
        
        return final_pred

# confusion matrix 확인
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score, accuracy_score
from sklearn.metrics import classification_report, multilabel_confusion_matrix

from sklearn.preprocessing import Binarizer

# 정확도, 정밀도, 재현율, F1 점수를 출력
def get_clf_eval(y_test , pred):
    accuracy = accuracy_score(y_test , pred)
    precision = precision_score(y_test , pred)
    recall = recall_score(y_test , pred)
    f1 = f1_score(y_test,pred)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1:{3:.4f}'.format(accuracy, precision, recall, f1))
    

# 임계값 변경   
def get_eval_threshold(y_test, pred, thresholds):
    for custom_thr in thresholds:
        cus_pred = Binarizer(threshold=custom_thr).fit_transform(pred)
        print("*" * 50)
        print(f"Thresholds: {custom_thr}")
        get_clf_eval(y_test, cus_pred)
        
# 다중레이블 분류 혼동행렬 및 분류 리포트 출력
def get_clf_eval_multi(y_test, pred, labels):
    confusion = multilabel_confusion_matrix(y_test, pred)
    # labels = dirty_mnist_answer.columns[1:].tolist()
    conf_mat={}
    for label_col in range(len(labels)):
        y_true_label = y_test[:, label_col]
        y_pred_label = pred[:, label_col]
        conf_mat[labels[label_col]] = confusion_matrix(y_true=y_true_label, y_pred=y_pred_label)
    for label, mat in conf_mat.items():
        print(f"Confusion mat for label : {label}")
        print(mat)
    print("---------Report---------")
    print(classification_report(y_test, pred, target_names=labels))

# ROC 커브 출력
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--')
    plt.axis([0,1,0,1])
    plt.xlabel("False Positive Rate", fontsize=16)
    plt.ylabel("True Positive Rate", fontsize=16)
    plt.grid(True)