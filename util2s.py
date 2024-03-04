import copy
import math
from os.path import join as pjoin
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import os
import yaml
# from yacs.config import CfgNode as CN
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
from torchvision import transforms
import torch.utils.data as data
import scipy.io as sio
import matplotlib.pyplot as plt
# from GPUtil import showUtilization as gpu_usage
# from numba import cuda
import time
import logging
import sys
from datetime import datetime

import pandas as pd




class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
    
    
    

def calculate_F1_score(prediction, label ,eps=1e-7, threshold = 0.5):
    prediction, label= _list_tensor(prediction, label)
    prediction = _threshold(prediction, threshold=threshold)
    label = _threshold(label, threshold=threshold)

    intersection = torch.logical_and(prediction, label)
    dice = 2 * torch.sum(intersection) / (torch.sum(prediction) + torch.sum(label))
    return dice

def calculate_acc(prediction, label,eps=1e-7, threshold = 0.5):
    prediction, label = _list_tensor(prediction, label)
    prediction = _threshold(prediction, threshold=threshold)
    label = _threshold(label, threshold=threshold)
    _,_,h, w = label.shape
    _,_,x, y = prediction.shape
    assert h == x and w == y

    total = h * w
    correct = torch.sum(prediction == label)
    return correct / total

def _list_tensor(x, y):
    m = torch.nn.Sigmoid()
    if type(x) is list:
        x = torch.tensor(np.array(x))
        y = torch.tensor(np.array(y))
        if x.min() < 0:
            x = m(x)
    else:
        x, y = x, y
        if x.min() < 0:
            x = m(x)
    return x, y

def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x

def calculate_IoU(prediction, label):
    h, w = label.shape
    x, y = prediction.shape
    assert h == x and w == y
    intersection = np.logical_and(prediction, label)
    union = np.logical_or(prediction, label)
    if np.sum(union) == 0:
        assert np.sum(intersection) == 0
        IoU = 1
    else:
        IoU = np.sum(intersection) / np.sum(union)
    return IoU

def iou(pr, gt, eps=1e-7, threshold = 0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_,dim=[1,2,3])
    union = torch.sum(gt_,dim=[1,2,3]) + torch.sum(pr_,dim=[1,2,3]) - intersection
    return ((intersection + eps) / (union + eps)).cpu().numpy()

def AJI(gt, output):
    n_ins = gt.max()
    n_out = output.max()
    if n_out == 0:
        if n_ins == 0:
            return 1
        else:
            return 0
    empty = 0
    Iand = 0
    Ior = 0
    for i in range(n_out):
        out_table = np.where(output == i +1, 1, 0)
        max_and = 0
        max_or = 0
        for j in range(n_ins):
            gt_table = np.where(gt == j +1, 1, 0)
            ior1 = np.sum(out_table + gt_table > 0)
            iand1 = np.sum(out_table) + np.sum(gt_table) - ior1
            if (iand1 > max_and):
                max_and = iand1
                max_or = ior1
        if max_and == 0:
            empty = empty + np.sum(out_table)
        Iand += max_and
        Ior += max_or
    return Iand / (Ior + empty)
def find_con(img,mask=None):
    img = np.uint8(img)
    num1, labels1 = cv2.connectedComponents(img)
    labels1 = labels1.astype('uint8')
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    labels=cv2.dilate(labels1, kernel2, iterations=1)

    #cv2.imshow('dst', labels.astype('uint8'))
    #cv2.waitKey(0)
    # if not os.path.exists(savepath):
    #     os.mkdir(savepath)
    # cv2.imwrite(os.path.join(savepath, mask), labels)
    return labels

def draw_iou(iou,now):
    iou_series = pd.Series(iou)

    # 应用移动平均平滑。这里的窗口大小可以根据需要调整
    iou = iou_series.rolling(window=5).mean()

    plt.subplot(2, 1, 2)
    plt.plot(iou, label='iou')

    plt.legend(loc='upper right')
    plt.ylabel('iou')
    plt.title('iou')
    plt.xlabel('epoch')
    plt.savefig('./log/no_train_iou{}.png'.format(now), dpi=100)
    plt.show()


def draw_acc(acc,now):
    acc_series = pd.Series(acc)

    # 应用移动平均平滑。这里的窗口大小可以根据需要调整
    acc = acc_series.rolling(window=5).mean()

    plt.subplot(2, 1, 2)
    plt.plot(acc, label='acc')

    plt.legend(loc='upper right')
    plt.ylabel('ACC')
    plt.title('acc')
    plt.xlabel('epoch')
    plt.savefig('./log/no_train_acc{}.png'.format(now), dpi=100)
    plt.show()
def draw_f1(f1,now):
    f1_series = pd.Series(f1)

# 应用移动平均平滑。这里的窗口大小可以根据需要调整
    f1 = f1_series.rolling(window=5).mean()

    plt.subplot(2, 1, 2)
    plt.plot(f1, label='F1')

    plt.legend(loc='upper right')
    plt.ylabel('F1')
    plt.title('F1')
    plt.xlabel('epoch')
    plt.savefig('./log/no_train_F1{}.png'.format(now), dpi=100)
    plt.show()


def draw_loss(train_loss,val_loss,now):
    plt.subplot(2, 1, 2)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('./log/TrainLoss_{}.png'.format(now), dpi=100)
    plt.show()


def get_iou(predict, label):
    predict_f = predict.flatten()
    label_f = label.flatten()
    intersection = np.sum(predict_f*label_f)
    iou = intersection/(np.sum(predict_f) + np.sum(label_f) - intersection)
    # print("intersection ",intersection)
    # print("get_iou ",iou)
    return iou

kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
def sem2ins(seg_mask,nem,cem,sharpen=0):
    if 0:
        result = seg_mask.copy()
    else:
        
        if sharpen == 1:
            image_sharp = cv2.filter2D(src=nem+cem, ddepth=-1, kernel=kernel)
            result = seg_mask - image_sharp
        elif sharpen == 0:
            result = seg_mask - nem - cem
        else:
            result = seg_mask.copy()
    result[result > 0] = 1
    result[result<0 ] = 0
    cv2.imwrite("1.png",255*result)

    seg_mask2 = cv2.imread("1.png")
    # print(seg_mask.shap e)
    seg_mask_g = cv2.cvtColor(seg_mask2,cv2.COLOR_BGR2GRAY) 
    contours, hierarchy = cv2.findContours(seg_mask_g, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    for i in range(len(contours)):
        cnt = contours[i]
        seg_mask_g = cv2.drawContours(seg_mask_g, [cnt], 0, i, -1)
        # print(seg_mask_g.shape)
        # cv2_imshow(seg_mask_g)  
        # print(set(seg_mask_g.flatten()))
        # seg_mask_g = cv2.cvtColor(seg_mask,cv2.COLOR_BGR2GRAY) 
    return seg_mask_g

def sem2ins_smooth(seg_mask,nem,cem):
    edge = nem + cem
    edge = np.float32(edge)
    kernel = np.ones((7,7),np.float32)/49
    smoth_edge = cv2.filter2D(edge,-1,kernel)

    result = seg_mask - smoth_edge

    result[result > 0] = 1
    result[result<0 ] = 0
    cv2.imwrite("1.png",255*result)

    seg_mask2 = cv2.imread("1.png")
    # print(seg_mask.shap e)
    seg_mask_g = cv2.cvtColor(seg_mask2,cv2.COLOR_BGR2GRAY) 
    contours, hierarchy = cv2.findContours(seg_mask_g, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    n = 0
    
    for i in range(len(contours)):
        cnt = contours[i]
        # print(cnt.shape)
        if cnt.shape[0] < 9:
          continue
        # print(n)
        seg_mask_g = cv2.drawContours(seg_mask_g, [cnt], 0, n, -1)
        n+=1
        # print(seg_mask_g.shape)
        # cv2_imshow(seg_mask_g)  
        # print(set(seg_mask_g.flatten()))
        # seg_mask_g = cv2.cvtColor(seg_mask,cv2.COLOR_BGR2GRAY) 
    seg_mask_g[seg_mask_g==255] = n+1
    return seg_mask_g

def sem2ins_smooth_con(seg_mask,nem,cem):
    edge = nem + cem
    edge = np.float32(edge)
    kernel = np.ones((7,7),np.float32)/49
    smoth_edge = cv2.filter2D(edge,-1,kernel)

    result = seg_mask - smoth_edge

    result[result > 0] = 1
    result[result<0 ] = 0
    seg_mask_g = find_con(result)

    
    return seg_mask_g


    # for i in range(len(contours)):
    #     cnt = contours[i]
    #     seg_mask_g = cv2.drawContours(seg_mask_g, [cnt], 0, i, -1)
    #     # print(seg_mask_g.shape)
    #     # cv2_imshow(seg_mask_g)  
    #     # print(set(seg_mask_g.flatten()))
    #     # seg_mask_g = cv2.cvtColor(seg_mask,cv2.COLOR_BGR2GRAY) 
    # return seg_mask_g

def _bd_loss(pred, target):
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = -(torch.sum(target[i]*torch.log(pred[i]+1e-6) + (1-target[i])*torch.log(1-pred[i]+1e-6)))
        IoU = IoU + Iand1/512/512

    return IoU/b

class BD(torch.nn.Module):
    def __init__(self, size_average = True):
        super(BD, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        return _bd_loss(pred, target)

def bd_loss(pred,label):
    loss = BD(size_average=True)
    bd_out = loss(pred, label)
    return bd_out

# CIA loss
def _cia_loss(pred, target, w):
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        classes = target[i] > 0
        Iand1 = -torch.sum(classes*torch.log(pred[i][0]+1e-6)/(torch.sum(classes)+1) + ~classes*torch.log(1-pred[i][0]+1e-6)/(torch.sum(~classes)+1))
        # print('class{}: {}'.format(j,Iand1))
        IoU = IoU + (1-w)*Iand1
        
        classes = target[i] > 1
        Iand1 = -torch.sum(classes*torch.log(pred[i][1]+1e-6)/(torch.sum(classes)+1) + ~classes*torch.log(1-pred[i][1]+1e-6)/(torch.sum(~classes)+1))
        # print('class2: {}'.format(Iand1))
        IoU = IoU + w*Iand1            

    return IoU/b

def _st_loss(pred, target, thresh):
    # Smooth Truncated Loss
    b = pred.shape[0]
    ST = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        w = target[i] > 0
        pt = w * pred[i][1]
        w = target[i] > 0
        pt = pt + w*pred[i][0]
        certain = pt > thresh
        Iand1 = -(torch.sum( certain*torch.log(pt+1e-6) + ~certain*(np.log(thresh) - (1-(pt/thresh)**2)/2) ))
        ST = ST + Iand1/512/512
    # print("_st",ST/b)
    return ST/b

class CIA(torch.nn.Module):
    def __init__(self, size_average = True):
        super(CIA, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target, w, thresh, lw):
        # print(_cia_loss(pred, target), _st_loss(pred, target, thresh))
        return _cia_loss(pred, target, w) + lw * _st_loss(pred, target, thresh)

def cia_loss(pred, label, w, thr=0.2, lamb=0.5):
    Cia_loss = CIA(size_average=True)
    cia_out = Cia_loss(pred, label, w, thr, lamb)
    return cia_out


# IOU loss
def _iou(pred, target, size_average = True):
    b = pred.shape[0]
    IoU = 0.0
    # print(target.shape)
    # print(target[0].shape)
    for i in range(0,b):
        #compute the IoU of the foreground
        w = target[i] == 0
        Iand1 = torch.sum(target[i]*pred[i])
        Ior1 = torch.sum(target[i]) + torch.sum(pred[i])-Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b

class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return _iou(pred, target, self.size_average)

def my_loss(pred,label):
    iou_loss = IOU(size_average=True)
    iou_out = iou_loss(pred, label)
    # print("iou_loss:", iou_out.data.cpu().numpy())
    return iou_out


class CIA_LOSS(torch.nn.Module):
    def __init__(self, size_average = True):
        super(CIA_LOSS, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target, thresh=0.5, lw=0.42):
        # print(_cia_loss(pred, target), _st_loss(pred, target, thresh))
        return lw *DiceLoss.dice_loss(pred, target) +  _st_loss(pred, target, thresh)

class CIA_LOSS2(torch.nn.Module):
    def __init__(self, size_average = True):
        super(CIA_LOSS2, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target, thresh=0.5, lw=0.42):
        # print(_cia_loss(pred, target), _st_loss(pred, target, thresh))
        return lw *_iou(pred, target) +  _st_loss(pred, target, thresh)

    
def adjust_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.02

def overlap_pred_gt(pred,gt):
    # print(gt[1].shape)
    pred = 255*np.uint8(pred)
    r_channel = gt[0,:,:]
    g_channel = gt[1,:,:]
    b_channel = gt[2,:,:]
    # r_channel, g_channel, b_channel = img[0,:,:],img[1,:,:],img[2,:,:]
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    gt_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

    h,w = pred.shape[:2]
    
    pred_c = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
    b_channel, g_channel, r_channel = cv2.split(pred_c)
    r_channel[r_channel>-100] = 0
    alpha_channel = np.ones(pred.shape, dtype=pred.dtype) * 5
    alpha_channel[pred==0] = 0
    pred_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    # cv2_imshow(gt_BGRA)
    # print(pred_BGRA.dtype)
    # print(gt_BGRA.dtype)
    gt_BGRA = np.float32(255*gt_BGRA)
    # cv2_imshow(gt_BGRA)
    dst=cv2.addWeighted(pred_BGRA,0.3,gt_BGRA,1,0,dtype=cv2.CV_32F)
    print(dst.shape)
    return dst

def create_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def get_fast_aji(true, pred):
    """AJI version distributed by MoNuSeg, has no permutation problem but suffered from 
    over-penalisation similar to DICE2.
    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4] 
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no 
    effect on the result.
    """
    true = np.copy(true)  # ? do we need this
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)
    # print("l",len(pred_masks))
    # prefill with value
    pairwise_inter = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    pairwise_union = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        # print("l2",len(pred_true_overlap))
        # print(pred_true_overlap)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0 or pred_id >= len(pred_masks):  # ignore
                continue  # overlaping background
            # print(pred_id)
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each true, dont care
    # about reusing pred instance multiple times
    try:
        paired_pred = np.argmax(pairwise_iou, axis=1)
    except:
        return 0
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()

    paired_true = list(paired_true + 1)  # index to instance ID
    paired_pred = list(paired_pred + 1)
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array(
        [idx for idx in true_id_list[1:] if idx not in paired_true]
    )
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    )
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        if pred_id >= len(pred_masks):
            continue
        overall_union += pred_masks[pred_id].sum()

    aji_score = overall_inter / overall_union
    return aji_score

def get_fast_aji_plus(true, pred):
    """AJI+, an AJI version with maximal unique pairing to obtain overall intersecion.
    Every prediction instance is paired with at most 1 GT instance (1 to 1) mapping, unlike AJI 
    where a prediction instance can be paired against many GT instances (1 to many).
    Remaining unpaired GT and Prediction instances will be added to the overall union.
    The 1 to 1 mapping prevents AJI's over-penalisation from happening.
    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4] 
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no 
    effect on the result.
    """
    true = np.copy(true)  # ? do we need this
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_inter = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    pairwise_union = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter
    #
    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    #### Munkres pairing to find maximal unique pairing
    paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
    ### extract the paired cost and remove invalid pair
    paired_iou = pairwise_iou[paired_true, paired_pred]
    # now select all those paired with iou != 0.0 i.e have intersection
    paired_true = paired_true[paired_iou > 0.0]
    paired_pred = paired_pred[paired_iou > 0.0]
    paired_inter = pairwise_inter[paired_true, paired_pred]
    paired_union = pairwise_union[paired_true, paired_pred]
    paired_true = list(paired_true + 1)  # index to instance ID
    paired_pred = list(paired_pred + 1)
    overall_inter = paired_inter.sum()
    overall_union = paired_union.sum()
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array(
        [idx for idx in true_id_list[1:] if idx not in paired_true]
    )
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    )
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()
    #
    aji_score = overall_inter / overall_union
    return aji_score

def get_fast_pq(true, pred, match_iou=0.5):
    """`match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique 
    (1 prediction instance to 1 GT instance mapping).
    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing. 
    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.    
    
    Fast computation requires instance IDs are in contiguous orderding 
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand 
    and `by_size` flag has no effect on the result.
    Returns:
        [dq, sq, pq]: measurement statistic
        [paired_true, paired_pred, unpaired_true, unpaired_pred]: 
                      pairing information to perform measurement
                    
    """
    assert match_iou >= 0.0, "Cant' be negative"

    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_iou = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]


#####
def get_fast_dice_2(true, pred):
    """Ensemble dice."""
    true = np.copy(true)
    pred = np.copy(pred)
    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred))

    overall_total = 0
    overall_inter = 0

    true_masks = [np.zeros(true.shape)]
    for t in true_id[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [np.zeros(true.shape)]
    for p in pred_id[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    for true_idx in range(1, len(true_id)):
        t_mask = true_masks[true_idx]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        try:  # blinly remove background
            pred_true_overlap_id.remove(0)
        except ValueError:
            pass  # just mean no background
        for pred_idx in pred_true_overlap_id:
            p_mask = pred_masks[pred_idx]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            overall_total += total
            overall_inter += inter

    return 2 * overall_inter / overall_total

def gray_to_bgr(gray_img):
    b_img = gray_img.copy()
    g_img = gray_img.copy()
    r_img = gray_img.copy()
    b_img[b_img!=255] = 0
    r_img[gray_img==255] = 255
    r_img[gray_img==76] = 255
    g_img[g_img==255] = 255
    g_img[g_img==150] = 255

    return cv2.merge([b_img,g_img,r_img])




def edge_detection(m,channel = 1):
    # gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
    if len(m.shape) == 2:
        m = np.expand_dims(m, axis=0)
    m = np.uint8(m)
    b,h,w = m.shape
    outputs = np.zeros((b,h,w))
    # m = np.array(m, np.uint8)
    # print("m shape ",m.shape)
    for i in range(b):
        contours, _ = cv2.findContours(m[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blank = np.zeros((512,512))
        # draw the contours on a copy of the original image
        cv2.drawContours(blank, contours, -1, 1, 2)
        outputs[i] = blank
    


    return outputs
    
def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot
