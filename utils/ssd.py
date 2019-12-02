#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import stuff
import os
import numpy as np

#import pandas as pd
from math import sqrt as sqrt
from itertools import product as product

import torch
import torch.utils.data as data
from torchvision import models
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function
import pandas as pd


# # SSDメインモデルを構築
# VGGベースモデルを構築。
# 
# TODO: Resnetベースへの改良？どこにつなげればよいか、正規化などを入れないといけないので参考文献などを読みたい。\
# ![SSD](https://image.slidesharecdn.com/05-singleshotmultiboxdetector-161028144820/95/ssd-single-shot-multibox-detector-upc-reading-group-20-638.jpg?cb=1477743905)

# In[2]:


# plot vgg model
vgg = models.vgg16(pretrained=False)


# In[3]:


# VGG
# 転移学習なしで学習するのか？
def make_vgg():
    layers = []
    in_channels = 3
    
    # VGGのモデル構造を記入
    cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "MC", 512, 512, 512, "M", 512, 512, 512]
    
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "MC":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)] #メモリ節約
            in_channels = v
    
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return nn.ModuleList(layers)

# 動作確認
vgg_test = make_vgg()

# 小さい物体のbbox検出用のextras moduleを追加
def make_extras():
    layers = []
    in_channels = 1024 # vgg module outputs
    
    # extra modeule configs
    cfg = [256, 512, 128, 256, 128, 256, 128, 256]
    
    layers += [nn.Conv2d(in_channels, cfg[0], kernel_size=1)]
    layers += [nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[1], cfg[2], kernel_size=1)]
    layers += [nn.Conv2d(cfg[2], cfg[3], kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[3], cfg[4], kernel_size=1)]
    layers += [nn.Conv2d(cfg[4], cfg[5], kernel_size=3)]
    layers += [nn.Conv2d(cfg[5], cfg[6], kernel_size=1)]
    layers += [nn.Conv2d(cfg[6], cfg[7], kernel_size=3)]
    
    return nn.ModuleList(layers)

extras_test = make_extras()


# ## locとconfに対するモジュール。

# In[5]:


# locとconfモジュールを作成

def make_loc_conf(num_classes=21, bbox_aspect_num=[4, 6, 6, 6, 4, 4]):
    loc_layers = []
    conf_layers = []
    
    # VGGの中間出力に対するレイヤ
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[0] * 4,
                            kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[0] * num_classes,
                             kernel_size=3, padding=1)]
    
    # VGGの最終そうに対するCNN
    loc_layers += [nn.Conv2d(1024, bbox_aspect_num[1] * 4,
                            kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(1024, bbox_aspect_num[1] * num_classes,
                             kernel_size=3, padding=1)]
    
    # source3
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[2] * 4,
                            kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[2] * num_classes,
                             kernel_size=3, padding=1)]
    # source4
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[3] * 4,
                            kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[3] * num_classes,
                             kernel_size=3, padding=1)]
    # source5
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[4] * 4,
                            kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[4] * num_classes,
                             kernel_size=3, padding=1)]
    # source6
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[5] * 4,
                            kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[5] * num_classes,
                             kernel_size=3, padding=1)]
    
    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)

loc_test, conf_test = make_loc_conf()


# ## L2 normの実装

# In[6]:


# ありなしで性能はどう変化するのか？
# 自作レイヤ
class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale
        self.reset_parameters()
        self.eps = 1e-10
        
    def reset_parameters(self):
        init.constant_(self.weight, self.scale) # weightの値が全てscaleになる
        
    def forward(self, x):
        """
        38x38の特徴量に対し、チャネル方向の和を求めそれを元に正規化する。
        また正規化したあとに係数（weight)をかける(ロスが減るように学習してくれるみたい)
        
        """
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps #チャネル方向の自乗和
        x = torch.div(x, norm) # 正規化
        
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) # 学習させるパラメータ
        out = weights * x
        
        return out

# binding boxを出力するクラス

class DBox(object):
    def __init__(self, cfg):
        super(DBox, self).__init__()
        
        self.image_size = cfg["input_size"]
        # 各sourceの特徴量マップのサイズ
        self.feature_maps = cfg["feature_maps"]
        self.num_priors = len(cfg["feature_maps"]) # number of sources
        self.steps = cfg["steps"] #各boxのピクセルサイズ
        self.min_sizes = cfg["min_sizes"] # 小さい正方形のサイズ
        self.max_sizes = cfg["max_sizes"] # 大きい正方形のサイズ
        self.aspect_ratios = cfg["aspect_ratios"]
        
    def make_dbox_list(self):
        mean = []
        # feature maps = 38, 19, 10, 5, 3, 1
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                # fxf画素の組み合わせを生成
                
                f_k = self.image_size / self.steps[k]
                # 300 / steps: 8, 16, 32, 64, 100, 300
                
                # center cordinates normalized 0~1
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                
                # small bbox [cx, cy, w, h]
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]
                
                # larger bbox
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]
                
                # その他のアスペクト比のdefbox
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
                    
        # convert the list to tensor
        output = torch.Tensor(mean).view(-1, 4)
        
        # はみ出すのを防ぐため、大きさを最小0, 最大1にする
        output.clamp_(max=1, min=0)
        
        return output
                    
def softnms(bboxes, scores, nms_threshold=0.3, soft_threshold=0.3, sigma=0.5, mode='union'):
    """
    soft-nms implentation according the soft-nms paper
    :param bboxes:
    :param scores:
    :param labels:
    :param nms_threshold:
    :param soft_threshold:
    :return:
    """


    box_keep = []
    labels_keep = []
    scores_keep = []

    c_boxes = bboxes
    c_scores = scores
    weights = c_scores.clone()
    x1 = c_boxes[:, 0]
    y1 = c_boxes[:, 1]
    x2 = c_boxes[:, 2]
    y2 = c_boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    _, order = weights.sort(0, descending=True)
    
    while order.numel() > 0:
        try:
            i = order[0]
        except:
            i = order
        box_keep.append(c_boxes[i])
        #labels_keep.append(c)
        scores_keep.append(c_scores[i])

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1 + 1).clamp(min=0)
        h = (yy2 - yy1 + 1).clamp(min=0)
        inter = w * h

        if mode == 'union':
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            ovr = inter / areas[order[1:]].clamp(max=areas[i])
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)

        ids_t= (ovr>=nms_threshold).nonzero().squeeze()

        weights[[order[ids_t+1]]] *= torch.exp(-(ovr[ids_t] * ovr[ids_t]) / sigma)

        ids = (weights[order[1:]] >= soft_threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        c_boxes = c_boxes[order[1:]][ids]
        c_scores = weights[order[1:]][ids]
        _, order = weights[order[1:]][ids].sort(0, descending=True)
        if c_boxes.dim()==1:
            c_boxes=c_boxes.unsqueeze(0)
            c_scores=c_scores.unsqueeze(0)
        x1 = c_boxes[:, 0]
        y1 = c_boxes[:, 1]
        x2 = c_boxes[:, 2]
        y2 = c_boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    return box_keep, scores_keep              
        


# In[9]:


# test boxes

# SSD300の設定
ssd_cfg = {
    'num_classes': 21,  # 背景クラスを含めた合計クラス数
    'input_size': 300,  # 画像の入力サイズ
    'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # 出力するDBoxのアスペクト比の種類
    'feature_maps': [38, 19, 10, 5, 3, 1],  # 各sourceの画像サイズ
    'steps': [8, 16, 32, 64, 100, 300],  # DBOXの大きさを決める
    'min_sizes': [30, 60, 111, 162, 213, 264],  # DBOXの大きさを決める
    'max_sizes': [60, 111, 162, 213, 264, 315],  # DBOXの大きさを決める
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
}

dbox = DBox(ssd_cfg)
dbox_list = dbox.make_dbox_list()

pd.DataFrame(dbox_list.numpy())


# # SSDクラスを実装する

# In[10]:


class SSD2(nn.Module):
    def __init__(self, phase, cfg):
        super(SSD2, self).__init__()
        
        self.phase = phase
        self.num_classes = cfg["num_classes"]
        
        # call SSD network
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(self.num_classes, cfg["bbox_aspect_num"])
        
        # make Dbox
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()
        
        # use Detect if inference
        if phase == "inference":
            self.detect = Detect()

# check operation
ssd_test = SSD2(phase="train", cfg=ssd_cfg)
#print(ssd_test)


# In[27]:


def decode(loc, dbox_list):
    """
    DBox(cx,cy,w,h)から回帰情報のΔを使い、
    BBox(xmin,ymin,xmax,ymax)方式に変換する。
    
    loc: [8732, 4] [Δcx, Δcy, Δw, Δheight]
    SSDのオフセットΔ情報
    
    dbox_list: (cx,cy,w,h)
    """
    
    boxes = torch.cat((
    dbox_list[:, :2] + loc[:, :2] * 0.1 * dbox_list[:, :2],
    dbox_list[:, 2:] * torch.exp(loc[:, 2:] * 0.2)), dim=1)
    
    # convert boxes to (xmin,ymin,xmax,ymax)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    
    return boxes

# test
dbox = DBox(ssd_cfg)
dbox_list = dbox.make_dbox_list()


loc = torch.ones(8732, 4)
loc[0, :] = torch.tensor([-10, 0, 1, 1])

dbox_process = decode(loc, dbox_list)

pd.DataFrame(dbox_process.numpy())


# In[32]:


def nms(boxes, scores, overlap=0.45, top_k=200):
    """
    Non-Maximum Suppressionを行う関数。
    boxesのうち被り過ぎ（overlap以上）のBBoxを削除する。

    Parameters
    ----------
    boxes : [確信度閾値（0.01）を超えたBBox数,4]
        BBox情報。
    scores :[確信度閾値（0.01）を超えたBBox数]
        confの情報

    Returns
    -------
    keep : リスト
        confの降順にnmsを通過したindexが格納
    count：int
        nmsを通過したBBoxの数
    """

    # returnのひな形を作成
    count = 0
    keep = scores.new(scores.size(0)).zero_().long()
    # keep：torch.Size([確信度閾値を超えたBBox数])、要素は全部0

    # 各BBoxの面積areaを計算
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)

    # boxesをコピーする。後で、BBoxの被り度合いIOUの計算に使用する際のひな形として用意
    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    # socreを昇順に並び変える
    v, idx = scores.sort(0)

    # 上位top_k個（200個）のBBoxのindexを取り出す（200個存在しない場合もある）
    idx = idx[-top_k:]

    # idxの要素数が0でない限りループする
    while idx.numel() > 0:
        i = idx[-1]  # 現在のconf最大のindexをiに

        # keepの現在の最後にconf最大のindexを格納する
        # このindexのBBoxと被りが大きいBBoxをこれから消去する
        keep[count] = i
        count += 1

        # 最後のBBoxになった場合は、ループを抜ける
        if idx.size(0) == 1:
            break

        # 現在のconf最大のindexをkeepに格納したので、idxをひとつ減らす
        idx = idx[:-1]

        # -------------------
        # これからkeepに格納したBBoxと被りの大きいBBoxを抽出して除去する
        # -------------------
        # ひとつ減らしたidxまでのBBoxを、outに指定した変数として作成する
        torch.index_select(x1, 0, idx, out=tmp_x1)
        torch.index_select(y1, 0, idx, out=tmp_y1)
        torch.index_select(x2, 0, idx, out=tmp_x2)
        torch.index_select(y2, 0, idx, out=tmp_y2)

        # すべてのBBoxに対して、現在のBBox=indexがiと被っている値までに設定(clamp)
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i])

        # wとhのテンソルサイズをindexを1つ減らしたものにする
        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)

        # clampした状態でのBBoxの幅と高さを求める
        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1

        # 幅や高さが負になっているものは0にする
        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)

        # clampされた状態での面積を求める
        inter = tmp_w*tmp_h

        # IoU = intersect部分 / (area(a) + area(b) - intersect部分)の計算
        rem_areas = torch.index_select(area, 0, idx)  # 各BBoxの元の面積
        union = (rem_areas - inter) + area[i]  # 2つのエリアのANDの面積
        IoU = inter/union

        # IoUがoverlapより小さいidxのみを残す
        idx = idx[IoU.le(overlap)]  # leはLess than or Equal toの処理をする演算です
        # IoUがoverlapより大きいidxは、最初に選んでkeepに格納したidxと同じ物体に対してBBoxを囲んでいるため消去

    # whileのループが抜けたら終了

    return keep, count


# # 推論用のクラスDetectの実装

# In[35]:

#from utils.nms2 import nms as nms2

class Detect(Function):
    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        self.softmax = nn.Softmax(dim=-1)
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        
    def forward(self, loc_data, conf_data, dbox_list):
        """
        SSDの推論結果を受け取り、bboxのデコードとnms処理を行う。
        """
        # 
        num_batch = loc_data.size(0)
        num_dbox = loc_data.size(1)
        num_classes = conf_data.size(2)
        
        # confをsoftmaxを使って正規化
        conf_data = self.softmax(conf_data)
        
        # 出力の方を作成する
        # [batch, class, topk, 5]
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)
        
        # conf_dataを[batch, 8732, classes]から[batch, classes, 8732]に変更
        conf_preds = conf_data.tranpose(2, 1)
        
        # batch毎にループ
        for i in range(num_batch):
            # 1. LocとDBoxからBBox情報に変換
            decoded_boxes = decode(loc_data, dbox_list)
            
            # confのコピー
            conf_scores = conf_preds[i].clone()
            
            # classごとにデコードとNMSを回す。
            for cl in range(1, num_classes): # 背景は飛ばす。
                # 2. 敷地を超えた結果を取り出す
                c_mask = conf_scores[cl].gt(self.conf_thresh) # gt=greater than
                # index maskを作成した。
                # threshを超えると1, 超えなかったら0に。
                # c_mask = [8732]
                
                scores = conf_scores[cl][c_mask]
                
                # sort scores and boxes
                _, order = torch.sort(scores, 0, True)
                
                
                if scores.nelement() == 0:
                    continue
                    # 箱がなかったら終わり。
                    
                # cmaskをboxに適応できるようにサイズ変更
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                # l_mask.size = [8732, 4]
                boxes = decoded_boxes[l_mask].view(-1, 4) # reshape to [boxnum, 4]
                
                # 3. NMSを適応する
                ids, count = nms(boxes[order], scores[order], self.nms_thresh, self.top_k)
                #keep = nms_gpu(boxes[order], scores[order])
                
                # torch.cat(tensors, dim=0, out=None) → Tensor
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 
                                                 1)
                
        return output # torch.size([batch, 21, 200, 5])
                
class Detect_Flip(nn.Module):
    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45, TTA=True, softnms=False):
        super(Detect_Flip, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        self.TTA = TTA
        self.softnms = softnms
        if TTA:
            print("test time flip is ON.")
        else:
            print("test time flip is OFF.")
        print("nms thresh is :", nms_thresh)
        print("soft nms is : ", softnms)
            
    def forward(self, loc_data, conf_data, loc_data2, conf_data2, dbox_list):
        """
        SSDの推論結果を受け取り、bboxのデコードとnms処理を行う。
        """
        # 
        num_batch = loc_data.size(0)
        num_dbox = loc_data.size(1)
        num_classes = conf_data.size(2)
        
        # confをsoftmaxを使って正規化
        conf_data = self.softmax(conf_data)
        conf_data2 = self.softmax(conf_data2)
        
        # 出力の方を作成する
        # [batch, class, topk, 5]
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)
        
        # conf_dataを[batch, 8732, classes]から[batch, classes, 8732]に変更
        conf_preds = conf_data.transpose(2, 1)
        conf_preds2 = conf_data2.transpose(2, 1)
        
        # batch毎にループ
        for i in range(num_batch):
            # 1. LocとDBoxからBBox情報に変換
            decoded_boxes = decode(loc_data[i], dbox_list)
            decoded_boxes2 = decode(loc_data2[i], dbox_list)
            
            
            # confのコピー
            conf_scores = conf_preds[i].clone()
            conf_scores2 = conf_preds2[i].clone()
            
            # classごとにデコードとNMSを回す。
            for cl in range(1, num_classes): # 背景は飛ばす。
                # 2. 敷地を超えた結果を取り出す
                c_mask = conf_scores[cl].gt(self.conf_thresh) # gt=greater than
                c_mask2 = conf_scores2[cl].gt(self.conf_thresh) # gt=greater than
                # index maskを作成した。
                # threshを超えると1, 超えなかったら0に。
                # c_mask = [8732]
                
                scores = conf_scores[cl][c_mask]
                scores2 = conf_scores2[cl][c_mask2]               
                               
                if scores.nelement() == 0:
                    continue
                    # 箱がなかったら終わり。
                    
                # cmaskをboxに適応できるようにサイズ変更
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                l_mask2 = c_mask2.unsqueeze(1).expand_as(decoded_boxes2)
                # l_mask.size = [8732, 4]
                boxes = decoded_boxes[l_mask].view(-1, 4) # reshape to [boxnum, 4]
                boxes2 = decoded_boxes2[l_mask2].view(-1, 4) # reshape to [boxnum, 4]
                
                # boxes2 are flipped.. fix that.
                tmpbox = boxes2
                #try:
                #    print("min box x", np.min(tmpbox.cpu().numpy()[:, 0]))
                #    print("max box x", np.max(tmpbox.cpu().numpy()[:, 2]))
                #except:
                #    print("no box")
                boxes2[:, 0] = 1 - tmpbox[:, 2]
                boxes2[:, 2] = 1 - tmpbox[:, 0]
                              
                # concat boxes and score
                if self.TTA:
                    boxes = torch.cat((boxes, boxes2), 0)
                    scores = torch.cat((scores, scores2), 0)
                
                #print(boxes.shape)
                #print(scores.shape)
                
                if not self.softnms:
                    # 3. NMSを適応する
                    ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                    #count = len(ids)
                    ##torch.cat(tensors, dim=0, out=None) → Tensor
                    output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)
                
                # 4. soft-nmsを適応する
                else:
                    boxes, scores = softnms(boxes, scores)                
                    if boxes == []:
                        continue                   
                    boxes = torch.stack(boxes)
                    scores = torch.stack(scores)

                    output[i, cl, :len(scores)] = torch.cat((scores.unsqueeze(1), boxes), 1)
                
        return output      
                
                


# In[37]:

import torch.nn as nn

class SSD(nn.Module):
    def __init__(self, phase, cfg):
        super(SSD, self).__init__()
        
        self.phase = phase
        self.num_classes = cfg["num_classes"]
        
        # call SSD network
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(self.num_classes, cfg["bbox_aspect_num"])
        
        # make Dbox
        dbox = DBox(cfg)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dbox_list = dbox.make_dbox_list()
        
        # use Detect if inference
        if phase == "inference":
            self.detect = Detect()
            
    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()
        
        # VGGのconv4_3まで計算
        for k in range(23):
            x = self.vgg[k](x)
        
        # conv4_3の出力をL2Normに入力。source1をsourceに追加
        source1 = self.L2Norm(x)
        sources.append(source1)
        
        # VGGを最後まで計算しsource2を取得
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        
        sources.append(x)
        
        # extra層の計算を行う。
        # source3-6に結果を格納。
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace = True)
            if k % 2 == 1:
                sources.append(x)
        
        # source 1-6にそれぞれ対応するconvを適応しconfとlocを得る。
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # Permuteは要素の順番を入れ替え
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        
        # convの出力は[batch, 4*anker, fh, fw]なので整形しなければならない。
        # まず[batch, fh, fw, anker]に整形
        
        # locとconfの形を変形
        # locのサイズは、torch.Size([batch_num, 34928])
        # confのサイズはtorch.Size([batch_num, 183372])になる
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        
        # さらにlocとconfの形を整える
        # locのサイズは、torch.Size([batch_num, 8732, 4])
        # confのサイズは、torch.Size([batch_num, 8732, 21])
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)
        # これで後段の処理につっこめるかたちになる。
        
        output = (loc, conf, self.dbox_list)
        
        if self.phase == "inference":
            # Detectのforward
            return self.detect(output[0], output[1], output[2].to(self.device))
        else:
            return output

class SSDFlip(nn.Module):
    """二回推論を行うクラス"""
    def __init__(self, eval_categories, net, device):
        super(SSDFlip, self).__init__()  # 親クラスのコンストラクタ実行
        print(device)
        self.net = net.to(device)  # SSDネットワーク
        self.device = device
        
    def ssd_predict(self, x):
        detections = self.net(x)
        
        # flip image
        x = x[:, :, ::-1, :] # x = [batch, height, width, channel]
        flipped_detections = self.net(x)
        
        # flip detections
        
        # confidence_levelが基準以上を取り出す
        predict_bbox = []
        pre_dict_label_index = []
        scores = []
        detections = detections.cpu().detach().numpy()

        # 条件以上の値を抽出
        find_index = np.where(detections[:, 0:, :, 0] >= data_confidence_level)
        detections = detections[find_index]
        for i in range(len(find_index[1])):  # 抽出した物体数分ループを回す
            if (find_index[1][i]) > 0:  # 背景クラスでないもの
                sc = detections[i][0]  # 確信度
                bbox = detections[i][1:] * [width, height, width, height]
                # find_indexはミニバッチ数、クラス、topのtuple
                lable_ind = find_index[1][i]-1
                # （注釈）
                # 背景クラスが0なので1を引く

                # 返り値のリストに追加
                predict_bbox.append(bbox)
                pre_dict_label_index.append(lable_ind)
                scores.append(sc)

        return rgb_img, predict_bbox, pre_dict_label_index, scores




