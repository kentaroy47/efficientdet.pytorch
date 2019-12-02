#!/usr/bin/env python
# coding: utf-8

# # ファイルパスリストを作成。
#     画像とアノテーション

# In[1]:


# pip install opencv-python --trusted-host pypi.python.org


# In[1]:


# import stuff
import os
import numpy as np

import torch
import torch.utils.data as data


def make_datapath_list(rootpath):
    img_path_template = os.path.join(rootpath, "JPEGImages", "%s.jpg")
    anno_path_template = os.path.join(rootpath, "Annotations", "%s.xml")
    
    # get id
    train_id_names = os.path.join(rootpath, "ImageSets", "Main", "trainval.txt")
    val_id_names = os.path.join(rootpath, "ImageSets", "Main", "test.txt")
    
    train_img_list = list()
    train_anno_list = list()
    
    for line in open(train_id_names):
        file_id = line.strip()
        img_path = (img_path_template % file_id)
        anno_path = (anno_path_template % file_id)
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)
        
    val_img_list = list()
    val_anno_list = list()
    if os.path.isfile(val_id_names):
        for line in open(val_id_names):
            file_id = line.strip()
            img_path = (img_path_template % file_id)
            anno_path = (anno_path_template % file_id)
            val_img_list.append(img_path)
            val_anno_list.append(anno_path)
        
    return train_img_list, train_anno_list, val_img_list, val_anno_list


# アノテーションファイルをリストに読み込む

# In[4]:


import xml.etree.ElementTree as ET
import cv2

class Anno_xml2list(object):
    def __init__(self, classes):
        self.classes = classes
    
    def __call__(self, xml_path, width, height):
        """
        アノテーションデータを読み込み、サイズを正規化してから出力
        """
        ret = [] # 出力を格納
        
        xml = ET.parse(xml_path).getroot()
        
        for obj in xml.iter("object"):
            difficult = int(obj.find("difficult").text)
            if difficult == 1:
                continue

            # make bb
            bndbox = []

            name = obj.find("name").text.lower().strip()
            #print(name)
            bbox = obj.find("bndbox")

            # アノテーションを取得し正規化
            pts = ["xmin", "ymin", "xmax", "ymax"]

            for pt in pts:
                cur_pixel = int(bbox.find(pt).text) - 1
                if pt == "xmin" or pt == "xmax":
                    cur_pixel /= width
                else:
                    cur_pixel /= height

                bndbox.append(cur_pixel)

            label_idx = self.classes.index(name)
            bndbox.append(label_idx)

            # add to results
            ret += [bndbox]
        return np.asarray(ret)


# フォルダ「utils」にあるdata_augumentation.pyからimport。
# 入力画像の前処理をするクラス
from utils.data_augumentation import Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans

class DatasetTransform():
    def __init__(self, input_size, color_mean):
        self.data_transform = {
        "train": Compose([
            ConvertFromInts(),  # intをfloat32に変換
            ToAbsoluteCoords(),  # アノテーションデータの規格化を戻す
            PhotometricDistort(),  # 画像の色調などをランダムに変化
            Expand(color_mean),  # 画像のキャンバスを広げる
            RandomSampleCrop(),  # 画像内の部分をランダムに抜き出す
            RandomMirror(),  # 画像を反転させる
            ToPercentCoords(),  # アノテーションデータを0-1に規格化
            Resize(input_size),  # 画像サイズをinput_size×input_sizeに変形
            SubtractMeans(color_mean)  # BGRの色の平均値を引き算
        ]),
        "val": Compose([
            ConvertFromInts(),  # intをfloatに変換
            Resize(input_size),  # 画像サイズをinput_size×input_sizeに変形
            SubtractMeans(color_mean)  # BGRの色の平均値を引き算
        ])  
        }
    def __call__(self, img, phase, boxes, labels):
        return self.data_transform[phase](img, boxes, labels)
            
        


# In[7]:


# # create dataset loader.

# In[21]:


class VOCDataset(data.Dataset):
    """
    VOC2012のDatasetを作成するクラス。PyTorchのDatasetクラスを継承。

    Attributes
    ----------
    img_list : リスト
        画像のパスを格納したリスト
    anno_list : リスト
        アノテーションへのパスを格納したリスト
    phase : 'train' or 'test'
        学習か訓練かを設定する。
    transform : object
        前処理クラスのインスタンス
    transform_anno : object
        xmlのアノテーションをリストに変換するインスタンス
    """
    def __init__(self, img_list, anno_list, phase, transform, transform_anno):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.transform_anno = transform_anno
        
    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.img_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のテンソル形式のデータとアノテーションを取得
        '''
        im, gt, h, w = self.pull_item(index)
        return im, gt
    
    def pull_item(self, index):
        '''前処理をした画像のテンソル形式のデータ、アノテーション、画像の高さ、幅を取得する'''
        # 1. 画像を取得。サイズも取得する。
        #print(self.img_list[index])
        img_path = self.img_list[index]
        img = cv2.imread(img_path)
        height, width, channel = img.shape
        
        # 2. xmlの情報を読み込み
        xml_path = self.anno_list[index]
        anno_list = self.transform_anno(xml_path, width, height)
        
        # 3. 画像とアノテーションの前処理を行う
        img, boxes, labels = self.transform(img, self.phase, anno_list[:, :4], anno_list[:, 4])
        
        # 4. transform BGR to RGB
        img = torch.from_numpy(img[:, :, (2, 1, 0)])
        # さらに（高さ、幅、色チャネル）の順を（色チャネル、高さ、幅）に変換
        img = img.permute(2, 0, 1)
        
        # BBoxとラベルをセットにしたnp.arrayを作成、変数名「gt」はground truth（答え）の略称
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        
        return img, gt, width, height
        


# In[27]:


# ## データセット作成の戦略について
# data.datasetクラスを作り、getすると画像とtargetを出力するように。
# このクラスはデータローディングや前処理などを定義する必要がある。
# 
# ↓
# 
# 次にdata.dataloaderにdata.datasetを入力し、シャッフルやミニバッチの機能を実現する。
# 
# ↓
# 
# 学習時はiter(data.dataloader)からデータを引っ張ってきて動かす。

# # make dataloader

# In[34]:


# サイズを変化させる関数
def od_collate_fn(batch):
    """
    Datasetから取り出すアノテーションデータのサイズが画像ごとに異なります。
    画像内の物体数が2個であれば(2, 5)というサイズですが、3個であれば（3, 5）など変化します。
    この変化に対応したDataLoaderを作成するために、
    カスタイマイズした、collate_fnを作成します。
    collate_fnは、PyTorchでリストからmini-batchを作成する関数です。
    ミニバッチ分の画像が並んでいるリスト変数batchに、
    ミニバッチ番号を指定する次元を先頭に1つ追加して、リストの形を変形します。
    """
    imgs = []
    targets = []
    
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    
    # imgsはミニバッチサイズのリストになっています
    # リストの要素はtorch.Size([3, 300, 300])です。
    # このリストをtorch.Size([batch_num, 3, 300, 300])のテンソルに変換します
    imgs = torch.stack(imgs, dim=0)
    
    return imgs, targets
