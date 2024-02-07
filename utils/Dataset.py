from utils.LabelProcessor import LabelProcessor
#导入相关库
import time
import torch
import torch.nn as nn
import pandas as pd
import config
import os
import torch as t
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from torchvision import models
from torch import nn
import six
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.models as models
from datetime import datetime
import matplotlib.pyplot as plt
from config import *
import tifffile as tiff
import random
import cv2
import albumentations as album
import warnings
warnings.filterwarnings("ignore")
#设置字体显示中文以及默认工作路径
plt.rcParams['font.family']='SimHei'

# 基于Image库打开
class MyDataset(Dataset):
    def __init__(self, file_path=[], augmentation=None):
        """para:
            file_path(list): 数据和标签路径,列表元素第一个为图片路径，第二个为标签路径
        """
        # 1 正确读入图片和标签路径
        if len(file_path) != 2:
            raise ValueError("同时需要图片和标签文件夹的路径，图片路径在前")
        self.img_path = file_path[0]
        self.label_path = file_path[1]
        # 2 从路径中取出图片和标签数据的文件名保持到两个列表当中（程序中的数据来源）
        self.imgs = self.read_file(self.img_path)
        self.labels = self.read_file(self.label_path)
        # 3 初始化数据处理函数设置
        self.augmentation = augmentation

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        # 图像读取，可以使用其他库但是要确保读取后为array形式
        img = tiff.imread(img)
        # 标签读取，读取后一定要保证有RGB对应字典的三个维度，确保可以让mask进行one-hot编码
        label = Image.open(label).convert('RGB')
        label = np.array(label)  # 以免不是np格式的数据，并且以整型显示
        label = Image.fromarray(label.astype('uint8'))
        label = label_processor.encode_label_img(label)
        """
        数据增强augmentation空间变化
        """
        if self.augmentation:
            img_label = self.augmentation(image=img, mask=label)
            img = img_label["image"]
            label = img_label["mask"]

        label = np.array(label, dtype="int16")
        # img=np.array(img,dtype="float32")
        img = np.clip(img, 0, 255).astype(np.uint8)  # 压缩img的像素值在0-255范围内

        imgs, labels = self.img_transform(img, label)

        sample = {'img': imgs, 'label': labels}
        return sample

    def __len__(self):
        return len(self.imgs)

    def read_file(self, path):
        """从文件夹中读取数据"""
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        #file_path_list.sort()   #纯数字会不按顺序加载1.2.3...——1.10.11....
        #file_path_list = sorted(file_path_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))  # 解决纯数字问题
        return file_path_list

    def img_transform(self, img, label):
        """
        像素转为tensor

        """
        # label = np.array(label)  # 以免不是np格式的数据
        img = np.array(img)
        label = Image.fromarray(label.astype('uint8'))
        label = np.array(label)
        transform_img = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=config.Mean, std=config.Std, )  # 标准化
                # 可加入归一化或标准化
            ]
        )
        img = transform_img(img)
        label = t.from_numpy(label)
        return img, label
label_processor = LabelProcessor(config.class_dict_path)