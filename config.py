import time
import torch
import torch.nn as nn
import pandas as pd
import os
import torch as t
import torchvision.transforms.functional as ff
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
#from torchinfo import summary
import matplotlib.pyplot as plt
import tifffile as tiff
import random
import cv2
import albumentations as album
#from segmentation_models_pytorch import losses
#import segmentation_models_pytorch
import warnings
from utils.LabelProcessor import LabelProcessor
from utils.Dataset import *
from utils.Metrics import *
from utils.augmentation import  *
from model.MFFnet import  *
warnings.filterwarnings("ignore")
#设置字体显示中文以及默认工作路径
plt.rcParams['font.family']='SimHei'
#设置全部随机种子
seed = 707
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU的话，需要设置这个函数来保证多GPU情况下的随机性固定
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
t.manual_seed(seed)
t.cuda.manual_seed(seed)
t.cuda.manual_seed_all(seed)  # 如果使用多个GPU的话，需要设置这个函数来保证多GPU情况下的随机性固定
t.backends.cudnn.deterministic = True
t.backends.cudnn.benchmark = False
#训练、验证、测试集以及引导文件路径
TRAIN_ROOT = "./dataset/trian_img"
TRAIN_LABEL = "./dataset/trian_mask"
VAL_ROOT = "./dataset/val_img"
VAL_LABEL = "./dataset/val_mask"
TEST_ROOT = "./dataset/test_img"
TEST_LABEL = "./dataset/test_mask"
class_dict_path = "./dataset/color.csv"        #放入utf-8编码的csv索引文件
#归一化和标准化（不会统计不要动）
Mean=(0.485, 0.456, 0.406)
Std=(0.229, 0.224, 0.225)
#模型参数设置
BATCH_SIZE = 4                      #batchsize大小
crop_size = 256   #训练上随机裁剪的尺寸大小
val_size=256    #验证上的尺寸大小
class_num=2     #分类类别数
LR=0.0001       #学习率
EPOCH_NUMBER=5  #循环次数
usemodel=MyNet(3,2)  #使用的网络
#输出参数位置设置
best_pth="./output/best_pth.pth"          #以.pth结尾
last_pth="./output/last_pth.pth"          #以.pth结尾
#保存损失可视化
title="MFFnet"
loss_figure="./output/loss_figure.png"           #以.png结尾
loss_csv="./output/loss_csv.csv"                 #以.csv结尾
#测试结果
Parameter_selection="best"                #不写入best将使用最后一次训练的参数
test_result="./output/test_result.txt"    #以.txt结尾

#预测结果路径
dir='./output/predict/'                   #输出文件夹最后加“/”

