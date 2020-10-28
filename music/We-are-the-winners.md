import torch
import numpy as np
import os
import random

import json
import os

if os.path.exists('/home/kesci/data/competition/')==True:
    p1 = json.load(open('/home/kesci/data/competition/train_set/Part1.json'))
    p2 = json.load(open('/home/kesci/data/competition/train_set/Part2.json'))
    p3 = json.load(open('/home/kesci/data/competition/train_set/Part3.json'))
    p4 = json.load(open('/home/kesci/data/competition/train_set/Part4.json'))
    test_list = os.listdir('/home/kesci/data/competition/test_set')
if os.path.exists('/home/kesci/data1/competition/')==True:
    p1 = json.load(open('/home/kesci/data1/competition/train_set/Part1.json'))
    p2 = json.load(open('/home/kesci/data1/competition/train_set/Part2.json'))
    p3 = json.load(open('/home/kesci/data1/competition/train_set/Part3.json'))
    p4 = json.load(open('/home/kesci/data1/competition/train_set/Part4.json'))
    test_list = os.listdir('/home/kesci/data1/competition/test_set')

dict_all = [p1, p2, p3, p4]
count_all = {'sunny':0,'cloudy':0, 'others':0}
length_all = 0
print('-'*65)
for dict in dict_all:
    count = {'sunny':0,'cloudy':0, 'others':0}
    length = len(dict)
    length_all += length
    for item in dict.items():
        count[item[1]] += 1
        count_all[item[1]] += 1
    print(length,count)
print('total:',length_all, count_all)
print('-'*65)
print('test:',len(test_list))
print('-'*65)

import matplotlib.pyplot as plt
from PIL import Image
import random

random.seed(0)
for index,dict in enumerate(dict_all):
    if os.path.exists('/home/kesci/data/competition/')==True:
        folder_path = os.path.join('/home/kesci/data/competition/train_set','Part'+str(index+1))
    if os.path.exists('/home/kesci/data1/competition/')==True:
        folder_path = os.path.join('/home/kesci/data1/competition/train_set','Part'+str(index+1))
    print(folder_path)
    name_list = list(dict.keys())
    random.shuffle(name_list)
    plt.figure(figsize=(20,20)) #设置窗口大小
    plt.suptitle('P' + str(index+1)) # 图片集名称
    for file_index,file_name in enumerate(name_list):
        file_path = os.path.join(folder_path, file_name)
        img = Image.open(file_path)
        label = dict[file_name]
        plt.subplot(4,4,file_index+1), plt.title(str(label))
        plt.imshow(img)
        if file_index == 15:
            break
    plt.show()
	
train_path_list = []
train_label_list = []
test_path_list = []

for index, dict in enumerate(dict_all):
    if os.path.exists('/home/kesci/data/competition/')==True:
        train_folder_path = os.path.join('/home/kesci/data/competition/train_set','Part'+str(index+1))
    if os.path.exists('/home/kesci/data1/competition/')==True:
        train_folder_path = os.path.join('/home/kesci/data1/competition/train_set','Part'+str(index+1))
    for train_name, train_label in dict.items():
        file_path = os.path.join(train_folder_path, train_name)
        train_path_list.append(file_path)
        train_label_list.append(train_label)

if os.path.exists('/home/kesci/data/competition/')==True:
    test_folder_path = '/home/kesci/data/competition/test_set'
if os.path.exists('/home/kesci/data1/competition/')==True:
    test_folder_path = '/home/kesci/data1/competition/test_set'
for file_name in test_list:
    test_path = os.path.join(test_folder_path, file_name)
    test_path_list.append(test_path)
print(len(test_path_list))

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torchtoolbox.transform import Cutout

train_transform = transforms.Compose([
    transforms.Resize((300,300)),
    Cutout(),
    transforms.ToTensor(),
    transforms.Normalize([0.471, 0.448, 0.408], [0.234, 0.239, 0.242])
])

val_transform = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor(),
    transforms.Normalize([0.471, 0.448, 0.408], [0.234, 0.239, 0.242])
])

# 打乱图像顺序
random.seed(0)
random.shuffle(train_path_list)
random.seed(0)
random.shuffle(train_label_list)
cut = int(len(train_label_list)*0.13)
# cut = 5000
matches = ['sunny','cloudy','others']

class WTDataset(Dataset):
    def __init__(self, train_transform, train=True):
        train_img = train_path_list[cut:]
        train_label = train_label_list[cut:]
        val_img = train_path_list[:cut]
        val_label = train_label_list[:cut]

        if train:
            self.img = train_img
            self.label = train_label
        else:
            self.img = val_img
            self.label = val_label

        self.train_transform = train_transform
    def __getitem__(self, index):
        img = Image.open(self.img[index])
        img = img.convert("RGB")
        label = self.label[index]
        label = matches.index(label)
        return self.train_transform(img), label

    def __len__(self):
        return len(self.img)

train_loader = torch.utils.data.DataLoader(
    WTDataset(train_transform, train=True),
    batch_size=32, shuffle=True, num_workers=16, pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    WTDataset(val_transform, train=False),
    batch_size=32, shuffle=False, num_workers=16, pin_memory=True
)

import torch.nn as nn
from tqdm import tqdm
from torchvision import models
from torchsummary import summary
from efficientnet_pytorch import EfficientNet
num_classes = 3
lr = 0.001
epochs = 12

model = EfficientNet.from_name('efficientnet-b3')
model.load_state_dict(torch.load('/home/kesci/work/adv-efficientnet-b3-cdd7c0f4.pth'))
fc_features = model._fc.in_features
model._fc = nn.Linear(fc_features, num_classes)

from torchtoolbox.optimizer import Lookahead
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

criterion = nn.CrossEntropyLoss()  # 损失函数

# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
optimizer = Lookahead(optimizer)
swa_model = AveragedModel(model)
scheduler = CosineAnnealingLR(optimizer, T_max=3)
swa_start = 0
swa_scheduler = SWALR(optimizer, swa_lr=0.0001)

from sklearn.metrics import f1_score
scaler = torch.cuda.amp.GradScaler()


def train():
    train_loss, train_f1_score = 0.0, 0.0
    model.cuda()
    model.train()
    for input, target in tqdm(train_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.cuda.amp.autocast():  # 混合精度加速训练
            output = model(input)
            loss = criterion(output, target)
        optimizer.zero_grad()  # 重置梯度，不加会爆显存
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #  计算f1_score
        pre = output.argmax(axis=1).cpu()
        label = target.cpu()
        f1score = f1_score(label, pre, average='weighted')
        train_f1_score += f1score
        train_loss += loss
    if epoch > swa_start:
        swa_model.update_parameters(model)
        swa_scheduler.step()
    else:
        scheduler.step()
    n = len(train_loader)
    return train_loss /n ,train_f1_score /n


def val():
    with torch.no_grad():  # 作用等于‘optimizer.zero_grad()’
        val_loss, val_f1_score = 0.0, 0.0
        model.eval()
        for input, target in tqdm(val_loader):
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)
            pre = output.argmax(axis=1).cpu()
            label = target.cpu()
            f1score = f1_score(label, pre, average='weighted')
            val_f1_score += f1score
            val_loss += loss
        n = len(val_loader)
        return val_loss/n, val_f1_score/n
		

for epoch in range(epochs):
    # model.load_state_dict(torch.load('/home/kesci/work/830/0.830.pth'))  # 接着上次训练
    train_loss, train_f1 = train()
    val_loss, val_f1 = val()
    torch.save(model.state_dict(), '/home/kesci/work/check_pointb3_fuxian/%0.3f.pth'%val_f1)
    print('Epoch %d: train_loss %.4f, train_F1_score %.3f, val_loss %.4f, val_F1_score %.3f'
          % (epoch, train_loss, train_f1, val_loss, val_f1))
		  
import torchvision.transforms as transforms
import torchvision.models as models
import os
from torch.utils.data.dataset import Dataset
import torch

if os.path.exists('/home/kesci/data/competition/')==True:
    test_dir = '/home/kesci/data/competition/test_set'
if os.path.exists('/home/kesci/data1/competition/')==True:
    test_dir = '/home/kesci/data1/competition/test_set'
test_list = os.listdir(test_dir)

print(len(test_list))

class TestDataset(Dataset):
    def __init__(self, transform):
        self.filename = test_list
        self.transform = transform
    
    def __getitem__(self, index):
        image_path = os.path.join(test_dir, self.filename[index])
        img = Image.open(image_path)
        img = img.convert("RGB")
        img_index = self.filename[index]
        return self.transform(img), img_index
    
    def __len__(self):
        return len(self.filename)

test_loader = torch.utils.data.DataLoader(
    TestDataset(val_transform),
    batch_size=256, shuffle=False, num_workers=16, pin_memory=True
)

def predict():
    model = EfficientNet.from_name("efficientnet-b3", num_classes=3).cuda()
    model.load_state_dict(torch.load('/home/kesci/work/fuxian_830/0.830.pth'))
    model.cuda()
    model.eval()
    
    with torch.no_grad():
        for input, img_index in tqdm(test_loader):
            input = input.cuda()
            outputs = model(input)
            outputs = outputs.argmax(axis=1).cpu()
            for name, value in zip(img_index,outputs):
                writer.writerow([str(name),int(value)])

import pandas as pd
import csv

while(1):
    headers = ['id','weather']
    f = open('/home/kesci/work/fuxian_830/fuxian.csv','w', newline='')
    writer = csv.writer(f)
    writer.writerow(headers)
    predict()
    df = pd.read_csv('/home/kesci/work/fuxian_830/fuxian.csv')
    if(df.shape[0]==72778):
        print('预测完成，请进行提交操作！')
        break

