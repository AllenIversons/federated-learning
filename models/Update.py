#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from loguru import logger
import numpy as np
import random
from sklearn import metrics

# 用于将全局数据集划分为多个本地数据集，并将每个本地数据集分配给不同的客户端
"""
    在这里创建了一个DatasetSplit类，其继承自Dataset类
    DatasetSplit 类接受两个参数，分别是原始数据集 dataset 和一个样本索引列表 idxs。
    在初始化时，它会将 idxs 转换为 Python 中的列表类型，并保存到对象属性 self.idxs 中。
    在获取数据时，它会根据索引列表 self.idxs 从原始数据集中选择对应的数据项，然后返回图像和标签作为模型输入。
"""
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None,clientId = None):
        self.args = args
        self.clientId = clientId
        # 定义了一个交叉熵损失函数
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        # 通过使用随机梯度下降sgd算法进行模型优化
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):

            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # 以下为模型训练的关键步骤
                # step1：将模型的参数梯度初始化为0
                net.zero_grad()
                # step2：通过MLP.forward(net,images) 前向传播计算 预测值（sever共享的模型）
                log_probs = net(images)
                # step3：计算交叉熵损失
                loss = self.loss_func(log_probs, labels)
                # step4：通过反响传播计算梯度
                loss.backward()
                # step5：更新所有参数
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    logger.info('Client #{} Update local model: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        self.clientId,iter+1, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                    # print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    #     iter, batch_idx * len(images), len(self.ldr_train.dataset),
                    #            100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        # 返回当前client基于自己本地数据训练所得到的新的模型参数
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

