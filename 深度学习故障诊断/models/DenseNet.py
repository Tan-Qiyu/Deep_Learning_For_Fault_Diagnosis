'''
Time: 2022/10/8 23:26
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/GNN_FD
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))

class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(num_channels * i + input_channels, num_channels))
            self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输⼊和输出
            X = torch.cat((X, Y), dim=1)
        return X

def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))



b1 = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# num_channels为当前的通道数
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # 上⼀个稠密块的输出通道数
    num_channels += num_convs * growth_rate
    # 在稠密块之间添加⼀个转换层，使通道数量减半
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

class DenseNet(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(DenseNet, self).__init__()
        self.conv1 = b1
        self.blocks = nn.Sequential(*blks)
        self.BN = nn.ReLU(nn.BatchNorm2d(num_channels))
        self.maxpool = nn.AdaptiveMaxPool2d((1,1))
        self.linear = nn.Sequential(
            nn.Linear(num_channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256,out_channel)
        )

    def forward(self,x):  # the shape of x -> (batch,1,image_size,image_size)

        self.f1 = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])  # 原始信号特征

        x = self.conv1(x)

        self.f2 = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])  # conv1特征

        x = self.blocks(x)
        x = self.BN(x)
        x = self.maxpool(x)

        self.f3 = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])  # conv2特征

        x = x.view(x.size(0),-1)   #保持batch_size数，后面维度展平
        x = self.linear(x)

        self.f4 = x  # FC特征

        out = F.log_softmax(x,dim=1)

        return out

    def get_fea(self):
        return [self.f1,self.f2,self.f3,self.f4]
