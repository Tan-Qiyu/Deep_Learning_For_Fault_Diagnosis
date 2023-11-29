'''
Time: 2022/10/8 23:26
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/GNN_FD
'''

import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):  # @save
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

b1 = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

def resnet_block(input_channels, num_channels, num_residuals,first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

class ResNet18(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(ResNet18, self).__init__()
        self.conv1   = b1
        self.conv2_x = b2
        self.conv3_x = b3
        self.conv4_x = b4
        self.conv5_x = b5
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Sequential(
            nn.Linear(512 * 1 * 1, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256,out_channel)
        )

    def forward(self,x):  # the shape of x -> (batch,1,image_size,image_size)

        self.f1 = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])  # 原始信号特征

        x = self.conv1(x)

        self.f2 = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])  # conv1特征

        x = self.conv2_x(x)

        self.f3 = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])  # conv2特征

        x = self.conv3_x(x)

        self.f4 = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])  # conv3特征

        x = self.conv4_x(x)

        self.f5 = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])  # conv4特征

        x = self.conv5_x(x)

        self.f6 = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])  # conv5特征

        x = self.avgpool(x)
        x = x.view(x.size(0),-1)   #保持batch_size数，后面维度展平
        x = self.linear(x)

        self.f7 = x  # FC特征

        out = F.log_softmax(x,dim=1)

        return out

    def get_fea(self):
        return [self.f1,self.f2,self.f3,self.f4,self.f5,self.f6,self.f7]