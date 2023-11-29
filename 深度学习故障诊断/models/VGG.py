'''
Time: 2022/10/8 23:26
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/GNN_FD
'''

import torch.nn as nn
import torch.nn.functional as F

class VGG(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(VGG, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.maxpool = nn.AdaptiveMaxPool2d((1,1))
        self.linear = nn.Sequential(
            nn.Linear(512 * 1 * 1, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_channel)
        )

    def forward(self,x):   # the shape of x -> (batch,1,image_size,image_size)

        self.f1 = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])  # 原始信号特征

        x = self.block1(x)

        self.f2 = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])  # conv1特征

        x = self.block2(x)

        self.f3 = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])  # conv2特征

        x = self.block3(x)

        self.f4 = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])  # conv3特征

        x = self.block4(x)

        self.f5 = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])  # conv4特征

        x = self.block5(x)

        self.f6 = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])  # conv5特征

        x = self.maxpool(x)
        x = x.view(x.size(0),-1)   #保持batch_size数，后面维度展平
        x = self.linear(x)

        self.f7 = x  # FC特征

        out = F.log_softmax(x,dim=1)

        return out

    def get_fea(self):
        return [self.f1,self.f2,self.f3,self.f4,self.f5,self.f6,self.f7]