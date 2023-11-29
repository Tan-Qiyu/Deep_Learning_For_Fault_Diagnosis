'''
Time: 2022/10/8 23:26
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/GNN_FD
'''

import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.linear = nn.Sequential(
            nn.Linear(256 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_channel)
        )

    def forward(self, x):  # the shape of x -> (batch,3,image_size,image_size)

        self.f1 = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])  # 原始信号特征

        x = self.conv1(x)

        self.f2 = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])  # conv1特征

        x = self.conv2(x)

        self.f3 = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])  # conv2特征

        x = self.conv3(x)

        self.f4 = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])  # conv3特征

        x = x.view(x.size(0), -1)  # 保持batch_size数，后面维度展平
        x = self.linear(x)

        self.f2 = x  # FC特征

        out = F.log_softmax(x, dim=1)

        return out

    def get_fea(self):
        return [self.f1,self.f2,self.f3,self.f4]
