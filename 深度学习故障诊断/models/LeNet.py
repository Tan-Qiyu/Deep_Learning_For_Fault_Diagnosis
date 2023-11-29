'''
Time: 2022/10/8 23:26
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/GNN_FD
'''
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):

    def __init__(self,in_channel,out_channel):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveMaxPool2d((5,5))
        )

        self.linear = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84,out_channel),
        )

    def forward(self, x):  # the shape of x -> (batch,1,image_size,image_size)

        self.f1 = x.reshape(x.shape[0],x.shape[1] * x.shape[2] * x.shape[3]) # 原始信号特征

        x = self.conv1(x)

        self.f2 = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])  # conv1特征

        x = self.conv2(x)

        self.f3 = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])  # conv3特征

        x = x.view(x.size(0),-1)
        x = self.linear(x)

        self.f4 = x  # FC特征

        out = F.log_softmax(x, dim=1)

        return out

    def get_fea(self):
        return [self.f1,self.f2,self.f3,self.f4]