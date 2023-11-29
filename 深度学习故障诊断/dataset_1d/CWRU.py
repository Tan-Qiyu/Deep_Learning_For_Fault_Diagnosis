'''
Time: 2022/10/9 13:19
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/GNN_FD
'''

import torch
import numpy as np
import os
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
from 深度学习故障诊断.dataset_1d.__user_functions import Slide_window_sampling,add_noise,norm,FFT


'''Normal Baseline Data由于仅有4个文件97.mat、98.mat、99.mat、100.mat，
为了方便访问及处理，将四个文件都拷贝到其余三个文件夹中'''
# 正常数据，负载0、1、2、3
NB = ['97.mat', '98.mat', '99.mat', '100.mat']

# 内圈故障数据,负载0、1、2、3
IR07 = ['105.mat', '106.mat', '107.mat', '108.mat'] #0.7mm
IR14 = ['169.mat', '170.mat', '171.mat', '172.mat'] #0.14mm
IR21 = ['209.mat', '210.mat', '211.mat', '212.mat'] #0.21mm

# 外圈故障数据,负载0、1、2、3
OR07 = ['130.mat', '131.mat', '132.mat', '133.mat'] #0.7mm
OR14 = ['197.mat', '198.mat', '199.mat', '200.mat'] #0.14mm
OR21 = ['234.mat', '235.mat', '236.mat', '237.mat'] #0.21mm

# 滚动体故障数据,负载0、1、2、3
B07 = ['118.mat', '119.mat', '120.mat', '121.mat']  # 0.7mm
B14 = ['185.mat', '186.mat', '187.mat', '188.mat']  # 0.14mm
B21 = ['222.mat', '223.mat', '224.mat', '225.mat']  # 0.21mm

'''-----------------------------------------------------------------------------'''

'''48k Drive End Bearing Fault Data'''
# 内圈故障数据0.7mm、0.14mm、0.21mm/负载0、1、2、3
IR07_48DE = ['109.mat', '110.mat', '111.mat', '112.mat']
IR14_48DE = ['174.mat', '175.mat', '176.mat', '177.mat']
IR21_48DE = ['213.mat', '214.mat', '215.mat', '217.mat']

# 外圈故障数据0.7mm、0.14mm、0.21mm/负载0、1、2、3 in Centerd @6:00
OR07_48DE = ['135.mat', '136.mat', '137.mat', '138.mat']
OR14_48DE = ['201.mat', '202.mat', '203.mat', '204.mat']
OR21_48DE = ['238.mat', '239.mat', '240.mat', '241.mat']

# 滚动体故障数据0.7mm、0.14mm、0.21mm/负载0、1、2、3
B07_48DE = ['122.mat', '123.mat', '124.mat', '125.mat']
B14_48DE = ['189.mat', '190.mat', '191.mat', '192.mat']
B21_48DE = ['226.mat', '227.mat', '228.mat', '229.mat']

# 全部数据，包括正常加九种不同类型、不同损伤程度的故障，一共十种
full_data_48DE = [NB, IR07_48DE, IR14_48DE, IR21_48DE, OR07_48DE, OR14_48DE, OR21_48DE, B07_48DE, B14_48DE,B21_48DE]

full_data_12DE = [NB, IR07, IR14, IR21, OR07, OR14, OR21, B07, B14,B21]

def CWRU_data(args):

    if args.dir_path == '12DE':
        data_name = full_data_12DE
        file_dir = args.dataset_path + '\\12k Drive End Bearing Fault Data'
    elif args.dir_path == '48DE':
        data_name = full_data_48DE
        file_dir = args.dataset_path + '\\48k Drive End Bearing Fault Data'

    dataset = np.empty((0,args.sample_length))
    labels = np.empty((0,))
    for label,each_data_name in enumerate(data_name):
        for each_data in each_data_name:
            data = loadmat(os.path.join(file_dir,each_data))
            if int(each_data.split('.')[0]) < 100:
                vibration = data['X0'+each_data.split('.')[0]+'_DE_time']
            if int(each_data.split('.')[0]) == 174:
                vibration = data['X' + '173' + '_DE_time']
            elif int(each_data.split('.')[0]) >= 100:
                vibration = data['X' + each_data.split('.')[0] + '_DE_time']

            vibration = add_noise(vibration, noise=args.noise, snr=args.snr)  # add noise
            vibration = Slide_window_sampling(vibration, args.sample_length, args.overlap)  # 滑窗采样,shape(总划分样本数，window_size)
            np.random.shuffle(vibration)  # 打乱顺序
            vibration = vibration[:args.sample_num,]  # 取出固定数量的样本
            vibration = norm(vibration,normal_type=args.norm_type)  #归一化
            vibration = FFT(vibration,domain=args.input_type)
            # vibration = np.reshape(vibration,(vibration.shape[0],image_size,image_size))  #vibration ——> grey image

            dataset = np.concatenate((dataset,vibration),axis=0)
            labels = np.concatenate((labels,np.repeat(label,vibration.shape[0])),axis=0)

    if args.dimension == '1D':  # 1D-vibration signal as input of 1D-CNN
        dataset = np.expand_dims(dataset,axis=1)  # expand channel dimension
    elif args.dimension == '2D':  # 2D grey image as input of 2D-CNN: image size = sqrt(window_size)
        image_size = int(np.sqrt(args.sample_length))
        dataset = np.reshape(dataset,(dataset.shape[0],1,image_size,image_size))

    train_x, test_x, train_y, test_y = train_test_split(dataset, labels, train_size=args.train_size, shuffle=True,stratify=labels)  # split train set and test set


    tensor_train_x = torch.Tensor(train_x)
    tensor_train_y = torch.LongTensor(train_y)
    tensor_test_x = torch.Tensor(test_x)
    tensor_test_y = torch.LongTensor(test_y)

    loader_train = DataLoader(TensorDataset(tensor_train_x, tensor_train_y), batch_size=args.batch_size)  # training set
    loader_test = DataLoader(TensorDataset(tensor_test_x, tensor_test_y), batch_size=args.batch_size)  # testing set

    return loader_train, loader_test