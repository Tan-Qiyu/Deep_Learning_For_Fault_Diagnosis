'''
Time: 2022/10/13 15:59
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

def DC_data(args):
    root = args.dataset_path
    csv_name = ['train.csv', 'test_data.csv']

    data1 = [[], [], [], [], [], [], [], [], [], []]
    with open(os.path.join(root, csv_name[0]), encoding='gbk') as file:
        for line in file.readlines()[1:]:  # 从第二行开始读取数据
            line = line.split(',')[1:]  # 用逗号分隔数据，并舍弃第一列的编号id  6001 ---6000 data + 1label
            line = list(map(lambda x: float(x), line))  # 将6001个字符转化为数字
            data1[int(line[-1])].append(line[:-1])  # 按照标签将6000个数据存到相应的列表中

    data = [[], [], [], [], [], [], [], [], [], []]
    for data1_index in range(len(data1)):
        data[data1_index].append(data1[data1_index][:43])

    data = np.array(data).squeeze(axis=1)  # shape: (10,43,6000)
    data = data[:, :, :(data.shape[2] // args.sample_length) * args.sample_length]  # shape: (10,43,5120)  when window_size == 1024

    data = data.reshape((data.shape[0], data.shape[1], data.shape[2] // args.sample_length,args.sample_length))  # 将5120个数据按照window_size划分  shape: (10,43,5,1024)
    data = data.reshape((data.shape[0], data.shape[1] * data.shape[2], data.shape[3]))  # shape: (10,215,1024)
    data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))  # shape: (10,215*1024)

    # 添加噪声
    noise_data = np.zeros((data.shape[0], data.shape[1]))
    for data_i in range(data.shape[0]):
        noise_data[data_i] = add_noise(data[data_i], args.noise, args.snr)

    # 滑窗采样
    sample_data = np.zeros((noise_data.shape[0], noise_data.shape[1] // args.sample_length, args.sample_length))
    for noise_data_i in range(noise_data.shape[0]):
        sample_data[noise_data_i] = Slide_window_sampling(noise_data[noise_data_i], window_size=args.sample_length,
                                                          overlap=args.overlap)

    sample_data = sample_data[:, :args.sample_num, :]

    # 归一化
    norm_data = np.zeros((sample_data.shape[0], sample_data.shape[1], sample_data.shape[2]))
    for sample_data_i in range(sample_data.shape[0]):
        norm_data[sample_data_i] = norm(sample_data[sample_data_i], args.norm_type)

    # 傅里叶变换
    fft_data = np.zeros((norm_data.shape[0], norm_data.shape[1], norm_data.shape[2]))
    for norm_data_i in range(norm_data.shape[0]):
        fft_data[norm_data_i] = FFT(norm_data[norm_data_i], args.input_type)

    dataset = np.empty((0, fft_data.shape[2]))
    labels = np.empty((0,))
    for label, each_class_data in enumerate(fft_data):
        dataset = np.concatenate((dataset, each_class_data), axis=0)
        labels = np.concatenate((labels, np.repeat(label, each_class_data.shape[0])), axis=0)

    if args.dimension == '1D':  # 1D-vibration signal as input of 1D-CNN
        dataset = np.expand_dims(dataset, axis=1)  # expand channel dimension
    elif args.dimension == '2D':  # 2D grey image as input of 2D-CNN: image size = sqrt(window_size)
        image_size = int(np.sqrt(args.sample_length))
        dataset = np.reshape(dataset, (dataset.shape[0], 1, image_size, image_size))

    train_x, test_x, train_y, test_y = train_test_split(dataset, labels, train_size=args.train_size, shuffle=True,
                                                        stratify=labels)  # split train set and test set

    tensor_train_x = torch.Tensor(train_x)
    tensor_train_y = torch.LongTensor(train_y)
    tensor_test_x = torch.Tensor(test_x)
    tensor_test_y = torch.LongTensor(test_y)

    loader_train = DataLoader(TensorDataset(tensor_train_x, tensor_train_y), batch_size=args.batch_size)  # training set
    loader_test = DataLoader(TensorDataset(tensor_test_x, tensor_test_y), batch_size=args.batch_size)  # testing set

    return loader_train, loader_test

