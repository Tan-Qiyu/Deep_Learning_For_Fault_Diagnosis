'''
Time: 2022/10/13 15:29
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/GNN_FD
'''

import torch
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
from 深度学习故障诊断.dataset_1d.__user_functions import Slide_window_sampling,add_noise,norm,FFT

def JNU_data(args):

    root = args.dataset_path

    health = ['n600_3_2.csv', 'n800_3_2.csv', 'n1000_3_2.csv']  # 600 800 1000转速下的正常信号
    inner = ['ib600_2.csv', 'ib800_2.csv', 'ib1000_2.csv']  # 600 800 1000转速下的内圈故障信号
    outer = ['ob600_2.csv', 'ob800_2.csv', 'ob1000_2.csv']  # 600 800 1000转速下的外圈故障信号
    ball = ['tb600_2.csv', 'tb800_2.csv', 'tb1000_2.csv']  # 600 800 1000转速下的滚动体故障信号

    file_name = []  # 存放三种转速下、四种故障状态的文件名，一共12种类型
    file_name.extend(health)
    file_name.extend(inner)
    file_name.extend(outer)
    file_name.extend(ball)

    data1 = [[], [], [], [], [], [], [], [], [], [], [], []]  # 创建一个长度为12的空列表存放12种故障数据(每一类数据不平衡)
    for num, each_name in enumerate(file_name):
        dir = os.path.join(root, each_name)
        with open(dir, "r", encoding='gb18030', errors='ignore') as f:
            for line in f:
                line = float(line.strip('\n'))  # 删除每一行后的换行符号，并将字符型转化为数字
                data1[num].append(line)  # 将取出来的数据逐个存放到相应的列表中

    data = [[], [], [], [], [], [], [], [], [], [], [], []]  # 创建一个长度为12的空列表存放12种故障数据（每一类数据平衡）shape：(12,500500)
    for data1_i in range(len(data1)):
        data[data1_i].append(data1[data1_i][:500500])  # 将所有类型数据总长度截取为500500

    data = np.array(data).squeeze(axis=1)  # shape：(12,500500)

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
