'''
Time: 2022/10/13 15:14
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/GNN_FD
'''

import torch
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
from 深度学习故障诊断.dataset_1d.__user_functions import Slide_window_sampling,add_noise,norm,FFT

def Bearing_311_data(args):
    file_dir = args.dataset_path

    file_name = ['43 正常', '保持架', '滚动体', '内圈', '外圈']

    holder_servity_name = ['40 较轻', '41 较严重']
    ball_servity_name = ['34 较轻', '37 较严重']
    inner_servity_name = ['29 较轻', '30 较严重']
    outer_servity_name = ['25 较轻', '27 较严重']

    data_name = ['c1.txt', 'c2.txt', 'c3.txt', 'c4.txt', 'c5.txt', 'c6.txt', 'c7.txt', 'c8.txt', 'c9.txt', 'c10.txt',
                 'c11.txt', 'c12.txt', ]

    data = [[], [], [], [], [], [], [], [], []]  # 创建9个空列表存放9中故障类型，包括正常和较轻、较严重下的保持架故障、滚动体故障、内圈故障、外圈故障

    for each_file_name in file_name:
        if each_file_name == '43 正常':
            for each_data_name in data_name:
                file = open(os.path.join(file_dir, each_file_name, each_data_name), encoding='gbk')
                for line in file:
                    data[0].append(float(line.strip()))  # 正常信号存在data[0]中

        elif each_file_name == '保持架':
            for num, each_holder_servity_name in enumerate(holder_servity_name):
                for each_data_name in data_name:
                    file = open(os.path.join(file_dir, each_file_name, each_holder_servity_name, each_data_name),
                                encoding='gbk')
                    for line in file:
                        data[num + 1].append(float(line.strip()))  # 保持架较轻故障信号存在data[1]中，保持架较严重故障信号存在data[2]中

        elif each_file_name == '滚动体':
            for num, each_ball_servity_name in enumerate(ball_servity_name):
                for each_data_name in data_name:
                    file = open(os.path.join(file_dir, each_file_name, each_ball_servity_name, each_data_name),
                                encoding='gbk')
                    for line in file:
                        data[num + 3].append(float(line.strip()))  # 滚动体较轻故障信号存在data[3]中，保持架较严重故障信号存在data[4]中

        elif each_file_name == '内圈':
            for num, each_inner_servity_name in enumerate(inner_servity_name):
                for each_data_name in data_name:
                    file = open(os.path.join(file_dir, each_file_name, each_inner_servity_name, each_data_name),
                                encoding='gbk')
                    for line in file:
                        data[num + 5].append(float(line.strip()))  # 内圈较轻故障信号存在data[5]中，保持架较严重故障信号存在data[6]中

        elif each_file_name == '外圈':
            for num, each_outer_servity_name in enumerate(outer_servity_name):
                for each_data_name in data_name:
                    file = open(os.path.join(file_dir, each_file_name, each_outer_servity_name, each_data_name),
                                encoding='gbk')
                    for line in file:
                        data[num + 7].append(float(line.strip()))  # 外圈较轻故障信号存在data[7]中，保持架较严重故障信号存在data[8]中

    data = np.array(data)  # data.shpe = (9 , 20480 * 12).  9 represent 9 classes faulty type, 20480 is sample frequecy each seconds, 12 is the all time length of sample

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
