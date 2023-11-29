'''
Time: 2022/10/13 12:29
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/GNN_FD
'''

import torch
import numpy as np
import os
from itertools import islice
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
from 深度学习故障诊断.dataset_1d.__user_functions import Slide_window_sampling,add_noise,norm,FFT

def SEU_data(args):

    root = args.dataset_path
    if "bearingset" in root:
        # Data names of 5 bearing fault types under two working conditions
        data_name = ["ball_20_0.csv", "comb_20_0.csv", "health_20_0.csv", "inner_20_0.csv", "outer_20_0.csv",
                     "ball_30_2.csv", "comb_30_2.csv", "health_30_2.csv", "inner_30_2.csv", "outer_30_2.csv"]

    elif "gearset" in root:  # Path of gearset
        # Data names of 5 gear fault types under two working conditions
        data_name = ["Chipped_20_0.csv", "Health_20_0.csv", "Miss_20_0.csv", "Root_20_0.csv", "Surface_20_0.csv",
                     "Chipped_30_2.csv", "Health_30_2.csv", "Miss_30_2.csv", "Root_30_2.csv", "Surface_30_2.csv"]

    data_list = [[], [], [], [], [], [], [], [], [], []]  # 创建10个空列表存放十种故障数据

    for num_state, dir_name in enumerate(data_name):

        dir = os.path.join(root, dir_name)
        b_g_data = open(dir, "r", encoding='gb18030', errors='ignore')

        i = 0

        if dir_name == 'ball_20_0.csv':
            for line in islice(b_g_data, 16, None):  # 逐行取出，移除前16行说明
                if i < args.sample_num * args.sample_length:  # 限制信号长度，因为信号总长为1048560，8传感器，10类型，即1048560*8*10接近上亿的数值会对计算机造成一定的影响
                    line = line.rstrip()  # 移除每一行末尾的字符
                    word = line.split(",", 8)  # 使用8个,进行分隔
                    word = list(map(float, word[:-1]))  # 将每行的str数据转化为float数值型，并移除最后一个空元素
                    data_list[num_state].append(word)  # 将故障信号存在相应的列表内
                    i += 1
                else:
                    break

        else:
            for line in islice(b_g_data, 16, None):
                if i < args.sample_num * args.sample_length:
                    line = line.rstrip()
                    word = line.split("\t", 8)
                    word = list(map(float, word))
                    data_list[num_state].append(word)
                    i += 1
                else:

                    break

    # 振动信号
    all_data = np.array(data_list)  # the dimention of bearing is: classes * sample_length * channels

    data = all_data[:, :, args.SEU_channel]

    # 添加噪声
    noise_data = np.zeros((data.shape[0], data.shape[1]))
    for data_i in range(data.shape[0]):
        noise_data[data_i] = add_noise(data[data_i], args.noise, args.snr)

    # 滑窗采样
    sample_data = np.zeros((noise_data.shape[0], noise_data.shape[1] // args.sample_length, args.sample_length))
    for noise_data_i in range(noise_data.shape[0]):
        sample_data[noise_data_i] = Slide_window_sampling(noise_data[noise_data_i], window_size=args.sample_length,overlap=args.overlap)

    sample_data = sample_data[:, :args.sample_num, :]

    # 归一化
    norm_data = np.zeros((sample_data.shape[0], sample_data.shape[1], sample_data.shape[2]))
    for sample_data_i in range(sample_data.shape[0]):
        norm_data[sample_data_i] = norm(sample_data[sample_data_i], args.norm_type)

    # 傅里叶变换
    fft_data = np.zeros((norm_data.shape[0], norm_data.shape[1], norm_data.shape[2]))
    for norm_data_i in range(norm_data.shape[0]):
        fft_data[norm_data_i] = FFT(norm_data[norm_data_i], args.input_type)

    dataset = np.empty((0,fft_data.shape[2]))
    labels = np.empty((0,))
    for label, each_class_data in enumerate(fft_data):
        dataset = np.concatenate((dataset, each_class_data), axis=0)
        labels = np.concatenate((labels, np.repeat(label, each_class_data.shape[0])), axis=0)

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

