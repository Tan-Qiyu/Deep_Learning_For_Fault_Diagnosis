'''
Time: 2022/10/13 15:19
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/GNN_FD
'''

import torch
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
from 深度学习故障诊断.dataset_1d.__user_functions import Slide_window_sampling,add_noise,norm,FFT

def QianPeng_data(args):
    dir = ['正常运行下', '点蚀', '点蚀磨损', '断齿', '断齿、磨损混合故障', '磨损']
    if args.QianPeng_rpm == 880:
        txt_name = [['normal880.txt', 'normal880-1.txt', 'normal880-2.txt', 'normal880-3.txt'],
                    ['dianshi880.txt', 'dianshi880-1.txt', 'dianshi880-2.txt', 'dianshi880-3.txt'],
                    ['dianmo880.txt', 'dianmo880-1.txt', 'dianmo880-2.txt', 'dianmo880-3.txt'],
                    ['duanchi880.txt', 'duanchi880-1.txt', 'duanchi880-2.txt', 'duanchi880-3.txt'],
                    ['duanmo880.txt', 'duanmo880-1.txt', 'duanmo880-2.txt', 'duanmo880-3.txt'],
                    ['mosun880.txt', 'mosun880-1.txt', 'mosun880-2.txt', 'mosun880-3.txt']]
    elif args.QianPeng_rpm == 1470:
        txt_name = [['normal1500.txt'],
                    ['dianshi1470.txt', 'dianshi1500.txt'],
                    ['dianmo1470.txt'],
                    ['duanchi1500.txt'],
                    ['duanmo1470.txt'],
                    ['mosun1470.txt']]

    data = []
    for dir_num, each_dir in enumerate(dir):
        sub_txt = txt_name[dir_num]
        subdata = []
        for each_txt in sub_txt:
            file_name = os.path.join(args.dataset_path, each_dir, each_txt)
            with open(file_name, encoding='gbk') as f:
                for line in f.readlines()[1:]:  # 从第二行开始读取，第一行为空格
                    line = line.strip('\t\n')  # 删除每一行的最后一个制表符和换行符号
                    line = line.split('\t')  # 按制表符\t进行分隔
                    subdata.append(list(map(lambda x: float(x), line)))  # 将字符数据转化为数字

        subdata = np.array(subdata)  # shape:(样本数，通道数)
        data.append(subdata)

    # 当每一类故障的样本不平衡时截取相同数量的样本
    sam_list = []
    list(map(lambda x: sam_list.append(x.shape[0]), data))
    sam_min = min(sam_list)
    max_min = max(sam_list)
    if sam_min != max_min:
        balance_data = [[], [], [], [], [], []]
        for all_data_index, class_data in enumerate(data):
            # np.random.shuffle(class_data)
            balance_data[all_data_index].append(data[all_data_index][:sam_min, :])
        data = np.array(balance_data).squeeze(axis=1)

    data = np.array(data)  # shape:(故障类型数，样本数，通道数)
    data = data[:, :, args.QianPeng_channel]  # 选择通道

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

