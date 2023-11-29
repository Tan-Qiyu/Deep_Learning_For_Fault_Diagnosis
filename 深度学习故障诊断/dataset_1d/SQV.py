'''
Time: 2022/10/13 16:07
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/GNN_FD
'''

import torch
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
from 深度学习故障诊断.dataset_1d.__user_functions import Slide_window_sampling,add_noise,norm,FFT

def SQV_data(args):
    root = args.dataset_path

    dir = ['NC', 'IF_1', 'IF_2', 'IF_3', 'OF_1', 'OF_2', 'OF_3']  # 不同损伤程度的故障类型名
    txt_name = [['REC3642_ch2.txt', 'REC3643_ch2.txt', 'REC3644_ch2.txt', 'REC3645_ch2.txt', 'REC3646_ch2.txt',
                 'REC3647_ch2.txt', 'REC3648_ch2.txt', 'REC3649_ch2.txt', 'REC3650_ch2.txt'],
                ['REC3597_ch2.txt', 'REC3598_ch2.txt', 'REC3599_ch2.txt', 'REC3600_ch2.txt', 'REC3601_ch2.txt',
                 'REC3602_ch2.txt', 'REC3603_ch2.txt', 'REC3604_ch2.txt', 'REC3605_ch2.txt', 'REC3606_ch2.txt'],
                ['REC3619_ch2.txt', 'REC3620_ch2.txt', 'REC3621_ch2.txt', 'REC3623_ch2.txt', 'REC3624_ch2.txt',
                 'REC3625_ch2.txt', 'REC3626_ch2.txt', 'REC3627_ch2.txt', 'REC3628_ch2.txt'],
                ['REC3532_ch2.txt', 'REC3533_ch2.txt', 'REC3534_ch2.txt', 'REC3535_ch2.txt', 'REC3536_ch2.txt',
                 'REC3537_ch2.txt'],
                ['REC3513_ch2.txt', 'REC3514_ch2.txt', 'REC3515_ch2.txt', 'REC3516_ch2.txt', 'REC3517_ch2.txt',
                 'REC3518_ch2.txt'],
                ['REC3494_ch2.txt', 'REC3495_ch2.txt', 'REC3496_ch2.txt', 'REC3497_ch2.txt', 'REC3498_ch2.txt',
                 'REC3499_ch2.txt'],
                ['REC3476_ch2.txt', 'REC3477_ch2.txt', 'REC3478_ch2.txt', 'REC3479_ch2.txt', 'REC3480_ch2.txt',
                 'REC3481_ch2.txt']]

    txt_index = [0, 0, 0, 0, 0, 0, 0]  # 元素对应每一个txt文件位置
    data1 = [[], [], [], [], [], [], []]
    for num, each_dir in enumerate(dir):
        with open(os.path.join(root, each_dir, txt_name[num][txt_index[num]])) as file:
            for line in file.readlines()[16:]:  # 前16行说明不读取
                line = line.strip('\n')  # 删除末尾的换行
                line = line.split('\t')
                line = list(map(lambda x: float(x), line))
                data1[num].append(line)

    min_value = min(list(map(lambda x: len(x), data1)))

    data = [[], [], [], [], [], [], []]
    for data1_index in range(len(data1)):
        data[data1_index] = data1[data1_index][:min_value]

    data = np.array(data)  # shape : (7,min_value,2)  --- egg:min_value = 460800 ; 第三个维度2 表示 时间+振动信号幅值
    data = data[:, :, 1]  # 振动信号 --- shape : (7,min_value)

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

