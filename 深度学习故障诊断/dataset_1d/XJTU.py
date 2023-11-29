'''
Time: 2022/10/13 15:02
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/GNN_FD
'''

import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
from 深度学习故障诊断.dataset_1d.__user_functions import Slide_window_sampling,add_noise,norm,FFT

def XJTU_data(args):

    WC = os.listdir(args.dataset_path)  # 遍历根目录，目录下包括三种工况的文件夹名

    datasetname1 = os.listdir(os.path.join(args.dataset_path, WC[0]))
    datasetname2 = os.listdir(os.path.join(args.dataset_path, WC[1]))
    datasetname3 = os.listdir(os.path.join(args.dataset_path, WC[2]))

    data_name = []
    data_name.extend(datasetname1)
    data_name.extend(datasetname2)
    data_name.extend(datasetname3)

    # 工况1数据及标签
    data = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    for i in tqdm(range(len(data_name))):
        if i >= 0 and i <= 4:
            dir = os.path.join('/tmp', args.dataset_path, WC[0], data_name[i])  # 工况1：35Hz12kN
            files = os.listdir(dir)
        elif i > 4 and i <= 9:
            dir = os.path.join('/tmp', args.dataset_path, WC[1], data_name[i])  # 工况2：37.5Hz11kN
            files = os.listdir(dir)
        elif i > 9 and i <= 14:
            dir = os.path.join('/tmp', args.dataset_path, WC[2], data_name[i])  # 工况3：40Hz10kN
            files = os.listdir(dir)

        # 提取振动信号最后故障时刻的采样值时，csv文件的读取顺序不是按照数字排序进行读取的，例如123在99之前，故根据.csv文件的前缀名进行读取
        files_list = list(map(lambda x: int(x[:-4]), files))
        load_file_name = list(range(np.array(files_list).max() - args.minute_value + 1,
                                    np.array(files_list).max() + 1))  # 取出最后minute_value分钟内需要处理的故障数据
        load_file_name = list(map(lambda y: str(y) + '.csv', load_file_name))

        data11 = np.empty((0,))
        for ii in range(args.minute_value):  # Take the data of the last three CSV files
            path1 = os.path.join(dir, load_file_name[ii])
            fl = pd.read_csv(path1)
            if args.XJTU_channel == 'X':  # 水平轴信号
                fl = fl["Horizontal_vibration_signals"]
            elif args.XJTU_channel == 'Y':  # 垂直轴信号
                fl = fl["Vertical_vibration_signals"]
            elif args.XJTU_channel == 'XY':  # 水平轴和垂直轴信号
                fl = fl
            else:
                print('the vibration signal with this channel is not exsisted!')

            fl = fl.values
            data11 = np.concatenate((data11, fl), axis=0)

        data[i].append(data11)

    data = np.array(data)
    data = data.squeeze(axis=1)

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

    train_x, test_x, train_y, test_y = train_test_split(dataset, labels, train_size=args.train_size, shuffle=True,stratify=labels)  # split train set and test set

    tensor_train_x = torch.Tensor(train_x)
    tensor_train_y = torch.LongTensor(train_y)
    tensor_test_x = torch.Tensor(test_x)
    tensor_test_y = torch.LongTensor(test_y)

    loader_train = DataLoader(TensorDataset(tensor_train_x, tensor_train_y), batch_size=args.batch_size)  # training set
    loader_test = DataLoader(TensorDataset(tensor_test_x, tensor_test_y), batch_size=args.batch_size)  # testing set

    return loader_train, loader_test