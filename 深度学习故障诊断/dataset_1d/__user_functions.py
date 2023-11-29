'''
Time: 2022/10/12 0:26
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/GNN_FD
'''

import numpy as np
import math

# 定义滑窗采样及其样本重叠数
def Slide_window_sampling(data, window_size,overlap):
    '''
    :param data: the data raw signals with length n
    :param window_size: the sampling length of each samples
    :param overlap: the data shift length of neibor two samples
    :return squence: the data after sampling
    '''

    count = 0  # 初始化计数器
    data_length = int(data.shape[0])  # 信号长度
    sample_num = math.floor((data_length - window_size) / overlap + 1)  # 该输入信号得到的样本个数
    squence = np.zeros((sample_num, window_size), dtype=np.float32)  # 初始化样本数组
    for i in range(sample_num):
        squence[count] = data[overlap * i: window_size + overlap * i].T  # 更新样本数组
        count += 1  # 更新计数器

    return squence


# 添加噪声
def add_noise(data, noise, snr):
    '''
    :param x: the raw siganl
    :param snr: the signal to noise ratio
    :return: noise signal
    '''
    if noise == 1:  #add noise
        x = data
        d = np.random.randn(len(x))  # generate random noise
        P_signal = np.sum(abs(x) ** 2)
        P_d = np.sum(abs(d) ** 2)
        P_noise = P_signal / 10 ** (snr / 10)
        noise = np.sqrt(P_noise / P_d) * d
        noise_signal = x.reshape(-1) + noise
        return noise_signal
    else:  #Don't add noise
        return data

# 归一化方式
def norm(all_data,normal_type):

    if normal_type == 'unnormalization':
        return all_data

    elif normal_type == 'Z-score Normalization':
        all_data_norm = []
        for data in all_data:
            mean, var = data.mean(axis=0), data.var(axis=0)
            data_norm = (data - mean) / np.sqrt(var)
            all_data_norm.append(data_norm)
        return np.array(all_data_norm)

    elif normal_type == 'Max-Min Normalization':
        all_data_norm = []
        for data in all_data:
            maxvalue, minvalue = data.max(axis=0), data.min(axis=0)
            data_norm = (data - minvalue) / (maxvalue - minvalue)
            all_data_norm.append(data_norm)
        return np.array(all_data_norm)

    elif normal_type == '-1 1 Normalization':
        all_data_norm = []
        for data in all_data:
            maxvalue, minvalue = data.max(axis=0), data.min(axis=0)
            data_norm = -1 + 2 * ((data - minvalue) / (maxvalue - minvalue))
            all_data_norm.append(data_norm)
        return np.array(all_data_norm)

# 傅里叶变换
def FFT(x,domain):
    '''
    :param x: the raw signal
    :return: the signal after FFT
    '''
    if domain == 'TD':
        return x
    elif domain == 'FD':
        y = np.empty((x.shape[0], x.shape[1]))
        for i in range(x.shape[0]):
            y[i] = (np.abs(np.fft.fft(x[i])) / len(x[i])) #傅里叶变换、取幅值、归一化
        return y
