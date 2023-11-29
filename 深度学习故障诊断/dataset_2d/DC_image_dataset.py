'''
Time: 2022/10/8 23:09
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/GNN_FD
'''

import numpy as np
import matplotlib
matplotlib.use('agg')
import os
from 深度学习故障诊断.dataset_2d._user_functions import Normal_signal,Slide_window_sampling,Add_noise
from 深度学习故障诊断.dataset_2d._gen_cwt_tf_image import gen_cwt_time_frequency_image
from 深度学习故障诊断.dataset_2d._gen_gaf_image import gaf_image
from 深度学习故障诊断.dataset_2d._gen_recurrence_image import RecurrencePlot_image
from 深度学习故障诊断.dataset_2d._gen_MarkovTransField_image import MarkovTransitionField_image
from tqdm import tqdm

def generate_DC_image(dataset_path,window_size,overlap,normalization,noise,snr,sample_number,image_type,save_path):
    root = dataset_path
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
    data = data[:, :, :(data.shape[2] // window_size) * window_size]  # shape: (10,43,5120)  when window_size == 1024

    data = data.reshape((data.shape[0], data.shape[1], data.shape[2] // window_size,
                         window_size))  # 将5120个数据按照window_size划分  shape: (10,43,5,1024)
    data = data.reshape((data.shape[0], data.shape[1] * data.shape[2], data.shape[3]))  # shape: (10,215,1024)
    data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))  # shape: (10,215*1024)

    # 添加噪声
    if noise == 1 or noise == 'y':
        noise_data = np.zeros((data.shape[0], data.shape[1]))
        for data_i in range(data.shape[0]):
            noise_data[data_i] = Add_noise(data[data_i], snr)
    else:
        noise_data = data

    # 滑窗采样
    sample_data = np.zeros((noise_data.shape[0], noise_data.shape[1] // window_size, window_size))
    for noise_data_i in range(noise_data.shape[0]):
        sample_data[noise_data_i] = Slide_window_sampling(noise_data[noise_data_i], window_size=window_size,
                                                          overlap=overlap)

    sample_data = sample_data[:, :sample_number, :]
    # 归一化
    if normalization != 'unnormalization':
        norm_data = np.zeros((sample_data.shape[0], sample_data.shape[1], sample_data.shape[2]))
        for sample_data_i in range(sample_data.shape[0]):
            norm_data[sample_data_i] = Normal_signal(sample_data[sample_data_i], normalization)
    else:
        norm_data = sample_data

    # ---------------------------------------------------------------------------------------------------------------------------------------------------
    # the dimention of sample_data is: classes * sample_length * channels
    for label, each_class_data in tqdm(enumerate(norm_data)):

        save_index = 1
        # 图像数据集文件夹命名规则：根目录 + 图像类型 + 故障数据集名称 + 噪声_信噪比 + 故障标签 + 编号.jpg
        if os.path.exists(save_path + '\{}'.format(image_type) + '\DC' + '\{}_{}'.format(noise, snr) + '\{}\\'.format(str(label))):  # 路径存在
            image_path = save_path + '\{}'.format(image_type) + '\DC' + '\{}_{}'.format(noise,snr) + '\{}\\'.format(str(label))
        else:  # 路径不存在
            os.makedirs(save_path + '\{}'.format(image_type) + '\DC' + '\{}_{}'.format(noise, snr) + '\{}\\'.format(str(label)))
            image_path = save_path + '\{}'.format(image_type) + '\DC' + '\{}_{}'.format(noise,snr) + '\{}\\'.format(str(label))

        for signal in each_class_data:
            if image_type == 'cwt_time_frequency_image':  # 小波成时频图
                gen_cwt_time_frequency_image(signal, image_path, save_index)
            elif image_type == 'GAF':  # 格拉姆角场
                gaf_image(image_size=32, signal=signal, image_path=image_path, save_index=save_index)
            elif image_type == 'Recurrence_image':  # 递归图
                RecurrencePlot_image(signal, image_path, save_index)
            elif image_type == 'MarkovTransField_image':  # 马尔可夫迁移场
                MarkovTransitionField_image(image_size=32, data=signal, image_path=image_path,
                                            save_index=save_index)
            elif image_type == 'STFT_image':  # 短时傅里叶变换时频图
                pass
            elif image_type == 'SNOW':  # 雪花图
                pass
            save_index += 1
