'''
Time: 2022/10/8 22:20
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

def generate_bearing311_image(dataset_path,window_size,overlap,normalization,noise,snr,sample_number,image_type,save_path):
    file_dir = dataset_path

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
                    file = open(os.path.join(file_dir, each_file_name, each_ball_servity_name, each_data_name),encoding='gbk')
                    for line in file:
                        data[num + 3].append(float(line.strip()))  # 滚动体较轻故障信号存在data[3]中，保持架较严重故障信号存在data[4]中

        elif each_file_name == '内圈':
            for num, each_inner_servity_name in enumerate(inner_servity_name):
                for each_data_name in data_name:
                    file = open(os.path.join(file_dir, each_file_name, each_inner_servity_name, each_data_name),encoding='gbk')
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
        if os.path.exists(save_path + '\{}'.format(image_type) + '\Bearing_311' + '\{}_{}'.format(noise, snr) + '\{}\\'.format(str(label))):  # 路径存在
            image_path = save_path + '\{}'.format(image_type) + '\Bearing_311' + '\{}_{}'.format(noise,snr) + '\{}\\'.format(str(label))
        else:  # 路径不存在
            os.makedirs(save_path + '\{}'.format(image_type) + '\Bearing_311' + '\{}_{}'.format(noise, snr) + '\{}\\'.format(str(label)))
            image_path = save_path + '\{}'.format(image_type) + '\Bearing_311' + '\{}_{}'.format(noise,snr) + '\{}\\'.format(str(label))

        for signal in each_class_data:
            if image_type == 'cwt_time_frequency_image':  # 小波成时频图
                gen_cwt_time_frequency_image(signal, image_path, save_index)
            elif image_type == 'GAF':  # 格拉姆角场
                gaf_image(image_size=32, signal=signal, image_path=image_path, save_index=save_index)
            elif image_type == 'Recurrence_image':  # 递归图
                RecurrencePlot_image(signal, image_path, save_index)
            elif image_type == 'MarkovTransField_image':  # 马尔可夫迁移场
                MarkovTransitionField_image(image_size=32, data=signal, image_path=image_path,save_index=save_index)
            elif image_type == 'STFT_image':  # 短时傅里叶变换时频图
                pass
            elif image_type == 'SNOW':  # 雪花图
                pass
            save_index += 1