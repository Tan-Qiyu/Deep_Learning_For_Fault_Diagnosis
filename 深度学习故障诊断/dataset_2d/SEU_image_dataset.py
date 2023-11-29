'''
Time: 2022/10/8 21:30
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/GNN_FD
'''

import numpy as np
import matplotlib
matplotlib.use('agg')
import os
from itertools import islice
from 深度学习故障诊断.dataset_2d._user_functions import Normal_signal,Slide_window_sampling,Add_noise
from 深度学习故障诊断.dataset_2d._gen_cwt_tf_image import gen_cwt_time_frequency_image
from 深度学习故障诊断.dataset_2d._gen_gaf_image import gaf_image
from 深度学习故障诊断.dataset_2d._gen_recurrence_image import RecurrencePlot_image
from 深度学习故障诊断.dataset_2d._gen_MarkovTransField_image import MarkovTransitionField_image
from tqdm import tqdm


def generate_SEU_image(dataset_path,channel,window_size,overlap,normalization,noise,snr,sample_number,image_type,save_path):
    root = dataset_path
    if "bearingset" in root:  # Path of bearingset
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
                if i < sample_number * window_size:  # 限制信号长度，因为信号总长为1048560，8传感器，10类型，即1048560*8*10接近上亿的数值会对计算机造成一定的影响
                    line = line.rstrip()  # 移除每一行末尾的字符
                    word = line.split(",", 8)  # 使用8个,进行分隔
                    word = list(map(float, word[:-1]))  # 将每行的str数据转化为float数值型，并移除最后一个空元素
                    data_list[num_state].append(word)  # 将故障信号存在相应的列表内
                    i += 1
                else:
                    break

        else:
            for line in islice(b_g_data, 16, None):
                if i < sample_number * window_size:
                    line = line.rstrip()
                    word = line.split("\t", 8)
                    word = list(map(float, word))
                    data_list[num_state].append(word)
                    i += 1
                else:

                    break

    # 振动信号
    all_data = np.array(data_list)  # the dimention of bearing is: classes * sample_length * channels

    data = all_data[:, :, channel]

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
        sample_data[noise_data_i] = Slide_window_sampling(noise_data[noise_data_i], window_size=window_size,overlap=overlap)

    sample_data = sample_data[:, :sample_number, :]

    # 归一化
    if normalization != 'unnormalization':
        norm_data = np.zeros((sample_data.shape[0], sample_data.shape[1], sample_data.shape[2]))
        for sample_data_i in range(sample_data.shape[0]):
            norm_data[sample_data_i] = Normal_signal(sample_data[sample_data_i], normalization)
    else:
        norm_data = sample_data

    if 'bearingset' in root:
        name = 'bearing'
    elif 'gearset' in root:
        name = 'gear'

    #---------------------------------------------------------------------------------------------------------------------------------------------------
    # the dimention of norm_data is: classes * sample_number * sample_length
    for label,each_class_data in tqdm(enumerate(norm_data)):

        save_index = 1
        # 图像数据集文件夹命名规则：根目录 + 图像类型 + 故障数据集名称 + 噪声_信噪比 + 故障标签 + 编号.jpg
        if os.path.exists(save_path + '\{}'.format(image_type) + '\SEU' + '\{}'.format(name) + '\{}_{}'.format(noise,snr) + '\{}\\'.format(str(label))):  # 路径存在
            image_path = save_path + '\{}'.format(image_type) + '\SEU' + '\{}'.format(name) + '\{}_{}'.format(noise,snr) + '\{}\\'.format(str(label))
        else:  # 路径不存在
            os.makedirs(save_path + '\{}'.format(image_type) + '\SEU' + '\{}'.format(name) + '\{}_{}'.format(noise,snr) + '\{}\\'.format(str(label)))
            image_path = save_path + '\{}'.format(image_type) + '\SEU' + '\{}'.format(name) + '\{}_{}'.format(noise,snr) + '\{}\\'.format(str(label))

        for signal in each_class_data:
            if image_type == 'cwt_time_frequency_image':  # 小波成时频图
                gen_cwt_time_frequency_image(signal, image_path, save_index)
            elif image_type == 'GAF':  # 格拉姆角场
                gaf_image(image_size=32,signal=signal,image_path=image_path,save_index=save_index)
            elif image_type == 'Recurrence_image':  # 递归图
                RecurrencePlot_image(signal,image_path,save_index)
            elif image_type == 'MarkovTransField_image':  # 马尔可夫迁移场
                MarkovTransitionField_image(image_size=32,data=signal,image_path=image_path,save_index=save_index)
            elif image_type == 'STFT_image':  # 短时傅里叶变换时频图
                pass
            elif image_type == 'SNOW':  # 雪花图
                pass
            save_index += 1



