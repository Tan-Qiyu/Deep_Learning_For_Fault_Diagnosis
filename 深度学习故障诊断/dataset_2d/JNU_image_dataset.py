'''
Time: 2022/10/8 22:40
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

def generate_JNU_image(dataset_path,window_size,overlap,normalization,noise,snr,sample_number,image_type,save_path):
    root = dataset_path

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
        if os.path.exists(save_path + '\{}'.format(image_type) + '\JNU' + '\{}_{}'.format(noise, snr) + '\{}\\'.format(str(label))):  # 路径存在
            image_path = save_path + '\{}'.format(image_type) + '\JNU' + '\{}_{}'.format(noise,snr) + '\{}\\'.format(str(label))
        else:  # 路径不存在
            os.makedirs(save_path + '\{}'.format(image_type) + '\JNU' + '\{}_{}'.format(noise, snr) + '\{}\\'.format(str(label)))
            image_path = save_path + '\{}'.format(image_type) + '\JNU' + '\{}_{}'.format(noise,snr) + '\{}\\'.format(str(label))

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