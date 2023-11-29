'''
Time: 2022/10/8 22:53
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/GNN_FD
'''

import numpy as np
import matplotlib
matplotlib.use('agg')
import os
from scipy.io import loadmat
from 深度学习故障诊断.dataset_2d._user_functions import Normal_signal,Slide_window_sampling,Add_noise
from 深度学习故障诊断.dataset_2d._gen_cwt_tf_image import gen_cwt_time_frequency_image
from 深度学习故障诊断.dataset_2d._gen_gaf_image import gaf_image
from 深度学习故障诊断.dataset_2d._gen_recurrence_image import RecurrencePlot_image
from 深度学习故障诊断.dataset_2d._gen_MarkovTransField_image import MarkovTransitionField_image
from tqdm import tqdm

def generate_WindGear_image(dataset_path,fault_mode,window_size,overlap,normalization,noise,snr,sample_number,image_type,save_path):

    root = dataset_path

    dir = ['data2', 'data1']  # data2 : 6 health state signal file;  data1 : 11 health state signal file
    mat_name = [['case_1.mat', 'case_2.mat', 'case_3.mat', 'case_4.mat', 'case_5.mat', 'case_6.mat'],
                ['case_1.mat', 'case_2.mat', 'case_3.mat', 'case_4.mat', 'case_5.mat', 'case_6.mat', 'case_7.mat',
                 'case_8.mat', 'case_9.mat', 'case_10.mat', 'case_11.mat']]

    if fault_mode == 17:  # 17分类
        data = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]  # 创建一个长度为17的列表存放17种状态信号
        data_index = 0
        for num, each_dir in enumerate(dir):
            each_mat = mat_name[num]
            for each_class in each_mat:
                file = loadmat(os.path.join(root, each_dir, each_class))
                data[data_index].append(file['gs'].squeeze(axis=1))

                data_index = data_index + 1

        data = np.array(data).squeeze(axis=1)  # shape: (17,585936)

    elif fault_mode == 2:  # 2分类
        data = [[], []]  # 创建一个长度为2的列表存放2种状态信号
        data_index = 0
        for num, each_dir in enumerate(dir):
            each_mat = mat_name[num]
            for each_class in each_mat:
                file = loadmat(os.path.join(root, each_dir, each_class))
                data[data_index].extend(file['gs'].squeeze(axis=1))

            data_index = data_index + 1

        data1 = [[], []]
        data1[0], data1[1] = data[0], data[1][:len(data[0])]

        data = np.array(data1)  # shape: (17,585936)

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
        if os.path.exists(save_path + '\{}'.format(image_type) + '\Wind_Gearbox' + '\{}'.format(fault_mode) + '\{}_{}'.format(noise, snr) + '\{}\\'.format(str(label))):  # 路径存在
            image_path = save_path + '\{}'.format(image_type) + '\Wind_Gearbox' + '\{}'.format(fault_mode) + '\{}_{}'.format(noise,snr) + '\{}\\'.format(str(label))
        else:  # 路径不存在
            os.makedirs(save_path + '\{}'.format(image_type) + '\Wind_Gearbox' + '\{}'.format(fault_mode) + '\{}_{}'.format(noise, snr) + '\{}\\'.format(str(label)))
            image_path = save_path + '\{}'.format(image_type) + '\Wind_Gearbox' + '\{}'.format(fault_mode) + '\{}_{}'.format(noise,snr) + '\{}\\'.format(str(label))

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