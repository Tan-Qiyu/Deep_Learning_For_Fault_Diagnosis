'''
Time: 2022/10/8 22:09
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/GNN_FD
'''

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import os
from 深度学习故障诊断.dataset_2d._user_functions import Normal_signal,Slide_window_sampling,Add_noise
from 深度学习故障诊断.dataset_2d._gen_cwt_tf_image import gen_cwt_time_frequency_image
from 深度学习故障诊断.dataset_2d._gen_gaf_image import gaf_image
from 深度学习故障诊断.dataset_2d._gen_recurrence_image import RecurrencePlot_image
from 深度学习故障诊断.dataset_2d._gen_MarkovTransField_image import MarkovTransitionField_image
from tqdm import tqdm

def generate_XJTU_image(dataset_path,channel,minute_value,window_size,overlap,normalization,noise,snr,sample_number,image_type,save_path):
    WC = os.listdir(dataset_path)  # 遍历根目录，目录下包括三种工况的文件夹名

    datasetname1 = os.listdir(os.path.join(dataset_path, WC[0]))
    datasetname2 = os.listdir(os.path.join(dataset_path, WC[1]))
    datasetname3 = os.listdir(os.path.join(dataset_path, WC[2]))

    data_name = []
    data_name.extend(datasetname1)
    data_name.extend(datasetname2)
    data_name.extend(datasetname3)

    # 工况1数据及标签
    data = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    for i in tqdm(range(len(data_name))):
        if i >= 0 and i <= 4:
            dir = os.path.join('/tmp', dataset_path, WC[0], data_name[i])  # 工况1：35Hz12kN
            files = os.listdir(dir)
        elif i > 4 and i <= 9:
            dir = os.path.join('/tmp', dataset_path, WC[1], data_name[i])  # 工况2：37.5Hz11kN
            files = os.listdir(dir)
        elif i > 9 and i <= 14:
            dir = os.path.join('/tmp', dataset_path, WC[2], data_name[i])  # 工况3：40Hz10kN
            files = os.listdir(dir)

        # 提取振动信号最后故障时刻的采样值时，csv文件的读取顺序不是按照数字排序进行读取的，例如123在99之前，故根据.csv文件的前缀名进行读取
        files_list = list(map(lambda x: int(x[:-4]), files))
        load_file_name = list(range(np.array(files_list).max() - minute_value + 1,
                                    np.array(files_list).max() + 1))  # 取出最后minute_value分钟内需要处理的故障数据
        load_file_name = list(map(lambda y: str(y) + '.csv', load_file_name))

        data11 = np.empty((0,))
        for ii in range(minute_value):  # Take the data of the last three CSV files
            path1 = os.path.join(dir, load_file_name[ii])
            fl = pd.read_csv(path1)
            if channel == 'X':  # 水平轴信号
                fl = fl["Horizontal_vibration_signals"]
            elif channel == 'Y':  # 垂直轴信号
                fl = fl["Vertical_vibration_signals"]
            elif channel == 'XY':  # 水平轴和垂直轴信号
                fl = fl
            else:
                print('the vibration signal with this channel is not exsisted!')

            fl = fl.values
            data11 = np.concatenate((data11, fl), axis=0)

        data[i].append(data11)

    data = np.array(data)
    data = data.squeeze(axis=1)

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

    # ---------------------------------------------------------------------------------------------------------------------------------------------------
    # the dimention of sample_data is: classes * sample_length * channels
    for label, each_class_data in tqdm(enumerate(norm_data)):

        save_index = 1
        # 图像数据集文件夹命名规则：根目录 + 图像类型 + 故障数据集名称 + 噪声_信噪比 + 故障标签 + 编号.jpg
        if os.path.exists(save_path + '\{}'.format(image_type) + '\XJTU' + '\{}_{}'.format(noise,snr) + '\{}\\'.format(str(label))):  # 路径存在
            image_path = save_path + '\{}'.format(image_type) + '\XJTU' + '\{}_{}'.format(noise,snr) + '\{}\\'.format(str(label))
        else:  # 路径不存在
            os.makedirs(save_path + '\{}'.format(image_type) + '\XJTU' + '\{}_{}'.format(noise,snr) + '\{}\\'.format(str(label)))
            image_path = save_path + '\{}'.format(image_type) + '\XJTU' + '\{}_{}'.format(noise,snr) + '\{}\\'.format(str(label))

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