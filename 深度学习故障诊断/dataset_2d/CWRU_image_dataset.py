'''
Time: 2022/10/7 14:03
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/GNN_FD
'''

import numpy as np
import matplotlib
matplotlib.use('agg')
from scipy.io import loadmat
import os
from 深度学习故障诊断.dataset_2d._user_functions import Normal_signal,Slide_window_sampling,Add_noise
from 深度学习故障诊断.dataset_2d._gen_cwt_tf_image import gen_cwt_time_frequency_image
from 深度学习故障诊断.dataset_2d._gen_gaf_image import gaf_image
from 深度学习故障诊断.dataset_2d._gen_recurrence_image import RecurrencePlot_image
from 深度学习故障诊断.dataset_2d._gen_MarkovTransField_image import MarkovTransitionField_image
from tqdm import tqdm

#Normal Baseline Data由于仅有4个文件97.mat、98.mat、99.mat、100.mat，为了方便访问及处理，将四个文件都拷贝到其余三个文件夹中

#正常数据，负载0、1、2、3
NB = ['97.mat', '98.mat', '99.mat', '100.mat']

'''12k Drive End Bearing Fault Data'''
#内圈故障数据0.7mm、0.14mm、0.21mm/负载0、1、2、3
IR07_12DE = ['105.mat', '106.mat', '107.mat', '108.mat']
IR14_12DE = ['169.mat', '170.mat', '171.mat', '172.mat']
IR21_12DE = ['209.mat', '210.mat', '211.mat', '212.mat']

#外圈故障数据0.7mm、0.14mm、0.21mm/负载0、1、2、3 in Centerd @6:00
OR07_12DE = ['130.mat', '131.mat', '132.mat', '133.mat']
OR14_12DE = ['197.mat', '198.mat', '199.mat', '200.mat']
OR21_12DE = ['234.mat', '235.mat', '236.mat', '237.mat']

#滚动体故障数据0.7mm、0.14mm、0.21mm/负载0、1、2、3
B07_12DE = ['118.mat', '119.mat', '120.mat', '121.mat']
B14_12DE = ['185.mat', '186.mat', '187.mat', '188.mat']
B21_12DE = ['222.mat', '223.mat', '224.mat', '225.mat']

#全部数据，包括正常加九种不同类型、不同损伤程度的故障，一共十种
full_data_12DE = [NB,  IR07_12DE, IR14_12DE, IR21_12DE, OR07_12DE, OR14_12DE, OR21_12DE,B07_12DE, B14_12DE, B21_12DE]

'''48k Drive End Bearing Fault Data'''
#内圈故障数据0.7mm、0.14mm、0.21mm/负载0、1、2、3
IR07_48DE = ['109.mat', '110.mat', '111.mat', '112.mat']
IR14_48DE = ['174.mat', '175.mat', '176.mat', '177.mat']
IR21_48DE = ['213.mat', '214.mat', '215.mat', '217.mat']

#外圈故障数据0.7mm、0.14mm、0.21mm/负载0、1、2、3 in Centerd @6:00
OR07_48DE = ['135.mat', '136.mat', '137.mat', '138.mat']
OR14_48DE = ['201.mat', '202.mat', '203.mat', '204.mat']
OR21_48DE = ['238.mat', '239.mat', '240.mat', '241.mat']

#滚动体故障数据0.7mm、0.14mm、0.21mm/负载0、1、2、3
B07_48DE = ['122.mat', '123.mat', '124.mat', '125.mat']
B14_48DE = ['189.mat', '190.mat', '191.mat', '192.mat']
B21_48DE = ['226.mat', '227.mat', '228.mat', '229.mat']

#全部数据，包括正常加九种不同类型、不同损伤程度的故障，一共十种
full_data_48DE = [NB, IR07_48DE, IR14_48DE, IR21_48DE, OR07_48DE, OR14_48DE, OR21_48DE,B07_48DE, B14_48DE, B21_48DE]
full_data = [full_data_12DE,full_data_48DE]


def generate_CWRU_image(dataset_path,dir_path,window_size,overlap,normalization,noise,snr,sample_number,image_type,save_path):

    File_dir = dataset_path

    dirname = ['Normal Baseline Data', '12k Drive End Bearing Fault Data', '48k Drive End Bearing Fault Data']

    if dir_path == '12DE':
        data_path = os.path.join(File_dir, dirname[1])
        file_number = 0
    elif dir_path == '48DE':
        data_path = os.path.join(File_dir, dirname[2])
        file_number = 1


    for bearing_state in tqdm(enumerate(full_data[file_number])):

        save_index = 1  # 保存图片的序号
        for num,load in enumerate(bearing_state[1]):
            data = loadmat(os.path.join(data_path, load))
            if eval(load.split('.')[0]) < 100:
                vibration = data['X0' + load.split('.')[0] + '_DE_time']
            elif eval(load.split('.')[0]) == 174:
                vibration = data['X' + '173' + '_DE_time']
            else:
                vibration = data['X' + load.split('.')[0] + '_DE_time']

            #添加不同信噪比的噪声
            if noise == 'y' or noise == 1:
                vibration = Add_noise(vibration,snr).reshape(Add_noise(vibration,snr).shape[0],1)
            elif noise == 'n' or noise == 0:
                vibration = vibration

            slide_data = Slide_window_sampling(vibration, window_size, overlap)  # 滑窗采样
            if normalization != 'unnormalization':
                data_x = Normal_signal(slide_data,normalization)  # 归一化
            else:
                data_x = slide_data

            np.random.shuffle(data_x)  # 将数据shuffle
            data_x = data_x[:sample_number,]  # 限制生成图像的数量


            # 图像数据集文件夹命名规则：根目录 + 图像类型 + 故障数据集名称 + 噪声_信噪比 + 故障标签 + 编号.jpg
            if os.path.exists(save_path + '\{}'.format(image_type) + '\CWRU' + '\{}'.format(dir_path) + '\{}_{}'.format(noise,snr) + '\{}\\'.format(str(bearing_state[0]))):  # 路径存在
                image_path = save_path + '\{}'.format(image_type) + '\CWRU' + '\{}'.format(dir_path) + '\{}_{}'.format(noise,snr) + '\{}\\'.format(str(bearing_state[0]))
            else:  # 路径不存在
                os.makedirs(save_path + '\{}'.format(image_type) + '\CWRU' + '\{}'.format(dir_path) + '\{}_{}'.format(noise,snr) + '\{}\\'.format(str(bearing_state[0])))
                image_path = save_path + '\{}'.format(image_type) + '\CWRU' + '\{}'.format(dir_path) + '\{}_{}'.format(noise, snr) + '\{}\\'.format(str(bearing_state[0]))

            # 生成二维图像
            for signal in data_x:
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

