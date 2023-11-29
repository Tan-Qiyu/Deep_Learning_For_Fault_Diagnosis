'''
Time: 2022/10/8 23:13
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

def generate_SQV_image(dataset_path,window_size,overlap,normalization,noise,snr,sample_number,image_type,save_path):
    root = dataset_path

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
        if os.path.exists(save_path + '\{}'.format(image_type) + '\SQV' + '\{}_{}'.format(noise, snr) + '\{}\\'.format(str(label))):  # 路径存在
            image_path = save_path + '\{}'.format(image_type) + '\SQV' + '\{}_{}'.format(noise,snr) + '\{}\\'.format(str(label))
        else:  # 路径不存在
            os.makedirs(save_path + '\{}'.format(image_type) + '\SQV' + '\{}_{}'.format(noise, snr) + '\{}\\'.format(str(label)))
            image_path = save_path + '\{}'.format(image_type) + '\SQV' + '\{}_{}'.format(noise,snr) + '\{}\\'.format(str(label))

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
