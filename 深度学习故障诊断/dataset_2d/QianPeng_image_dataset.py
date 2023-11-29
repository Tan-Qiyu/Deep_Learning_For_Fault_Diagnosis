'''
Time: 2022/10/8 22:29
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

def generate_QianPeng_image(dataset_path,rpm,channel,window_size,overlap,normalization,noise,snr,sample_number,image_type,save_path):
    dir = ['正常运行下', '点蚀', '点蚀磨损', '断齿', '断齿、磨损混合故障', '磨损']
    if rpm == 880:
        txt_name = [['normal880.txt', 'normal880-1.txt', 'normal880-2.txt', 'normal880-3.txt'],
                    ['dianshi880.txt', 'dianshi880-1.txt', 'dianshi880-2.txt', 'dianshi880-3.txt'],
                    ['dianmo880.txt', 'dianmo880-1.txt', 'dianmo880-2.txt', 'dianmo880-3.txt'],
                    ['duanchi880.txt', 'duanchi880-1.txt', 'duanchi880-2.txt', 'duanchi880-3.txt'],
                    ['duanmo880.txt', 'duanmo880-1.txt', 'duanmo880-2.txt', 'duanmo880-3.txt'],
                    ['mosun880.txt', 'mosun880-1.txt', 'mosun880-2.txt', 'mosun880-3.txt']]
    elif rpm == 1470:
        txt_name = [['normal1500.txt'],
                    ['dianshi1470.txt', 'dianshi1500.txt'],
                    ['dianmo1470.txt'],
                    ['duanchi1500.txt'],
                    ['duanmo1470.txt'],
                    ['mosun1470.txt']]

    data = []
    for dir_num, each_dir in enumerate(dir):
        sub_txt = txt_name[dir_num]
        subdata = []
        for each_txt in sub_txt:
            file_name = os.path.join(dataset_path, each_dir, each_txt)
            with open(file_name, encoding='gbk') as f:
                for line in f.readlines()[1:]:  # 从第二行开始读取，第一行为空格
                    line = line.strip('\t\n')  # 删除每一行的最后一个制表符和换行符号
                    line = line.split('\t')  # 按制表符\t进行分隔
                    subdata.append(list(map(lambda x: float(x), line)))  # 将字符数据转化为数字

        subdata = np.array(subdata)  # shape:(样本数，通道数)
        data.append(subdata)

    # 当每一类故障的样本不平衡时截取相同数量的样本
    sam_list = []
    list(map(lambda x: sam_list.append(x.shape[0]), data))
    sam_min = min(sam_list)
    max_min = max(sam_list)
    if sam_min != max_min:
        balance_data = [[], [], [], [], [], []]
        for all_data_index, class_data in enumerate(data):
            # np.random.shuffle(class_data)
            balance_data[all_data_index].append(data[all_data_index][:sam_min, :])
        data = np.array(balance_data).squeeze(axis=1)

    data = np.array(data)  # shape:(故障类型数，样本数，通道数)
    data = data[:, :, channel]  # 选择通道

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
        if os.path.exists(save_path + '\{}'.format(image_type) + '\QianPeng_gear' + '\{}'.format(rpm) + '\{}_{}'.format(noise, snr) + '\{}\\'.format(str(label))):  # 路径存在
            image_path = save_path + '\{}'.format(image_type) + '\QianPeng_gear' + '\{}'.format(rpm) + '\{}_{}'.format(noise,snr) + '\{}\\'.format(str(label))
        else:  # 路径不存在
            os.makedirs(save_path + '\{}'.format(image_type) + '\QianPeng_gear' + '\{}'.format(rpm) + '\{}_{}'.format(noise, snr) + '\{}\\'.format(str(label)))
            image_path = save_path + '\{}'.format(image_type) + '\QianPeng_gear' + '\{}'.format(rpm) + '\{}_{}'.format(noise,snr) + '\{}\\'.format(str(label))

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

