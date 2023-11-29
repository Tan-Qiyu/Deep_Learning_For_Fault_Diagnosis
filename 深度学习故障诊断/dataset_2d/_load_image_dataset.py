'''
Time: 2022/10/13 11:18
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/GNN_FD
'''

'''加载自定义图像数据集'''

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
from glob import glob
from tqdm import tqdm

def load_image_dataset(args):

    # --------索引图片数据集路径---------
    if args.dimension == 'image':  # 自定义图片数据集
        if args.dataset_name == 'CWRU':
            dataset_path = args.image_path + '\{}'.format(args.image_type) + '\{}'.format(args.dataset_name) + '\{}'.format(args.dir_path) + '\{}_{}'.format(args.noise, args.snr)
        elif args.dataset_name == 'SEU':
            if 'bearingset' in args.dataset_path:
                dataset_path = args.image_path + '\{}'.format(args.image_type) + '\{}'.format(args.dataset_name) + '\\bearing' + '\{}_{}'.format(args.noise, args.snr)
            elif 'gearset' in args.dataset_path:
                dataset_path = args.image_path + '\{}'.format(args.image_type) + '\{}'.format(args.dataset_name) + '\\gear' + '\{}_{}'.format(args.noise, args.snr)
        elif args.dataset_name == 'QianPeng_gear':
            dataset_path = args.image_path + '\{}'.format(args.image_type) + '\{}'.format(args.dataset_name) + '\{}'.format(args.QianPeng_rpm) + '\{}_{}'.format(args.noise, args.snr)
        else:
            dataset_path = args.image_path + '\{}'.format(args.image_type) + '\{}'.format(args.dataset_name) + '\{}_{}'.format(args.noise, args.snr)

    data_x_list = []  # 建立一个空列表存放信号的array数据，最终列表长度为per_class_number * classes，每个元素为256*256*3的array
    data_y_list = []  # 存放标签

    #--------遍历文件夹下的每个故障类型子文件夹----------------
    for label, each_state in enumerate(tqdm(glob(dataset_path+'\*'))):  # 遍历文件夹的子文件夹，n个子文件夹对应n种故障类型
        for img_idx, each_image in enumerate(glob(each_state + '\*.jpg')):  # 遍历子文件夹下后缀为Jpg的文件图片
            if img_idx < args.sample_num:
                image = cv2.imread(each_image)  # 读取输出格式为(h,w,c)，像素值在0-255
                data_x_list.append(image)
                data_y_list.append(label)
            else:
                break

    data_x = np.array(data_x_list)  # 将存放故障信号的列表转化为array，转化后的data_x得shape为样本数*h*w*c
    data_x = np.transpose(data_x,(0,3,1,2))  # 保持第一个维度即样本数不变，将通道数提前，即样本数*c*h*w
    data_y = np.array(data_y_list)

    #训练集、验证集、测试集划分
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, train_size=args.train_size, shuffle=True,stratify=data_y)

    #将numpy数据转化为tensor类型
    tensor_train_x = torch.Tensor(train_x)
    tensor_train_y = torch.LongTensor(train_y)
    tensor_test_x = torch.Tensor(test_x)
    tensor_test_y = torch.LongTensor(test_y)

    #训练集
    loader_train = DataLoader(TensorDataset(tensor_train_x, tensor_train_y), batch_size=args.batch_size)
    #测试集
    loader_test = DataLoader(TensorDataset(tensor_test_x, tensor_test_y), batch_size=args.batch_size)

    return loader_train,loader_test