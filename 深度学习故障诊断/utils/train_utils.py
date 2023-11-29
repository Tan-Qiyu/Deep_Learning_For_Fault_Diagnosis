'''
Time: 2022/10/12 1:33
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/GNN_FD
'''

import datetime
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from 深度学习故障诊断.utils._confusion import confusion
from 深度学习故障诊断.utils._tsne import plot_embedding
from 深度学习故障诊断.dataset_1d.CWRU import CWRU_data
from 深度学习故障诊断.dataset_1d.SEU import SEU_data
from 深度学习故障诊断.dataset_1d.XJTU import XJTU_data
from 深度学习故障诊断.dataset_1d.Bearing311 import Bearing_311_data
from 深度学习故障诊断.dataset_1d.QianPeng_gear import QianPeng_data
from 深度学习故障诊断.dataset_1d.JNU import JNU_data
from 深度学习故障诊断.dataset_1d.MFPT import MFPT_data
from 深度学习故障诊断.dataset_1d.wind_gearbox import wind_gearbox_data
from 深度学习故障诊断.dataset_1d.UoC import UoC_data
from 深度学习故障诊断.dataset_1d.DC import DC_data
from 深度学习故障诊断.dataset_1d.SQV import SQV_data
from 深度学习故障诊断.dataset_2d._load_image_dataset import load_image_dataset
from 深度学习故障诊断.models.CNN_1D import CNN_1D
from 深度学习故障诊断.models.LeNet import LeNet
from 深度学习故障诊断.models.AlexNet import AlexNet
from 深度学习故障诊断.models.VGG import VGG
from 深度学习故障诊断.models.DenseNet import DenseNet
from 深度学习故障诊断.models.ResNet import ResNet18

def train_utils(args):

    if args.dataset_name == 'SEU':
        if 'bearingset' in args.dataset_path:
            save_name = args.dataset_name + '-bearing-' + args.model_type + '-' + str(args.noise) + '-' + str(args.snr) + '(' + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + ')'
        elif 'gearset' in args.dataset_path:
            save_name = args.dataset_name + '-gear-' + args.model_type + '-' + str(args.noise) + '-' + str(args.snr) + '(' + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + ')'
    else:
        save_name = args.dataset_name + '-' + args.model_type + '-' + str(args.noise) + '-' + str(args.snr) + '(' + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + ')'

    #==============================================================1、加载训练集、测试集===================================================

    if args.dimension == 'image':  # 自定义图片数据集
        loader_train, loader_test = load_image_dataset(args=args)

    elif args.dimension == '1D' or args.dimension == '2D':  # 振动信号或灰度图数据集
        if args.dataset_name == 'CWRU':
            loader_train, loader_test = CWRU_data(args)
        elif args.dataset_name == 'SEU':
            loader_train, loader_test = SEU_data(args)
        elif args.dataset_name == 'XJTU':
            loader_train, loader_test = XJTU_data(args)
        elif args.dataset_name == 'Bearing_311':
            loader_train, loader_test = Bearing_311_data(args)
        elif args.dataset_name == 'QianPeng_gear':
            loader_train, loader_test = QianPeng_data(args)
        elif args.dataset_name == 'JNU':
            loader_train, loader_test = JNU_data(args)
        elif args.dataset_name == 'MFPT':
            loader_train, loader_test = MFPT_data(args)
        elif args.dataset_name == 'Wind_Gearbox':
            loader_train, loader_test = wind_gearbox_data(args)
        elif args.dataset_name == 'UoC':
            loader_train, loader_test = UoC_data(args)
        elif args.dataset_name == 'DC':
            loader_train, loader_test = DC_data(args)
        elif args.dataset_name == 'SQV':
            loader_train, loader_test = SQV_data(args)

    # ==============================================================2、model===================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 故障类型数——数据输出通道
    classes = {'CWRU':10,'SEU':10,'XJTU':15,'Bearing_311':9,'QianPeng_gear':6,'JNU':12,'MFPT':15,'Wind_Gearbox':args.wind_mode,'UoC':9,'DC':10,'SQV':7}

    # 数据输入通道
    if args.dimension == '1D' or args.dimension == '2D':  # 一维振动信号/灰度图
        input_channel = 1
    elif args.dimension == 'image':
        input_channel = 3

    if args.model_type == 'CNN_1D':
        model = CNN_1D(in_channel=input_channel,out_channel=classes[args.dataset_name]).to(device)
    elif args.model_type == 'LeNet':
        model = LeNet(in_channel=input_channel,out_channel=classes[args.dataset_name]).to(device)
    elif args.model_type == 'AlexNet':
        model = AlexNet(in_channel=input_channel,out_channel=classes[args.dataset_name]).to(device)
    elif args.model_type == 'VGG':
        model = VGG(in_channel=input_channel,out_channel=classes[args.dataset_name]).to(device)
    elif args.model_type == 'DenseNet':
        model = DenseNet(in_channel=input_channel,out_channel=classes[args.dataset_name]).to(device)
    elif args.model_type == 'ResNet':
        model = ResNet18(in_channel=input_channel,out_channel=classes[args.dataset_name]).to(device)

    # ==============================================================3、超参数===================================================
    epochs = args.epochs
    lr = args.learning_rate
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif args.optimizer == 'Momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    elif args.optimizer == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    elif args.optimizer == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    elif args.optimizer == 'Adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    else:
        print('this optimizer is not existed!!!')

    # ==============================================================4、训练===================================================
    all_train_loss = []
    all_train_accuracy = []
    train_time = []

    for epoch in range(epochs):

        start = time.perf_counter()

        model.train()
        correct_train = 0
        train_loss = 0
        for step, (train_x,train_y) in enumerate(loader_train):
            train_x, train_y = train_x.to(device), train_y.to(device)
            train_out = model(train_x)
            loss = F.nll_loss(train_out, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = train_loss + loss.item()
            pre_train = torch.max(train_out.cpu(), dim=1)[1].data.numpy()
            correct_train = correct_train + (pre_train == train_y.cpu().data.numpy()).astype(int).sum()

        end = time.perf_counter()

        train_time.append(end - start)  # 记录训练时间

        train_accuracy = correct_train / len(loader_train.dataset)
        all_train_loss.append(train_loss)
        all_train_accuracy.append(train_accuracy)

        print('epoch：{} '
              '| train loss：{:.4f} '
              '| train accuracy：{}/{}({:.4f}) '
              '| train time：{}(s/epoch)'.format(
            epoch, train_loss, correct_train, len(loader_train.dataset),100 * train_accuracy, end - start))

    # ==============================================================5、测试===================================================
    y_fea = []
    list(map(lambda x: y_fea.append([]), range(len(model.get_fea()))))  # y_fea = [] 根据可视化的层数来创建相应数量的空列表存放特征

    prediction = np.empty(0, )  # 存放预测标签绘制混淆矩阵
    model.eval()
    correct_test = 0
    for (test_x,test_y) in loader_test:
        test_x, test_y = test_x.to(device), test_y.to(device)
        test_out = model(test_x)
        pre_test = torch.max(test_out.cpu(), dim=1)[1].data.numpy()
        correct_test = correct_test + (pre_test == test_y.cpu().data.numpy()).astype(int).sum()
        prediction = np.append(prediction, pre_test)  # 保存预测结果---混淆矩阵
        list(map(lambda j: y_fea[j].extend(model.get_fea()[j].cpu().detach().numpy()),range(len(y_fea))))  # 保存每一层特征---tsne

    test_accuracy = correct_test / len(loader_test.dataset)

    print('test accuracy：{}/{}({:.4f}%)'.format(correct_test,len(loader_test.dataset),100 * test_accuracy))
    print('all train time：{}(s/100epoch)'.format(np.array(train_time).sum()))

    """
    # 保存每个epoch的loss和accuracy
    import pandas as pd
    all_loss = np.array(all_train_loss)
    all_acc = np.array(all_train_accuracy)*100
    writer = pd.ExcelWriter(args.visualization_dirpath + '\loss_acc-{}.xlsx'.format(save_name)')  # 创建excel表格
    columns_list = list(range(1,101))
    all_loss = pd.DataFrame(all_loss)
    all_acc = pd.DataFrame(all_acc)
    all_loss.index = columns_list
    all_loss.columns = ['loss']
    all_acc.columns = ['acc']
    all_loss.to_excel(writer, 'sheet1', float_format='%.5f')
    all_acc.to_excel(writer, 'sheet1',index=False, float_format='%.2f',startrow=0,startcol=2)
    writer.save()
    """

    if args.visualization == True:

        # 混淆矩阵
        label = np.array(loader_test.dataset.tensors[1])  # the label of testing dataset
        confusion_data = confusion_matrix(label, prediction)  # confusion matrix
        confusion(confusion_matrix=confusion_data)  # 混淆矩阵绘制

        # 保存混淆矩阵
        if args.save_visualization_results == True:
            plt.savefig(args.visualization_dirpath + '\混淆矩阵-{}.tif'.format(save_name), dpi=300,bbox_inches='tight')  # 保存混淆矩阵图

        # tsne
        for num, fea in enumerate(y_fea):
            tsne = TSNE(n_components=2, init='pca')
            result = tsne.fit_transform(np.array(fea))  # 对特征进行降维
            fig = plot_embedding(result, label, classes=classes[args.dataset_name])

            # 保存tsne可视化结果
            if args.save_visualization_results == True:
                plt.savefig(args.visualization_dirpath + '\\tsne{}-{}.tif'.format(num + 1, save_name), dpi=300,bbox_inches='tight')  # 保存tsne特征可视化图

        plt.show()

    return test_accuracy
