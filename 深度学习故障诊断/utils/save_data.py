'''
Time: 2022/10/14 14:04
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/GNN_FD
'''

"""
保存多次实验的结果为excel文件
"""

import datetime
import pandas as pd
import numpy as np
from 深度学习故障诊断.utils.train_utils import train_utils


def save_data(args,trials):
    if args.dataset_name == 'SEU':
        if 'bearingset' in args.dataset_path:
            save_name = args.dataset_name + '-bearing-' + args.model_type + '-' + str(args.noise) + '-' + str(args.snr) + '(' + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + ')'
        elif 'gearset' in args.dataset_path:
            save_name = args.dataset_name + '-gear-' + args.model_type + '-' + str(args.noise) + '-' + str(args.snr) + '(' + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + ')'
    else:
        save_name = args.dataset_name + '-' + args.model_type + '-' + str(args.noise) + '-' + str(args.snr) + '(' + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + ')'

    writer = pd.ExcelWriter(args.save_data_dirpath + "\{}.xlsx".format(save_name))

    all_acc = []
    for i in range(trials):

        acc = train_utils(args)

        all_acc.append(acc)
    all_acc = np.array(all_acc)
    print(all_acc)
    mean = all_acc.mean()
    std = all_acc.std()
    all_result = np.append(all_acc,mean)
    all_result = np.append(all_result,std)
    all_result = all_result * 100
    print(all_result)

    columns_list = list(map(lambda i: 'trial ' + str(i + 1), range(trials)))
    columns_list.append('mean value')
    columns_list.append('std value')
    all_result = pd.DataFrame(all_result)
    all_result.index = columns_list
    all_result.columns = ['{}_{}'.format(args.noise,args.snr)]
    all_result.to_excel(writer,'sheet1',float_format='%.2f')

    writer.save()
