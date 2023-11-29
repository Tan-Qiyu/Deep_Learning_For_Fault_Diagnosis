'''
Time: 2022/10/7 22:45
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/GNN_FD
'''

"""
生成递归图
"""

import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot
import matplotlib
matplotlib.use('agg')


def RecurrencePlot_image(data,image_path,save_index):
    # Recurrence plot transformation
    rp = RecurrencePlot(threshold='point', percentage=20)
    X_rp = rp.fit_transform(data.reshape(1,data.shape[0]))

    # Show the results for the first time series
    plt.figure(figsize=(2.56, 2.56))  # 定义画布大小
    plt.gca().xaxis.set_major_locator(plt.NullLocator())  # 去掉 x 轴
    plt.gca().yaxis.set_major_locator(plt.NullLocator())  # 去掉 y 轴
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # 去掉四周空白
    plt.margins(0, 0)  # 去掉四周空白

    plt.imshow(X_rp[0], cmap='jet', origin='lower')  # cmap控制生成图像的颜色
    #plt.imshow(X_rp[0])
    # plt.title('Recurrence Plot', fontsize=16)

    plt.savefig(image_path + str(save_index) + '.jpg')
    plt.close()