'''
Time: 2022/10/7 15:42
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/GNN_FD
'''

"""
生成格拉姆角场图像
"""

import matplotlib.pyplot as plt
import matplotlib
from pyts.image import GramianAngularField
matplotlib.use('agg')

def gaf_image(image_size,signal, image_path, save_index):

    data = signal.reshape(1,signal.shape[0])
    gasf = GramianAngularField(image_size=image_size, method='summation')
    sin_gasf = gasf.fit_transform(data)

    plt.figure(figsize=(2.56, 2.56))  # 定义画布大小
    plt.gca().xaxis.set_major_locator(plt.NullLocator())  # 去掉 x 轴
    plt.gca().yaxis.set_major_locator(plt.NullLocator())  # 去掉 y 轴
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # 去掉四周空白
    plt.margins(0, 0)  # 去掉四周空白

    plt.imshow(sin_gasf[0])
    plt.grid(False)

    plt.savefig(image_path + str(save_index) + '.jpg')
    plt.axis('off')
    plt.clf()
    plt.close()

