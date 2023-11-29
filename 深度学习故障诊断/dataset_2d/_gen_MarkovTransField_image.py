'''
Time: 2022/10/7 23:15
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/GNN_FD
'''

"""
生成马尔可夫迁移场
"""

import matplotlib.pyplot as plt
from pyts.image import MarkovTransitionField
import matplotlib
matplotlib.use('agg')

def MarkovTransitionField_image(image_size,data,image_path,save_index):
    # MTF transformation
    mtf = MarkovTransitionField(image_size=image_size)  # image_size马尔可夫迁移场图像的格数
    X_mtf = mtf.fit_transform(data.reshape(1,data.shape[0]))

    # Show the image for the first time series
    plt.figure(figsize=(2.56, 2.56))  # 定义画布大小
    plt.gca().xaxis.set_major_locator(plt.NullLocator())  # 去掉 x 轴
    plt.gca().yaxis.set_major_locator(plt.NullLocator())  # 去掉 y 轴
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # 去掉四周空白
    plt.margins(0, 0)  # 去掉四周空白

    plt.imshow(X_mtf[0], cmap='jet', origin='lower')

    #plt.title('Markov Transition Field', fontsize=18)
    #plt.colorbar(fraction=0.0457, pad=0.04)
    #plt.tight_layout()

    plt.savefig(image_path + str(save_index) + '.jpg')
    plt.close()

