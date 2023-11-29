'''
Time: 2022/10/7 13:59
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/GNN_FD
'''

"""
小波变换生成时频图像
"""

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pywt
import cv2
import PIL.Image as Image

def Wavelet(input_signal, sampling_rate=1024, wavename='cmor3-3', totalscal=256):  # 小波变换，cmor3-3为母小波
    fc = pywt.central_frequency(wavename)  # 小波的中心频率
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)
    # 求连续小波系数和频率
    [cwtmatr, frequencies] = pywt.cwt(input_signal, scales, wavename, 1.0 / sampling_rate)

    return cwtmatr, frequencies

def build_figure(cwtmatr, frequencies):
    figure = plt.figure(figsize=(2.56, 2.56))  # 定义画布大小
    plt.gca().xaxis.set_major_locator(plt.NullLocator())  # 去掉 x 轴
    plt.gca().yaxis.set_major_locator(plt.NullLocator())  # 去掉 y 轴
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # 去掉四周空白
    plt.margins(0, 0)  # 去掉四周空白
    Window_size = 1024
    t = np.arange(Window_size)
    plt.contourf(t, frequencies, abs(cwtmatr))
    plt.pcolormesh(t, frequencies, abs(cwtmatr), shading='auto', cmap='jet')  # cmap控制生成图像的格式颜色等
    plt.axis('off')  # 不显示坐标轴

    plt.close(figure)

    return figure

def fig2data(fig):
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    image = np.asarray(image)
    return image

def gen_cwt_time_frequency_image(input_signal, path, index):  # 小波变换生成时频图

    cwtmatr, frequencies = Wavelet(input_signal)  # 连续小波系数和频率

    figure = build_figure(cwtmatr, frequencies)  # 绘图

    # 将图片转换成数组格式
    image = fig2data(figure)
    # print(image.shape)

    # 保存图片，注意保存路径不能包含中文
    cv2.imwrite(path + str(index) + '.jpg', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # print('已保存')

    return image