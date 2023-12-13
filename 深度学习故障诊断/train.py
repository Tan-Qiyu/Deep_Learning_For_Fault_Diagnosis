'''
Time: 2022/10/8 23:27
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/Deep_Learning_For_Fault_Diagnosis
Dataset Download Link：https://github.com/Tan-Qiyu/Mechanical_Fault_Diagnosis_Dataset
引用格式:[1]谭启瑜,马萍,张宏立.基于图卷积神经网络的滚动轴承故障诊断[J].噪声与振动控制,2023,43(06):101-108+116.
'''

'''
--------------------------------------------------参数介绍------------------------------------------
dataset_name： 数据集名称
    CWRU、SEU（bearing、gear）、XJTU、Bearing_311、QianPeng_gear、JNU、MFPT、Wind_Gearbox、UoC、DC、SQV 共11个公开数据集

dataset_path： 数据集目录地址
    CWRU  "E:\故障诊断数据集\凯斯西储大学数据" -------------------------------------------# 凯斯西储大学轴承数据集
    SEU   "E:\故障诊断数据集\东南大学\Mechanical-datasets\gearbox\bearingset" --------- -# 东南大学轴承子数据
    SEU   "E:\故障诊断数据集\东南大学\Mechanical-datasets\gearbox\gearset" --------------# 东南大学齿轮子数据
    XJTU  "E:\故障诊断数据集\XJTU-SY_Bearing_Datasets\Data\XJTU-SY_Bearing_Datasets"----# 西安交通大学-昇阳轴承退化数据集
    Bearing_311    "E:\故障诊断数据集\LW轴承实验数据2016(西安交通大学)"-----------------------# 西交试验台数据集：马萍、张西宁等
    QianPeng_gear  "E:\故障诊断数据集\齿轮箱故障数据_千鹏公司"--------------------------------# 千鹏公司齿轮箱数据集
    JNU   "E:\故障诊断数据集\江南大学轴承数据(一)\数据"-------------------------------------# 江南大学轴承数据集
    MFPT  "E:\故障诊断数据集\MFPT Fault Data Sets"--------------------------------------# 美国-机械故障预防技术学会MFPT数据集
    Wind_Gearbox   "E:\故障诊断数据集\fault-dataset-collection-main\fault-dataset-collection-main\HS_gear"--------# 风机齿轮箱数据集
    UoC   "E:\故障诊断数据集\美国-康涅狄格大学-齿轮数据集"------------------------------------# 美国-康涅狄格大学-齿轮数据集
    DC    "E:\故障诊断数据集\DC轴承数据"--------------------------------------------------# 中国-轴承数据集（DC竞赛）
    SQV   "E:\故障诊断数据集\SQV-public"-------------------------------------------------# SQV轴承变转速数据集

dir_path:   CWRU数据集的采样频率和传感器位置
    12DE  # 12kHZ Drive End Dataset
    48DE  # 48kHZ Drive End Dataset

SEU_channel:  SEU数据集的数据通道
    0、1、2、3、4、5、6、7  共8个通道

minute_value：  XJTU-SY数据集使用最后多少个文件数据进行实验验证

XJTU_channel：   XJTU-SY数据集的数据通道
    X Y XY  共3种通道

QianPeng_rpm:   QianPeng_gear数据集的电机转速（rpm）
    880 1470

QianPeng_channel:   QianPeng_gear数据集的信号通道
    0、1、2、3、4、5、6、7

wind_mode：  Wind_Gearbox数据集分类任务
    2 17

sample_num：   每一种故障类型的样本数
    CWRU：一维振动信号输入/灰度图输入——每种故障类型的每种电机转速下的样本数，每一类故障样本数为sample_num * 工况数 = sample_num * 4；
    其余数据集包括CWRU自定义图像输入时为每一种故障下的一种工况的样本数

train_size： 训练集比例

sample_length：  样本采样长度 = 网络的输入特征长度

overlap：   滑窗采样偏移量，当sample_length = overlap时为无重叠顺序采样

norm_type：  原始振动信号的归一化方式
    unnormalization        # 不进行归一化
    Z-score Normalization  # 均值方差归一化
    Max-Min Normalization  # 最大-最小归一化：归一化到0到1之间
    -1 1 Normalization     # 归一化到-1到1之间

noise:   是否往原始振动信号添加噪声
    0   # 不添加噪声
    1   # 添加噪声

snr：   当noise = 1时添加的噪声的信噪比大小；当noise = 0时此参数无效

input_type:   输入节点特征的类型
    TD  # 时域输入
    FD  # 频域输入

dimension:      输入信号的数据类型
    1D      # 一维振动信号（时域/频域）
    2D      # 一维振动信号构成的灰度图
    image   # 自定义图像数据集（例如时频图等，需要提前生成）

image_path:   自定义图像数据集的根目录
    r"E:\FD_image_datasets"

image_type:   自定义图像数据集的类型
    cwt_time_frequency_image  # 小波成时频图
    GAF                       # 格拉姆角场
    Recurrence_image          # 递归图
    MarkovTransField_image    # 马尔可夫迁移场
    STFT_image                # 短时傅里叶变换时频图
    
batch_size:   批处理大小

model_type:  网络模型
    CNN_1D、 LeNet、 AlexNet、 VGG、 DenseNet、 ResNet

epochs：  迭代轮数

learning_rate：  学习率

momentum：  动量因子

optimizer:  优化器

visualization:  是否绘制混淆矩阵与每一层网络的t-SNE可视化图
    True   # 进行可视化
    False  # 不进行可视化

visualization_dirpath:   可视化结果的保存路径（仅save_visualization_results = True时成立）
    保存路径为： visualization_dirpath + \混淆矩阵-dataset_name-noise-snr(年月日 时分秒).tif'  # 混淆矩阵
               visualization_dirpath + \tsne1/2/3/4...-dataset_name-noise-snr(年月日 时分秒).tif'  # tsne特征可视化

save_data：   是否保存多次实验结果并存为excel文件
    True   # 保存
    False  # 不保存

save_data_dirpath：   多次实验结果存为excel文件的路径
    保存路径：  args.save_data_dirpath + "\dataset_name-noise-snr(年月日 时分秒).xlsx".format(save_name)

'''

import argparse
from 深度学习故障诊断.utils.train_utils import train_utils
from 深度学习故障诊断.utils.save_data import save_data

def parse_args():
    parser = argparse.ArgumentParser()
    # basic parameters
    #===================================================dataset_2d parameters=============================================================================
    parser.add_argument('--dataset_name', type=str, default='SEU', help='the name of the dataset')
    parser.add_argument('--dataset_path', type=str, default=r"E:\故障诊断数据集\东南大学\Mechanical-datasets\gearbox\bearingset", help='the file path of the dataset')
    parser.add_argument('--dir_path', type=str, default='12DE', help='CWRU：12DE ; 48DE')
    parser.add_argument('--SEU_channel', type=int, default=1, help='SEU channel signal：0-7')
    parser.add_argument('--minute_value', type=int, default=10, help='the last (minute_value) csv file of XJTU datasets each fault class')
    parser.add_argument('--XJTU_channel', type=str, default='X', help='XJTU channel signal:X 、Y 、XY')
    parser.add_argument('--QianPeng_rpm', type=int, default=880, help='the motor rpm of QianPeng_gear dataset_2d--880 or 1470')
    parser.add_argument('--QianPeng_channel', type=int, default=3,help='the signal channel of QianPeng_gear dataset_2d--0-8')
    parser.add_argument('--wind_mode', type=int, default=17,help='Wind_Gearbox mode-- 17 fault class / 2 fault class')

    # ===================================================data preprocessing parameters=============================================================================
    parser.add_argument('--sample_num', type=int, default=200,help='the number of samples')
    parser.add_argument('--train_size', type=float, default=0.6, help='train size')
    parser.add_argument('--sample_length', type=int, default=1024, help='the length of each samples')
    parser.add_argument('--overlap', type=int, default=1024, help='the sampling shift of neibor two samples')  # 滑窗采样偏移量
    parser.add_argument('--norm_type', type=str, default='unnormalization',help='the normalization methods')
    parser.add_argument('--noise', type=int, default=0, help='whether add noise')
    parser.add_argument('--snr', type=int, default=0, help='the snr of noise')
    parser.add_argument('--input_type', type=str, default='FD',help='time domain signal or frequency domain signal as input')

    parser.add_argument('--dimension', type=str, default='1D',help='the shape of input')  # 一维振动信号/灰度图/图片
    parser.add_argument('--image_path', type=str, default=r"E:\FD_image_datasets", help='the dataset path of image')
    parser.add_argument('--image_type', type=str, default="cwt_time_frequency_image", help='the type of self-defined image')
    parser.add_argument('--batch_size', type=int, default=64)

    # ===================================================model parameters=============================================================================
    parser.add_argument('--model_type', type=str, default='CNN_1D', help='the model of training and testing')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='Adam')

    # ===================================================visualization parameters=============================================================================
    parser.add_argument('--visualization', type=bool, default=False, help='whether visualize')
    parser.add_argument('--save_visualization_results', type=bool, default=False, help='whether save visualization results')
    parser.add_argument('--visualization_dirpath', type=str,default=r"C:\Users\Administrator\Desktop\故障诊断开源代码\results\save_visualization",help='the save dirpath of visualization')
    parser.add_argument('--save_data', type=bool, default=False, help='whether save data of each trials')
    parser.add_argument('--save_data_dirpath', type=str,default=r"C:\Users\Administrator\Desktop\故障诊断开源代码\results\save_data",help='the dirpath of saved data')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    if args.save_data == False:
        acc = train_utils(args)
    else:
        save_data(args,trials=10)
