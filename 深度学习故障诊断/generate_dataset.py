'''
Time: 2022/10/8 20:26
Author: Tan Qiyu
Code: https://github.com/Tan-Qiyu/GNN_FD
'''

"""
生成不同数据集的不同图像数据集
"""

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
    Wind_Gearbox   "E:\故障诊断数据集\fault-dataset_2d-collection-main\fault-dataset_2d-collection-main\HS_gear"--------# 风机齿轮箱数据集
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

sample_num：   生成每一种故障类型的样本数（CWRU数据集除外，为每一种故障下的一种工况的样本数，即CWRU每一类故障样本数为sample_num * 工况数 = sample_num * 4）


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

image_type:   生成图像的类型
    cwt_time_frequency_image   # 小波时频图
    GAF                        # 格拉姆角场
    Recurrence_image           # 递归图
    MarkovTransField_image     # 马尔可夫迁移场
    STFT_image                 # 短时傅里叶变换时频图

save_dataset_dirpath：  保存图像数据集文件的路径
'''

import argparse
from 深度学习故障诊断.dataset_2d.CWRU_image_dataset import generate_CWRU_image
from 深度学习故障诊断.dataset_2d.SEU_image_dataset import generate_SEU_image
from 深度学习故障诊断.dataset_2d.XJTU_image_dataset import generate_XJTU_image
from 深度学习故障诊断.dataset_2d.Bearing311_image_dataset import generate_bearing311_image
from 深度学习故障诊断.dataset_2d.QianPeng_image_dataset import generate_QianPeng_image
from 深度学习故障诊断.dataset_2d.JNU_image_dataset import generate_JNU_image
from 深度学习故障诊断.dataset_2d.MFPT_image_dataset import generate_MFPT_image
from 深度学习故障诊断.dataset_2d.WindGearbox_image_dataset import generate_WindGear_image
from 深度学习故障诊断.dataset_2d.Uoc_image_dataset import generate_UoC_image
from 深度学习故障诊断.dataset_2d.DC_image_dataset import generate_DC_image
from 深度学习故障诊断.dataset_2d.SQV_image_dataset import generate_SQV_image


def parse_args():
    parser = argparse.ArgumentParser()
    # basic parameters
    #===================================================dataset_2d parameters=============================================================================
    parser.add_argument('--dataset_name', type=str, default='CWRU', help='the name of the dataset_2d')
    parser.add_argument('--dataset_path', type=str, default=r"E:\故障诊断数据集\凯斯西储大学数据", help='the file path of the dataset_2d')
    parser.add_argument('--dir_path', type=str, default='12DE', help='CWRU：12DE ; 48DE')
    parser.add_argument('--SEU_channel', type=int, default=1, help='SEU channel signal：0-7')
    parser.add_argument('--minute_value', type=int, default=10, help='the last (minute_value) csv file of XJTU datasets each fault class')
    parser.add_argument('--XJTU_channel', type=str, default='X', help='XJTU channel signal:X 、Y 、XY')
    parser.add_argument('--QianPeng_rpm', type=int, default=880, help='the motor rpm of QianPeng_gear dataset_2d--880 or 1470')
    parser.add_argument('--QianPeng_channel', type=int, default=3,help='the signal channel of QianPeng_gear dataset_2d--0-8')
    parser.add_argument('--wind_mode', type=int, default=17,help='Wind_Gearbox mode-- 17 fault class / 2 fault class')

    # ===================================================data preprocessing parameters=============================================================================
    parser.add_argument('--sample_num', type=int, default=200,help='the number of samples')
    parser.add_argument('--sample_length', type=int, default=1024, help='the length of each samples')
    parser.add_argument('--overlap', type=int, default=1024, help='the sampling shift of neibor two samples')
    parser.add_argument('--norm_type', type=str, default='unnormalization',help='unnormalization、Z-score Normalization、Max-Min Normalization、-1 1 Normalization')
    parser.add_argument('--noise', type=int, default=0, help='whether add noise')
    parser.add_argument('--snr', type=int, default=0, help='the snr of noise')
    parser.add_argument('--image_type', type=str, default='cwt_time_frequency_image', help='the type of generated image')
    parser.add_argument('--save_dataset_dirpath', type=str,default=r"E:\FD_image_datasets",help='the dirpath of saved dataset_2d')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    if args.dataset_name == 'CWRU':
        generate_CWRU_image(dataset_path=args.dataset_path, sample_number=args.sample_num, dir_path=args.dir_path, window_size=args.sample_length,
                            overlap=args.overlap, normalization=args.norm_type, noise=args.noise, snr=args.snr,image_type=args.image_type,
                            save_path=args.save_dataset_dirpath)

    elif args.dataset_name == 'SEU':
        generate_SEU_image(dataset_path=args.dataset_path, channel=args.SEU_channel,sample_number=args.sample_num, window_size=args.sample_length,
                            overlap=args.overlap, normalization=args.norm_type, noise=args.noise, snr=args.snr,image_type=args.image_type,save_path=args.save_dataset_dirpath)

    elif args.dataset_name == 'XJTU':
        generate_XJTU_image(dataset_path=args.dataset_path, channel=args.XJTU_channel, minute_value=args.minute_value,sample_number=args.sample_num,window_size=args.sample_length,
                            overlap=args.overlap, normalization=args.norm_type, noise=args.noise, snr=args.snr,image_type=args.image_type, save_path=args.save_dataset_dirpath)

    elif args.dataset_name == 'Bearing_311':
        generate_bearing311_image(dataset_path=args.dataset_path,sample_number=args.sample_num, window_size=args.sample_length,
                            overlap=args.overlap, normalization=args.norm_type, noise=args.noise, snr=args.snr,image_type=args.image_type, save_path=args.save_dataset_dirpath)

    elif args.dataset_name == 'QianPeng_gear':
        generate_QianPeng_image(dataset_path=args.dataset_path, rpm=args.QianPeng_rpm,channel=args.QianPeng_channel,sample_number=args.sample_num,
                                  window_size=args.sample_length,overlap=args.overlap, normalization=args.norm_type, noise=args.noise, snr=args.snr,
                                  image_type=args.image_type, save_path=args.save_dataset_dirpath)

    elif args.dataset_name == 'JNU':
        generate_JNU_image(dataset_path=args.dataset_path, sample_number=args.sample_num,window_size=args.sample_length,overlap=args.overlap,
                           normalization=args.norm_type, noise=args.noise, snr=args.snr,image_type=args.image_type, save_path=args.save_dataset_dirpath)

    elif args.dataset_name == 'MFPT':
        generate_MFPT_image(dataset_path=args.dataset_path, sample_number=args.sample_num,window_size=args.sample_length,overlap=args.overlap,
                            normalization=args.norm_type, noise=args.noise, snr=args.snr,image_type=args.image_type, save_path=args.save_dataset_dirpath)

    elif args.dataset_name == 'Wind_Gearbox':
        generate_WindGear_image(dataset_path=args.dataset_path,fault_mode=args.wind_mode, sample_number=args.sample_num,window_size=args.sample_length,
                                  overlap=args.overlap, normalization=args.norm_type, noise=args.noise, snr=args.snr,image_type=args.image_type, save_path=args.save_dataset_dirpath)

    elif args.dataset_name == 'UoC':
        generate_UoC_image(dataset_path=args.dataset_path, sample_number=args.sample_num,window_size=args.sample_length,overlap=args.overlap,
                           normalization=args.norm_type, noise=args.noise, snr=args.snr,image_type=args.image_type, save_path=args.save_dataset_dirpath)

    elif args.dataset_name == 'DC':
        generate_DC_image(dataset_path=args.dataset_path, sample_number=args.sample_num,window_size=args.sample_length, overlap=args.overlap,normalization=args.norm_type,
                          noise=args.noise, snr=args.snr, image_type=args.image_type,save_path=args.save_dataset_dirpath)

    elif args.dataset_name == 'SQV':
        generate_SQV_image(dataset_path=args.dataset_path,sample_number=args.sample_num, window_size=args.sample_length,overlap=args.overlap,
                           normalization=args.norm_type, noise=args.noise, snr=args.snr,image_type=args.image_type, save_path=args.save_dataset_dirpath)


