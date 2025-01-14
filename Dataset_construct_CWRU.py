

import scipy.io as scio
import numpy as np
from signal_1d_transform_and_analysis import *
from signal_2d_transfor_and_save import *
from plot_utils import *
from scipy.io import loadmat
from os_utils import *
from skimage.transform import resize
def from_index_to_filepath(index=[105], root_path=r'E:\datasets\凯斯西储大学数据'):
   
    dir_list = read_obsolute_path_dir(root_path)
    file_path_list = []
    for dir in dir_list:
        file_list = read_obsolete_path_file(dir)  
        file_path_list.extend(file_list)
    result_dict = {os.path.splitext(os.path.basename(path))[0]: path for path in file_path_list} 
    index_file_path_list = []
    for index0 in index:
        file_path = result_dict[index0]
        index_file_path_list.append(file_path)
    return index_file_path_list

def filepath_to_samplelist(filepath, sample_len=2048, overloop=100, index=3, fs=None, label=None, nperseg=128, noverlop=127, threshhold=95):
   
    data = loadmat(filepath) 
    data_key_list = list(data.keys()) 
    desired_key = f'X{os.path.splitext(os.path.basename(filepath))[0]}_DE_time'
    # desired_key = data_key_list[index]  
    fl = data[desired_key].flatten()
    fl = fl.reshape(-1,)
    data = []
    lab = []
    start, end = 0, sample_len
    while end <= fl.shape[0]:
      
        data.append(fl[start:end])
        lab.append(label)
     
        start += sample_len
        end += sample_len
    return data, lab, desired_key

# 数据集信息
CWRU_12K_Drive_End_Bearing_Fauly_Data_dir = r'E:\datasets\凯斯西储大学数据\12k Drive End Bearing Fault Data'  # 文件夹1，12K Drive End Bearing Fault Data
CWRU_48K_Drive_End_Bearing_Fauly_Data_dir = r'E:\datasets\凯斯西储大学数据\48k Drive End Bearing Fault Data'  # 文件夹2，48K Drive End Bearing Fault Data
CWRU_12K_Fan_End_Bearing_Fauly_Data_dir = r'E:\datasets\凯斯西储大学数据\12k Fan End Bearing Fault Data'  # 文件夹3，12K Fan End Bearing Fault Data
CWRU_48k_normal_data_dir = r'E:\datasets\凯斯西储大学数据\Normal Baseline Data'
data_path = [CWRU_12K_Drive_End_Bearing_Fauly_Data_dir, CWRU_48K_Drive_End_Bearing_Fauly_Data_dir, CWRU_12K_Fan_End_Bearing_Fauly_Data_dir, CWRU_48k_normal_data_dir]

# 获取待分析数据文件的路径
root_path = r'E:\datasets\凯斯西储大学数据'
# file_index = ['105', '106', '107', '108', '169', '170', '171', '172', '209', '210', '211', '212', '118', '119', '120',
#               '121', '223', '130', '131', '132', '133', '197', '198', '199', '200', '234', '235', '236', '237',  '144',
#               '145', '146','147', '246', '247', '248', '249', '156', '158', '159', '160', '258', '259', '260', '261']
# file_index = ['105', '106', '169', '170', '209', '210', '118', '119', '185', '186', '222', '223',
#               '130', '131', '197', '198', '234', '235', '144', '145', '246', '247', '156', '158', '258', '259']  # 驱动端轴承故障，DE,12K
# file_index = ['279', '274', '275', '270', '271', '282', '283', '286', '287', '294', '295', '313',
#               '315', '298', '299', '309', '310', '316', '302', '305']  # 风扇端轴承故障，FE,12K
file_index = ['175', '213', '214', '122', '189', '226', '135', '136', '201', '202', '238',
              '239', '148', '149', '250', '251', '161', '162', '262', '263']  # 驱动端端轴承故障，DE,48K'109', '110', '174'
fs = 48e3
sample_len = 4096
file_path = from_index_to_filepath(file_index, root_path)
for index_file_path in file_path:
    sample_list, _, key = filepath_to_samplelist(index_file_path, sample_len, overloop=int(sample_len/2), index=5)  # 获取数据文件形成的样本列表3 4 6 7
   
    for i in range(len(sample_list)):
        sample = sample_list[i]
      
        t = np.linspace(0, sample_len/fs, sample_len)
        Wavelet_scatter_analysis = wavelet_scaterring_analysis(J=10, Q=(11, 1)) 
        fig = Wavelet_scatter_analysis.no_lowpass_scattering_result_visualisation(sample, fs, lama1=6)
        save_dir = r'E:\datasets\SWT_CAM\all_image\bearing_fault2'
        # file_name = f'{os.path.splitext(os.path.basename(index_file_path))[0]}-{key}-{i}.png'
        file_name = f'{key}-{i}.png'
        save_path = os.path.join(save_dir, file_name)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)

