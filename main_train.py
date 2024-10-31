'''
该脚本实现对改进的vgg模型进行训练，目的是验证训练好模型的可解释性。使用的训练集是轴承故障与健康的二分类数据集
'''

import argparse
import logging
import os
from datetime import datetime  #
from train_object import train_utils
def parse_args():
    '''
    不需要输入参数，返回一个参数解析器对象
    '''
    parser = argparse.ArgumentParser(description='Train')  # 创建参数解析器对象
    # 解析器参数的定义
    # basic parameters
    parser.add_argument('--model_name', type=str, default='pretrained_vgg_CAM', help='the name of the model')
    parser.add_argument('--data_name', type=str, default='CWRU2classes', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default= "E:\datasets\stft_224224_datasets\Train", help='the directory of the data')
    parser.add_argument('--normlizetype', type=str, choices=['0-1', '1-1', 'mean-std'], default='0-1', help='data normalization methods')
    parser.add_argument('--processing_type', type=str, choices=['R_A', 'R_NA', 'O_A'], default='R_A',
                        help='R_A: random split with data augmentation, R_NA: random split without data augmentation, O_A: order split with data augmentation')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    parser.add_argument("--pretrained", type=bool, default=True, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=32, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')

    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='fix', help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='9', help='the learning rate decay for step and stepLR')


    # save, load and display information
    parser.add_argument('--max_epoch', type=int, default=20, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=10, help='the interval of log training information')
    args = parser.parse_args()
    return args


def setlogger(path):
    '''
    该函数的作用是设置全局的日志记录器配置，在全局使用日志记录器时遵循该设置，不返回任何值
    input:记录日志的文件的路径
    '''
    logger = logging.getLogger()  # 创建记录器，提供接口
    logger.setLevel(logging.INFO)  # 决定日志记录的级别
    logFormatter = logging.Formatter("%(asctime)s %(message)s", "%m-%d %H:%M:%S")  # 设置日志内容的格式,以及时间的格式
    # 设置日志处理器
    fileHandler = logging.FileHandler(path)  # 文件记录类型处理器
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()  # 屏幕输出类型记录器
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)


if __name__ == '__main__':
    # 示例和测试
    args = parse_args()  # 实例化参数解析器
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()  # 调用参数
    sub_dir = args.model_name + '_' + args.data_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')  # 指定训练好的模型参数的存储路径
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    setlogger(os.path.join(save_dir, 'training.log'))  # 设置日志记录器

    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))  # 将初始化各个参数写入日志中

    trainer = train_utils(args, save_dir)
    # 完成数据经集、网络模型、损失函数、优化器的准备工作
    trainer.setup()
