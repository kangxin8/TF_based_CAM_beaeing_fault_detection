'''
在此定义一个对象，该对象主要包含训练前的准备和训练过程中的记录
训练前的准备：数据集准备、模型确定、优化器、优化器参数设置、损失函数等
训练过程中：每一代的训练和验证的结果，训练好的模型的存储
训练前的准备：
1、根据设备条件选择训练的方式：CPU,GPU
2、dataloder的构建
3、模型结构的确定
4、优化器的选择
5、损失函数的选择
训练过程具体实现的内容：
1、记录每一代epoch训练的信息：当前代次/总代次、每一代的学习率、指定epoch的损失、预测准确率、每一代训练的时间
2、每一个epoch的标准流程：the train loop \ the validation loop
3、在console打印本代次的训练信息（损失、准确率）
4、存储预测准确率高的模型参数
训练过程中：
训练过程中可以打印以下的信息：
1、在指定代次范围内的预测正确率和损失（由print_step', type=int, default=10参数决定），表示每10代输出一次
'''
import argparse
#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import os
import time
import warnings
import torch
from torch import nn
from torch import optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import vgg16, VGG16_Weights
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, c=None):
        super().__init__()  # super是内置函数，调用该函数以确保父类Module被正确的初始化
        # img = images
        self.fc = nn.Linear(512, c)

    def forward(self, x):
        batch_size = x.size(0)
        # x = x.view(512, 7 * 7).mean(1).view(1, -1)
        x = x.view(batch_size, 512, 7 * 7).mean(2)
        x = self.fc(x)
        return x
def get_model_input_sample(Dataset, index):
    '''
    该函数用于利用Datasets对象，返回指定索引的样本，并将样本以深度学习模型可接受的格式返回
    :param Dataset: Datasets对象
    :param index: 指定索引
    :return:
    返回满足模型输入要求的样本形式和标签
    '''
    sample = Dataset[index]

    # 处理图像数据
    if isinstance(sample, torch.Tensor):
        return sample.unsqueeze(0)

    # 处理图像和标签的情况
    elif isinstance(sample, tuple) and len(sample) == 2:
        image, label = sample
        return image.unsqueeze(0), torch.tensor([label])

def accuracy_model_test(model, data_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
class train_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

    def setup(self):

        args = self.args

        # 确定使用GPu还是CPU进行训练
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        # 构建datasets与dataloader
        data_loaders = None
        img_transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])  # transforms
        train_data_dir = datasets.ImageFolder(root=r"E:\datasets\SWT_CAM\all_image", transform=img_transform,
                                              is_valid_file=None, target_transform=None)  # 创建供dataloader使用的datasets对象
        test_data_dir = datasets.ImageFolder(root=r"E:\datasets\SWT_CAM\val", transform=img_transform,
                                             is_valid_file=None, target_transform=None)
        # region datasets对象了解
        # image, label = train_data_dir[0]  # 获取第一个样本和标签
        # classes_name_list = train_data_dir.classes  # 返回一个列表，列表的每个元素代表某个类别的名称（子文件夹的名称）
        # name_to_idx = train_data_dir.class_to_idx  # 返回一个字典，键为类别名称，值为该类对应的标签
        # samples = train_data_dir.samples  # 返回一个由元组形成的列表，每个元组包含样本的路径和标签
        # endregion
        batch_size = 32  # 批大小
        train_loader = torch.utils.data.DataLoader(train_data_dir, batch_size=batch_size,
                                                   shuffle=True)  # 创建供模型使用的dataloader
        test_loader = torch.utils.data.DataLoader(test_data_dir, batch_size=batch_size, shuffle=False)
        # image = get_model_input_sample(train_data_dir, 0)  # 利用datasets获取样本
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # class_names = VGG16_Weights.IMAGENET1K_V1.meta["categories"]  # 获取原始训练信息
        # region 模型对象的了解
        # print(vgg16)
        # vgg16.train()  # 调整为训练模式
        # vgg16.eval()  # 调整为测试模式（改变如dropout层的行为）
        # vgg16.to(device='cpu')  # 将模型移动到指定的计算平台
        # model_parameters = vgg16.state_dict()  # 返回一个字典包含模型的所有参数
        # parameters = vgg16.parameters()  # 返回一个迭代器，包含模型的所有参数
        # named_parameters = vgg16.named_parameters()  # 返回一个迭代器，包含模型的所有参数名称和值
        # vgg16.load_state_dict(parameters)  # 从提供的state_dict加载模型参数
        # features = vgg16.features  # 这是一个 torch.nn.Sequential 对象，包含VGG16的特征提取部分（卷积层和池化层）
        # avgpool = vgg16.avgpool  # 平均池化层，用于在特征提取和分类器之间调整特征图大小。
        # classifier = vgg16.classifier  # 这是一个 torch.nn.Sequential 对象，包含VGG16的分类器部分（全连接层）
        # training_state = vgg16.training  # 布尔值，表示模型是否处于训练模式。
        # result = vgg16(image[0])
        # probabilities = torch.nn.functional.softmax(result, dim=1)  # 应用softmax获取概率分布
        # proba_max, predicted_idx = torch.max(probabilities, 1)  # 获取最大概率及其索引
        # predicted_idx = predicted_idx.item()
        # predicted_class = class_names[predicted_idx]
        # endregion
        # 修改模型结构
        # layer_list = list(model.children())  # 只返回模型定义的顶层模块，而不返回子模块
        mod = nn.Sequential(*list(model.children())[:-1])  # 返回除去分类层之外的其他层，*表示解包操作，将列表分割为单独的元素作为容器的参数
        add_net = Net(c=2)
        new_model = nn.Sequential(mod, add_net)
        # model_parameters = model.state_dict()
        # 确定需要训练的模型参数
        # device = 'cpu'
        trainable_parameters = []
        for name, p in new_model.named_parameters():
            if "fc" in name:
                trainable_parameters.append(p)
        # 创建优化器
        optimizer = torch.optim.SGD(params=trainable_parameters, lr=0.1, momentum=1e-5)  # 这些参数都是有标记和属性的，不是普通的张量数据
        # 定义损失函数
        criterion = nn.CrossEntropyLoss()

        # 训练模型
        num_epochs = 100
        # total_step = len(train_loader)
        step = 0  # 总的批次记录器
        # best_acc = 0.0  # 记录所有epoch训练完后的准确率
        batch_count = 0  # 记录一个epoch中的第几个batch
        batch_loss = 0.0  # 记录一个batch的损失
        batch_acc = 0  # 在一个代次训练过程中，不同阶段（不同批次范围内）的预测准确率
        step_start = time.time()
        trained_model_save_dir = self.save_dir
        for epoch in range(num_epochs):
            epoch_acc = 0  # 每一批次的准确率
            epoch_loss = 0.0  # 每一批次的损失
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, num_epochs - 1) + '-'*5)  # 记录当前代次信息
            epoch_start = time.time()
            for i, (images, labels) in enumerate(train_loader):
                new_model.train()
                # 前向传播
                outputs = new_model(images)
                loss = criterion(outputs, labels)
                # loss_list.append(loss.item())
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # 返回训练过程的信息
                total = labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                bach_correct = (predicted == labels).sum().item()
                batch_acc += bach_correct  # 同一代次，不同批次范围内的正确率
                batch_count += images.size(0)  #  记录当前批次范围内累计样本个数
                batch_loss += loss.item() * images.size(0)
                epoch_loss += loss.item() * images.size(0)  # 更新损失
                epoch_acc += bach_correct #
                # 输出在同一代次指定批次范围内的训练信息
                if step % args.print_step == 0:
                    batch_loss = batch_loss / batch_count  # 训练过程不同时期，指定代次范围内的损失
                    batch_acc = batch_acc / batch_count  # 训练过程不同时期，指定代次范围内正确率
                    temp_time = time.time()
                    train_time = temp_time - step_start  # 训练一个批次所需的时间
                    step_start = temp_time
                    batch_time = train_time / args.print_step if step != 0 else train_time
                    sample_per_sec = 1.0 * batch_count / train_time
                    logging.info('Epoch: {} [{}/{}], Train Loss: {:.4f} Train Acc: {:.4f},'
                                 '{:.1f} examples/sec {:.2f} sec/batch'.format(
                        epoch, i * images.size(0), len(train_loader.dataset),
                        batch_loss, batch_acc, sample_per_sec, batch_time
                    ))  # 打印当前批次、当前批次遍历的样本数/总样本数，当前代次的损失值，当前指定代次范围内的预测正确的精度，每秒处理的样本数，处理一批次样本所需要的时间
                    batch_acc = 0
                    batch_loss = 0.0
                    batch_count = 0
                step += 1
                # Print the train and val information via each epoch
            # 输出每一代次的训练信息
            epoch_loss = epoch_loss / len(train_loader.dataset)
            epoch_acc = epoch_acc / len(train_loader.dataset)
            logging.info('Epoch: {} Loss: {:.4f} Acc: {:.4f}, Cost {:.4f} sec'.format(
                epoch, epoch_loss, epoch_acc, time.time() - epoch_start))


            test_acc = accuracy_model_test(new_model, test_loader, device=self.device)  # 计算测试集精度
            logging.info("epoch:{}test_acc {:.4f}".format(epoch, test_acc))
            # 保存模型
            if (epoch + 1) % 5 == 0:
                model_state_dict = new_model.state_dict()
                torch.save(model_state_dict,
                           os.path.join(trained_model_save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, test_acc)))

























