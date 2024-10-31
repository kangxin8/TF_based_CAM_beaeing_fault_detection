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
        super().__init__() 
        # img = images
        self.fc = nn.Linear(512, c)

    def forward(self, x):
        batch_size = x.size(0)
        # x = x.view(512, 7 * 7).mean(1).view(1, -1)
        x = x.view(batch_size, 512, 7 * 7).mean(2)
        x = self.fc(x)
        return x
def get_model_input_sample(Dataset, index):
   
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
                                              is_valid_file=None, target_transform=None) 
        test_data_dir = datasets.ImageFolder(root=r"E:\datasets\SWT_CAM\val", transform=img_transform,
                                             is_valid_file=None, target_transform=None)
       
        batch_size = 32  
        train_loader = torch.utils.data.DataLoader(train_data_dir, batch_size=batch_size,
                                                   shuffle=True) 
        test_loader = torch.utils.data.DataLoader(test_data_dir, batch_size=batch_size, shuffle=False)
        # image = get_model_input_sample(train_data_dir, 0)  
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # class_names = VGG16_Weights.IMAGENET1K_V1.meta["categories"] 
        
        # layer_list = list(model.children()) 
        mod = nn.Sequential(*list(model.children())[:-1])  
        add_net = Net(c=2)
        new_model = nn.Sequential(mod, add_net)
        # model_parameters = model.state_dict()
 
        # device = 'cpu'
        trainable_parameters = []
        for name, p in new_model.named_parameters():
            if "fc" in name:
                trainable_parameters.append(p)

        optimizer = torch.optim.SGD(params=trainable_parameters, lr=0.1, momentum=1e-5) 
      
        criterion = nn.CrossEntropyLoss()

        num_epochs = 100
        # total_step = len(train_loader)
        step = 0 
        # best_acc = 0.0  
        batch_count = 0  
        batch_loss = 0.0 
        batch_acc = 0  
        step_start = time.time()
        trained_model_save_dir = self.save_dir
        for epoch in range(num_epochs):
            epoch_acc = 0 
            epoch_loss = 0.0  
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, num_epochs - 1) + '-'*5)  
            epoch_start = time.time()
            for i, (images, labels) in enumerate(train_loader):
                new_model.train()
              
                outputs = new_model(images)
                loss = criterion(outputs, labels)
                # loss_list.append(loss.item())
           
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
          
                total = labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                bach_correct = (predicted == labels).sum().item()
                batch_acc += bach_correct 
                batch_count += images.size(0) 
                batch_loss += loss.item() * images.size(0)
                epoch_loss += loss.item() * images.size(0) 
                epoch_acc += bach_correct #
              
                if step % args.print_step == 0:
                    batch_loss = batch_loss / batch_count 
                    batch_acc = batch_acc / batch_count
                    temp_time = time.time()
                    train_time = temp_time - step_start 
                    step_start = temp_time
                    batch_time = train_time / args.print_step if step != 0 else train_time
                    sample_per_sec = 1.0 * batch_count / train_time
                    logging.info('Epoch: {} [{}/{}], Train Loss: {:.4f} Train Acc: {:.4f},'
                                 '{:.1f} examples/sec {:.2f} sec/batch'.format(
                        epoch, i * images.size(0), len(train_loader.dataset),
                        batch_loss, batch_acc, sample_per_sec, batch_time
                    )) 
                    batch_acc = 0
                    batch_loss = 0.0
                    batch_count = 0
                step += 1
                # Print the train and val information via each epoch
           
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

























