# -*- coding: utf-8 -*-
# @Time    : 2022/7/18 9:46
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : classification.py
# @Software: PyCharm
# @Source  : https://blog.csdn.net/weixin_41645749/article/details/115751000
import os
import math
import random
import time
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchsummary
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)  # 数据归一化处理，使其均值为0，方差为1，可有效避免梯度消失

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)  # 数据归一化处理，使其均值为0，方差为1，可有效避免梯度消失

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)  # 数据归一化处理，使其均值为0，方差为1，可有效避免梯度消失

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d((7, 7))
        self.classifier = nn.ModuleList()
        self.classifier.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier.append(nn.Flatten())
        self.classifier.append(nn.Linear(512 * block.expansion, self.num_classes))
        # self.classifier.append(nn.Flatten())
        # self.classifier.append(nn.Linear(512 * block.expansion, self.num_classes))
        self.classifier = nn.Sequential(*self.classifier)

        self.features = nn.ModuleList()
        self.features.append(self.conv1)
        self.features.append(self.relu)
        self.features.append(self.bn1)
        self.features.append(self.maxpool)
        self.features.append(self.layer1)
        self.features.append(self.layer2)
        self.features.append(self.layer3)
        self.features.append(self.layer4)
        # self.features.append(self.avgpool)

        # 遍历所有模块，然后对其中参数进行初始化
        for m in self.modules():  # self.modules()采用深度优先遍历的方式，存储了net的所有模块
            if isinstance(m, nn.Conv2d):  # 判断是不是卷积
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))  # 对权值参数初始化
            elif isinstance(m, nn.BatchNorm2d):  # 判断是不是数据归一化
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 当是 3 4 6 3 的第一层时，由于跨层要做一个步长为 2 的卷积 size会变成二分之一，所以此处跳连接 x 必须也是相同维度
            downsample = nn.Sequential(  # 对跳连接 x 做 1x1 步长为 2 的卷积，保证跳连接的时候 size 一致
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))  # 将跨区的第一个要做步长为 2 的卷积添加到layer里面
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):  # 将除去第一个的剩下的 block 添加到layer里面
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.classifier(x)
        x = x.view(-1, self.num_classes)
        return x


def resnet50(pretrained=False):
    """Constructs a ResNet-50 model.
      Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=2)
    # if pretrained:    # 加载已经生成的模型
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

    return model


# 初始化根目录
train_path = r'P:/GAN/CycleGAN-liuye-master/CycleGAN-liuye-master/datasets/crack/'


# 定义读取文件的格式
# 数据集
class MyDataSet(Dataset):
    def __init__(self, data_path: str, key='train'):
        super(MyDataSet, self).__init__()
        self.data_path = data_path
        self.key = key
        self.transform = transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.path_list_Positive = os.listdir(data_path + self.key + '_Positive')
        self.path_list_Negative = os.listdir(data_path + self.key + '_Negative')

    def __getitem__(self, idx: int):

        label = random.choice([0, 1])
        if label == 1:
            img_path = self.path_list_Positive[idx]
            self.data_path_ = self.data_path + self.key + '_Positive'
            label = [1, 0]
        else:
            img_path = self.path_list_Negative[idx]
            self.data_path_ = self.data_path + self.key + '_Negative'
            label = [0, 1]

        label = torch.as_tensor(label, dtype=torch.float32)
        img_path = os.path.join(self.data_path_, img_path)
        img = Image.open(img_path)
        img = self.transform(img)
        return img, label

    def __len__(self) -> int:
        return len(self.path_list_Positive)


train_ds = MyDataSet(train_path, key='train')
val_ds = MyDataSet(train_path, key='val')
test_ds = MyDataSet(train_path, key='test')

# 数据加载
new_train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True, num_workers=0)
new_val_loader = DataLoader(val_ds, batch_size=32, shuffle=True, pin_memory=True, num_workers=0)
new_test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, pin_memory=True, num_workers=0)

LR = 0.0005  # 设置学习率
EPOCH_NUM = 100  # 训练轮次


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


model = resnet50()
torchsummary.summary(model, (3, 224, 224), 1, device='cpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

train_data = new_train_loader
test_data = new_test_loader

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


def train(epoch, loss_list):
    running_loss = 0.0
    for batch_idx, data in enumerate(new_train_loader, 0):
        inputs, target = data[0], data[1]
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        running_loss += loss.item()
        if batch_idx % 100 == 99:
            print(f'[{time_since(start)}] Epoch {epoch}', end='')
            print('[%d, %5d] loss:%.3f' % (epoch + 1, batch_idx + 1, running_loss / 100))
            running_loss = 0.0

    return loss_list


def val():
    correct = 0
    total = 0
    with torch.no_grad():
        for _, data in enumerate(new_test_loader, 0):
            inputs, target = data[0], data[1]
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            prediction = torch.argmax(outputs, dim=1)
            target = torch.argmax(target, dim=1)
            total += target.size(0)
            correct += (prediction == target).sum().item()
        print('Accuracy on test set: (%d/%d)%d %%' % (correct, total, 100 * correct / total))
        if 100 * correct / total > 98.5:
            torch.save(model.state_dict(), 'Model.pth')
        with open("test.txt", "a") as f:
            f.write('Accuracy on test set: (%d/%d)%d %% \n' % (correct, total, 100 * correct / total))


if __name__ == '__main__':
    start = time.time()

    with open("test.txt", "a") as f:
        f.write('Start write!!! \n')

    loss_list = []
    for epoch in range(EPOCH_NUM):
        train(epoch, loss_list)
        val()

    x_ori = []
    for i in range(len(loss_list)):
        x_ori.append(i)
    plt.title("Graph")
    plt.plot(x_ori, loss_list)
    plt.ylabel("Y")
    plt.xlabel("X")
    plt.show()
