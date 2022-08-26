# -*- coding: utf-8 -*-
# @Time    : 2022/7/18 9:43
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : CAM.py
# @Software: PyCharm
# @Source    : https://blog.csdn.net/sinat_37532065/article/details/103362517
# coding: utf-8
import os

from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from classification import resnet50
from torchvision import transforms
global features_grad
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


size = (224, 224)

transform = transforms.Compose(
            [
                transforms.Resize(size=size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )


def draw_CAM(model, img_path, save_path, transform=None, visual_heatmap=False):
    """
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    """
    # 图像加载&预处理
    global features_grad
    img = Image.open(img_path).convert('RGB')
    if transform:
        img = transform(img)
    img = img.unsqueeze(0)
    img = img.cuda()
    # 获取模型输出的feature/score
    model.eval()
    model.features = torch.nn.Sequential(*model.features)
    features = model.features(img)
    output = model.classifier(features)

    # 为了能读取到中间梯度定义的辅助函数
    def extract(g):
        global features_grad
        features_grad = g

    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]

    features.register_hook(extract)
    pred_class.backward()  # 计算梯度

    grads = features_grad  # 获取梯度

    pooled_grads = F.adaptive_avg_pool2d(grads, (1, 1))

    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    features = features[0]
    # 512是最后一层feature的通道数
    for i in range(512 * 4):
        features[i, ...] *= pooled_grads[i, ...]

    # 以下部分同Keras版实现
    heatmap = features.detach().cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()

    img = cv2.imread(img_path)  # 用cv2加载原始图像
    img = cv2.resize(img, size)
    heatmap = cv2.resize(heatmap, size)  # 将热力图的大小调整为与原始图像相同
    heatmap = (heatmap > 0.5).astype(np.uint8)
    if heatmap.max() > 1:
        raise
    cv2.imwrite(save_path[:-4] + '.png', heatmap)  # 将图像保存到硬盘


img_path = r'P:\GAN\CycleGAN-liuye-master\CycleGAN-liuye-master\datasets\crack\train_Positive/'
save_path = r'P:\GAN\CycleGAN-liuye-master\CycleGAN-liuye-master\datasets\crack\train_Positive_Ablation_CAM_mask/'

model = resnet50()
model.cuda()
model.load_state_dict(torch.load(r'P:\GAN\CycleGAN-liuye-master\CycleGAN-liuye-master\Comparison\Model.pth'))

target_layers = [model.layer4[-1]]
T = transform


for file in os.listdir(img_path):
    img_path_ = img_path + file
    save_path_ = save_path + file
    # draw_CAM(model, img_path_, save_path_, transform=T)
    img_pil = Image.open(img_path_)
    input_tensor = T(img_pil)
    input_tensor = input_tensor.reshape((1, 3, 224, 224))
    input_tensor.cuda()
    cam = AblationCAM(model=model, target_layers=target_layers, use_cuda=True)
    targets = None
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam.swapaxes(0, 1)
    grayscale_cam = grayscale_cam.swapaxes(2, 1)
    grayscale_cam = cv2.resize(grayscale_cam, size)
    grayscale_cam = grayscale_cam.reshape(size)
    grayscale_cam = (grayscale_cam > 0.5).astype(np.uint8)
    cv2.imwrite(save_path_[:-4] + '.png', grayscale_cam)





