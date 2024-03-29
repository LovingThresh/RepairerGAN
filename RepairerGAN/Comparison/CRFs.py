# -*- coding: utf-8 -*-
# @Time    : 2022/7/20 9:33
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : CRFs.py
# @Software: PyCharm

import cv2
import numpy as np
import pydensecrf.densecrf as dcrf
from cv2 import imread, imwrite, resize

from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

"""
original_image_path  原始图像路径
predicted_image_path  之前用自己的模型预测的图像路径
CRF_image_path  即将进行CRF后处理得到的结果图像保存路径
"""


def CRFs(original_image_path, predicted_image_path, CRF_image_path):
    img = imread(original_image_path)
    img = resize(img, (224, 224))
    # 将predicted_image的RGB颜色转换为uint32颜色 0xbbggrr
    anno_rgb = imread(predicted_image_path, cv2.IMREAD_GRAYSCALE).astype(np.uint32)
    anno_rgb = np.repeat(anno_rgb.reshape(224, 224, 1), 3, axis=-1)
    anno_lbl = anno_rgb[:, :, 0] + (anno_rgb[:, :, 1] << 8) + (anno_rgb[:, :, 2] << 16)

    # 将uint32颜色转换为1,2,...
    colors, labels = np.unique(anno_lbl, return_inverse=True)

    # 如果你的predicted_image里的黑色（0值）不是待分类类别，表示不确定区域，即将分为其他类别
    # 那么就取消注释以下代码
    HAS_UNK = 0 in colors
    if HAS_UNK:
        colors = colors[1:]

    # 创建从predicted_image到32位整数颜色的映射。
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # 计算predicted_image中的类数。
    n_labels = len(set(labels.flat))
    # n_labels = len(set(labels.flat)) - int(HAS_UNK) ##如果有不确定区域，用这一行代码替换上一行

    ###########################
    #        设置CRF模型       #
    ###########################
    use_2d = False
    # use_2d = True

    # 不是很清楚什么情况用2D
    # 作者说“对于图像，使用此库的最简单方法是使用DenseCRF2D类”
    # 作者还说“DenseCRF类可用于通用（非二维）密集CRF”
    # 但是根据我的测试结果一般情况用DenseCRF比较对

    if use_2d:
        # 使用densecrf2d类
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

        # 得到一元势（负对数概率）
        U = unary_from_labels(labels, n_labels, gt_prob=0.2, zero_unsure=None)
        # U = unary_from_labels(labels, n_labels, gt_prob=0.2, zero_unsure=HAS_UNK)## 如果有不确定区域，用这一行代码替换上一行
        d.setUnaryEnergy(U)

        # 增加了与颜色无关的术语，功能只是位置而已
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # 增加了颜色相关术语，即特征是(x,y,r,g,b)
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img, compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
    else:
        # 使用densecrf类
        d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

        # 得到一元势（负对数概率）
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=None)
        # U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)## 如果有不确定区域，用这一行代码替换上一行
        d.setUnaryEnergy(U)

        # 这将创建与颜色无关的功能，然后将它们添加到CRF中
        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        # 这将创建与颜色相关的功能，然后将它们添加到CRF中
        feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    # 进行5次推理
    Q = d.inference(5)

    # 找出每个像素最可能的类
    MAP = np.argmax(Q, axis=0)

    # 将predicted_image转换回相应的颜色并保存图像
    # MAP = colorize[MAP, :]
    MAP = ((((MAP * 2) / 255) > 0.5) * 255).astype(np.uint8)
    imwrite(CRF_image_path, MAP.reshape(224, 224))
    print("CRF图像保存在", CRF_image_path, "!")


# a = ['train_Positive_Grad_CAMPlusPlus_mask', 'train_Positive_Score_CAM_mask', 'train_Positive_Ablation_CAM_mask']
# b = ['train_Positive_Grad_CAMPlusPlus_CRFs_mask', 'train_Positive_Score_CAM_CRFs_mask', 'train_Positive_Ablation_CAM_CRFs_mask']
#
# o_img_path = '/root/auto-tmp/CycleGAN-liuye-master/CycleGAN-liuye-master/datasets/crack/train_Positive/'
# p_img_path = '/root/auto-tmp/CycleGAN-liuye-master/CycleGAN-liuye-master/datasets/crack/ann_dir/train_Positive_Grad_CAM_mask/'
# save_path =  '/root/auto-tmp/CycleGAN-liuye-master/CycleGAN-liuye-master/datasets/crack/ann_dir/train_Positive_Grad_CAM_CRFs_mask/'
# os.makedirs(save_path)
#
# for file in os.listdir(o_img_path):
#     img_path_ = o_img_path + file
#     p_img_path_ = p_img_path + file[:-4] + '.png'
#     save_path_ = save_path + file[:-4] + '.png'
#     CRFs(img_path_, p_img_path_, save_path_)


def CRFs_array(original_image, predicted_image):
    original_image = np.asarray(original_image)
    predicted_image = np.asarray(predicted_image)
    img = cv2.resize(original_image, (448, 448))
    # 将predicted_image的RGB颜色转换为uint32颜色 0xbbggrr
    anno_rgb = predicted_image.astype(np.uint32)
    anno_rgb = np.repeat(anno_rgb.reshape(448, 448, 1), 3, axis=-1)
    anno_lbl = anno_rgb[:, :, 0] + (anno_rgb[:, :, 1] << 8) + (anno_rgb[:, :, 2] << 16)

    # 将uint32颜色转换为1,2,...
    colors, labels = np.unique(anno_lbl, return_inverse=True)

    # 如果你的predicted_image里的黑色（0值）不是待分类类别，表示不确定区域，即将分为其他类别
    # 那么就取消注释以下代码
    HAS_UNK = 0 in colors
    if HAS_UNK:
        colors = colors[1:]

    # 创建从predicted_image到32位整数颜色的映射。
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # 计算predicted_image中的类数。
    n_labels = len(set(labels.flat))
    # n_labels = len(set(labels.flat)) - int(HAS_UNK) ##如果有不确定区域，用这一行代码替换上一行

    ###########################
    ###     设置CRF模型     ###
    ###########################
    use_2d = False
    # use_2d = True
    ###########################################################
    ##不是很清楚什么情况用2D
    ##作者说“对于图像，使用此库的最简单方法是使用DenseCRF2D类”
    ##作者还说“DenseCRF类可用于通用（非二维）密集CRF”
    ##但是根据我的测试结果一般情况用DenseCRF比较对
    #########################################################33
    if use_2d:
        # 使用densecrf2d类
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

        # 得到一元势（负对数概率）
        U = unary_from_labels(labels, n_labels, gt_prob=0.2, zero_unsure=None)
        # U = unary_from_labels(labels, n_labels, gt_prob=0.2, zero_unsure=HAS_UNK)## 如果有不确定区域，用这一行代码替换上一行
        d.setUnaryEnergy(U)

        # 增加了与颜色无关的术语，功能只是位置而已
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # 增加了颜色相关术语，即特征是(x,y,r,g,b)
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img, compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
    else:
        # 使用densecrf类
        d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

        # 得到一元势（负对数概率）
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=None)
        # U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)## 如果有不确定区域，用这一行代码替换上一行
        d.setUnaryEnergy(U)

        # 这将创建与颜色无关的功能，然后将它们添加到CRF中
        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        # 这将创建与颜色相关的功能，然后将它们添加到CRF中
        feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    ####################################
    ###         做推理和计算         ###
    ####################################

    # 进行5次推理
    Q = d.inference(5)

    # 找出每个像素最可能的类
    MAP = np.argmax(Q, axis=0)

    # 将predicted_image转换回相应的颜色并保存图像
    # MAP = colorize[MAP, :]
    MAP = MAP.reshape(448, 448)
    return MAP


image = np.ones((448, 448, 3))

image = cv2.imread(r'M:\CycleGAN(WSSS)\File\FeatureImage\iter-000020000.jpg')
crop_image = image[:448, :448]
cv2.imwrite(r'M:\CycleGAN(WSSS)\File\FeatureImage\image_5.png', crop_image)
pre_image = image[:448, 448 * 2: 448 * 3]
cv2.imwrite(r'M:\CycleGAN(WSSS)\File\FeatureImage\image_5_pre.png', pre_image)

image = cv2.imread(r'C:\Users\liuye\Desktop\image\outputs\attachments\image_5_1.png')
image = cv2.imwrite(r'C:\Users\liuye\Desktop\image\outputs\attachments\image_5_1_label.png',
                    (image > 10).astype(np.uint8) * 255)
