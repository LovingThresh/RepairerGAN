# -*- coding: utf-8 -*-
# @Time    : 2022/4/24 9:44
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : Data_choose.py
# @Software: PyCharm

# BridgeData
# 1、将BridgeData复制到指定的文件夹
# 2、修改文件名，确保一一对应
# 3、使用Teacher模型进行预测Mask
# 4、使用修复算法进行还原
# TODO 5、存在一小部分的图像需要增加标注，大概是40张左右

import os
import cv2
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

path = r'L:\crack segmentation in UAV images\Dataset/'
file_list = os.listdir(path)
random.shuffle(file_list)

train_list = file_list[:int(len(file_list) * 0.8)]
val_list = file_list[int(len(file_list) * 0.8): int(len(file_list) * 0.9)]
test_list = file_list[int(len(file_list) * 0.9):]

train_path = r'L:\crack segmentation in UAV images\Dataset\train\mask'
val_path = r'L:\crack segmentation in UAV images\Dataset\val\mask'
test_path = r'L:\crack segmentation in UAV images\Dataset\test\mask'

for file_path, file_list in zip((train_path, val_path, test_path), (train_list, val_list, test_list)):
    for file in file_list:
        file = file[:-4] + '.png'
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (512, 512))
        cv2.imwrite(os.path.join(file_path, file), img)


file_list = test_list
with open(os.path.join(path, 'test.txt'), 'w') as f:
    for file in file_list[:-1]:
        a = file + ',' + file[:-4] + '.png' + '\n'
        f.write(a)
    file = file_list[-1]
    f.write(file + ',' + file[:-4] + '.png')

# 其实也不一定需要修改文件名
# 第二步完成

train_label_path = r'P:\GAN\CycleGAN-liuye-master\CycleGAN-liuye-master\datasets\bridge_crack\train_Positive_mask'
val_label_path = r'P:\GAN\CycleGAN-liuye-master\CycleGAN-liuye-master\datasets\bridge_crack\val_Positive_mask'
test_label_path = r'P:\GAN\CycleGAN-liuye-master\CycleGAN-liuye-master\datasets\bridge_crack\test_Positive_mask'

model = keras.models.load_model(r'E:\output\2022-03-06-23-18-41.346776_SOTA4\checkpoint\ep025-val_loss2001.124')
for file_path, out_path in zip((train_path, val_path, test_path), (train_label_path, val_label_path, test_label_path)):
    files = os.listdir(file_path)
    for i in files:
        image = cv2.imread(os.path.join(file_path, i))
        image = cv2.resize(image, (448, 448))
        image = image / 255.
        image = image * 2 - 1

        predict = model.predict(image.reshape(1, 448, 448, 3))
        predict = cv2.resize(predict[-1].reshape(448, 448, 2), (512, 512))
        predict = (predict[:, :, 0] > 0.2).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        predict = cv2.erode(predict, kernel, iterations=1)
        predict = cv2.dilate(predict, kernel, iterations=1)
        cv2.imwrite(os.path.join(out_path, i), predict)

# 然后是根据mask进行修复，这一部分使用sample-imageinpainting-HiFill-master就能实现（华为）
train_Restoration_path = r'P:\GAN\CycleGAN-liuye-master\CycleGAN-liuye-master\datasets\bridge_crack\train_Negative'
val_Restoration_path = r'P:\GAN\CycleGAN-liuye-master\CycleGAN-liuye-master\datasets\bridge_crack\val_Negative'
test_Restoration_path = r'P:\GAN\CycleGAN-liuye-master\CycleGAN-liuye-master\datasets\bridge_crack\test_Negative'


# MCFF_Crack
# 1、将MCFF_Data复制到指定的文件夹
# 2、预测Mask应该为膨胀(3, 3),iteration=3,再复制到指定位置
# 3、使用修复算法进行还原

raw_train_path = r'L:\ALASegmentationNets_v2\Data\Stage_4\train\mask'
raw_val_path = r'L:\ALASegmentationNets_v2\Data\Stage_4\val\mask'
raw_test_path = r'L:\ALASegmentationNets_v2\Data\Stage_4\test\mask'

train_label_path = r'P:\GAN\CycleGAN-liuye-master\CycleGAN-liuye-master\datasets\MCFF_crack\train_Positive_mask'
val_label_path = r'P:\GAN\CycleGAN-liuye-master\CycleGAN-liuye-master\datasets\MCFF_crack\val_Positive_mask'
test_label_path = r'P:\GAN\CycleGAN-liuye-master\CycleGAN-liuye-master\datasets\MCFF_crack\test_Positive_mask'

for file_path, out_path in zip((raw_train_path, raw_val_path, raw_test_path), (train_label_path, val_label_path, test_label_path)):
    files = os.listdir(file_path)
    for i in files:
        image = cv2.imread(os.path.join(file_path, i))
        image = cv2.dilate(image, (3, 3), iterations=3)
        cv2.imwrite(os.path.join(out_path, i), image)






