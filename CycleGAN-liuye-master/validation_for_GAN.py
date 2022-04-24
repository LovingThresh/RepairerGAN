# -*- coding: utf-8 -*-
# @Time    : 2022/4/21 22:39
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : validation_for_GAN.py
# @Software: PyCharm

# 这个的代码的作用就是计算IoU值
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU


predict_image = cv2.imread(r'C:\Users\liuye\Desktop\Predict\01001.jpg')
label_image = cv2.imread(r'C:\Users\liuye\Desktop\Mask\01001.jpg')

predict_image = np.uint8(predict_image[:, :, 0] > 127)
label = np.uint8(label_image[:, :, 0] > 127)
label_img = label[:224, :224]
plt.imshow(label_img)
plt.show()


def Calculate_IoU(model, sequence):
    iou = MeanIoU(num_classes=1)
    iou_array = np.array([])
    for m, n in enumerate(sequence):
        iou_result = iou.update_state(n[0], n[1])
        iou_result = iou_result.result().numpy()
        np.append(iou_array, iou_result)
    return iou_array.mean()

x = label_img

