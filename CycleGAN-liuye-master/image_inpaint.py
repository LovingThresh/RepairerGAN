# -*- coding: utf-8 -*-
# @Time    : 2022/4/24 9:56
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : image_inpaint.py
# @Software: PyCharm
import numpy as np
import cv2

img = cv2.imread('CFD_001.jpg')
mask = cv2.imread('CFD_001_mask.jpg', 0)

mask = cv2.dilate(mask, (3, 3), iterations=2)
dst = cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)

cv2.imshow('dst', dst)
cv2.waitKey(0)
