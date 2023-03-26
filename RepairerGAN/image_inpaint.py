# -*- coding: utf-8 -*-
# @Time    : 2022/4/24 9:56
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : image_inpaint.py
# @Software: PyCharm
import cv2


def image_inpaint(image_path, mask_path, output_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)
    output = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    cv2.imwrite(output_path, output)
    return output


def image_resize(image_path, output_path, size):
    image = cv2.imread(image_path)
    output = cv2.resize(image, size)
    cv2.imwrite(output_path, output)
    return output

