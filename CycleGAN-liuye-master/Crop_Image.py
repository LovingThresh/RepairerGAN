# -*- coding: utf-8 -*-
# @Time    : 2022/9/27 15:26
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : Crop_Image.py
# @Software: PyCharm
import cv2
import numpy as np

iter_100_image = cv2.imread(r'P:\GAN\CycleGAN-Tensorflow-2-master\output\crackattention_v2\samples_training\iter-000000100.jpg')
iter_100_crack_image = iter_100_image[:227, :227]
iter_100_crack_image = cv2.resize(iter_100_crack_image, (224, 224))
cv2.imwrite('M:/CycleGAN(WSSS)/File/image/iter_100_crack_image.png', iter_100_crack_image)
iter_100_gan_image = iter_100_image[:227, 227:454]
iter_100_gan_image = cv2.resize(iter_100_gan_image, (224, 224))
cv2.imwrite('M:/CycleGAN(WSSS)/File/image/iter_100_gan_image.png', iter_100_gan_image)

iter_60000_image = cv2.imread(r'P:\GAN\CycleGAN-Tensorflow-2-master\output\crackattention_v2\samples_training\iter-000060000.jpg')
iter_60000_crack_image = iter_60000_image[:227, :227]
iter_60000_crack_image = cv2.resize(iter_60000_crack_image, (224, 224))
cv2.imwrite('M:/CycleGAN(WSSS)/File/image/iter_60000_crack_image.png', iter_60000_crack_image)
iter_60000_gan_image = iter_60000_image[:227, 227:454]
iter_60000_gan_image = cv2.resize(iter_60000_gan_image, (224, 224))
cv2.imwrite('M:/CycleGAN(WSSS)/File/image/iter_60000_gan_image.png', iter_60000_gan_image)

iter_121800_image = cv2.imread(r'P:\GAN\CycleGAN-Tensorflow-2-master\output\crackattention_v2\samples_training\iter-000121800.jpg')
iter_121800_crack_image = iter_121800_image[:227, :227]
iter_121800_crack_image = cv2.resize(iter_121800_crack_image, (224, 224))
cv2.imwrite('M:/CycleGAN(WSSS)/File/image/iter_120000_crack_image.png', iter_121800_crack_image)
iter_121800_gan_image = iter_121800_image[:227, 227:454]
iter_121800_gan_image = cv2.resize(iter_121800_gan_image, (224, 224))
cv2.imwrite('M:/CycleGAN(WSSS)/File/image/iter_120000_gan_image.png', iter_121800_gan_image)

