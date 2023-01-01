# -*- coding: utf-8 -*-
# @Time    : 2022/9/27 15:26
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : Crop_Image.py
# @Software: PyCharm
import os

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

iter_6600_image = cv2.imread(r'M:\CycleGAN(WSSS)\File\image\iter-000006600.jpg')
iter_6600_crack_image = iter_6600_image[:227, :227]
iter_6600_crack_image = cv2.resize(iter_6600_crack_image, (224, 224))
cv2.imwrite('M:/CycleGAN(WSSS)/File/image/iter_6600_crack_image.png', iter_6600_crack_image)
iter_6600_gan_image = iter_6600_image[:227, 227:454]
iter_6600_gan_image = cv2.resize(iter_6600_gan_image, (224, 224))
cv2.imwrite('M:/CycleGAN(WSSS)/File/image/iter_6600_gan_image.png', iter_6600_gan_image)
iter_6600_att_image = iter_6600_image[:227, 454:454+227]
iter_6600_att_image = cv2.resize(iter_6600_att_image, (224, 224))
cv2.imwrite('M:/CycleGAN(WSSS)/File/image/iter_6600_att_image.png', iter_6600_att_image)

iter_26800_image = cv2.imread(r'M:\CycleGAN(WSSS)\File\image\iter-000026800.jpg')
iter_26800_crack_image = iter_26800_image[:227, :227]
iter_26800_crack_image = cv2.resize(iter_26800_crack_image, (224, 224))
cv2.imwrite('M:/CycleGAN(WSSS)/File/image/iter_26800_crack_image.png', iter_26800_crack_image)
iter_26800_gan_image = iter_26800_image[:227, 227:454]
iter_26800_gan_image = cv2.resize(iter_26800_gan_image, (224, 224))
cv2.imwrite('M:/CycleGAN(WSSS)/File/image/iter_26800_gan_image.png', iter_26800_gan_image)
iter_26800_att_image = iter_26800_image[:227, 454:454+227]
iter_26800_att_image = cv2.resize(iter_26800_att_image, (224, 224))
cv2.imwrite('M:/CycleGAN(WSSS)/File/image/iter_26800_att_image.png', iter_26800_att_image)

m = 9

image = cv2.imread(r'C:\Users\liuye\Desktop\image\iter-000000001.jpg')
image_1 = image[:227, :227]
image_2 = image[:227, 227:454]
image_3 = image[:227, 454:454 + 227]
cv2.imwrite(r'C:\Users\liuye\Desktop\image/{}.png'.format(m), cv2.resize(image_1, (224, 224)))
cv2.imwrite(r'C:\Users\liuye\Desktop\image/{}.png'.format(m + 1), cv2.resize(image_2, (224, 224)))
cv2.imwrite(r'C:\Users\liuye\Desktop\image/{}.png'.format(m + 2), cv2.resize(image_3, (224, 224)))















