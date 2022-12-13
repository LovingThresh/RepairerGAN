# -*- coding: utf-8 -*-
# @Time    : 2022/7/22 9:15
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : SuperPixel.py
# @Software: PyCharm
import math
import os
import cv2
from skimage import io, color
import numpy as np
from tqdm import trange

global path

class Cluster(object):
    cluster_index = 1

    def __init__(self, h, w, l=0, a=0, b=0):
        self.update(h, w, l, a, b)
        self.pixels = []
        self.no = self.cluster_index
        self.cluster_index += 1

    def update(self, h, w, l, a, b):
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b

    def __str__(self):
        return "{},{}:{} {} {} ".format(self.h, self.w, self.l, self.a, self.b)

    def __repr__(self):
        return self.__str__()


class SLICProcessor(object):
    @staticmethod
    def open_image(path):
        """
        Return:
            3D array, row col [LAB]
        """
        rgb = io.imread(path)
        lab_arr = color.rgb2lab(rgb)
        return lab_arr

    @staticmethod
    def save_lab_image(path, lab_arr):
        """
        Convert the array to RBG, then save the image
        """
        rgb_arr = color.lab2rgb(lab_arr)
        label = (rgb_arr[:, :, 0] < 127.5).astype(np.uint8)
        eroded = cv2.erode(label, (3, 3), 1)
        dilated = cv2.dilate(eroded, (3, 3), 1)
        rgb_arr = dilated
        io.imsave(path, rgb_arr)

    def make_cluster(self, h, w):
        h, w = int(h), int(w)
        return Cluster(h, w,
                       self.data[h][w][0],
                       self.data[h][w][1],
                       self.data[h][w][2])

    def __init__(self, filename, K, M):
        self.K = K
        self.M = M

        self.data = self.open_image(filename)
        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]
        self.N = self.image_height * self.image_width
        self.S = int(math.sqrt(self.N / self.K))

        self.clusters = []
        self.label = {}
        self.dis = np.full((self.image_height, self.image_width), np.inf)

    def init_clusters(self):
        h = self.S / 2
        w = self.S / 2
        while h < self.image_height:
            while w < self.image_width:
                self.clusters.append(self.make_cluster(h, w))
                w += self.S
            w = self.S / 2
            h += self.S

    def get_gradient(self, h, w):
        if w + 1 >= self.image_width:
            w = self.image_width - 2
        if h + 1 >= self.image_height:
            h = self.image_height - 2

        gradient = self.data[w + 1][h + 1][0] - self.data[w][h][0] + \
                   self.data[w + 1][h + 1][1] - self.data[w][h][1] + \
                   self.data[w + 1][h + 1][2] - self.data[w][h][2]
        return gradient

    def move_clusters(self):
        for cluster in self.clusters:
            cluster_gradient = self.get_gradient(cluster.h, cluster.w)
            for dh in range(-1, 2):
                for dw in range(-1, 2):
                    _h = cluster.h + dh
                    _w = cluster.w + dw
                    new_gradient = self.get_gradient(_h, _w)
                    if new_gradient < cluster_gradient:
                        cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])
                        cluster_gradient = new_gradient

    def assignment(self):
        for cluster in self.clusters:
            for h in range(cluster.h - 2 * self.S, cluster.h + 2 * self.S):
                if h < 0 or h >= self.image_height: continue
                for w in range(cluster.w - 2 * self.S, cluster.w + 2 * self.S):
                    if w < 0 or w >= self.image_width: continue
                    L, A, B = self.data[h][w]
                    Dc = math.sqrt(
                        math.pow(L - cluster.l, 2) +
                        math.pow(A - cluster.a, 2) +
                        math.pow(B - cluster.b, 2))
                    Ds = math.sqrt(
                        math.pow(h - cluster.h, 2) +
                        math.pow(w - cluster.w, 2))
                    D = math.sqrt(math.pow(Dc / self.M, 2) + math.pow(Ds / self.S, 2))
                    if D < self.dis[h][w]:
                        if (h, w) not in self.label:
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        else:
                            self.label[(h, w)].pixels.remove((h, w))
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        self.dis[h][w] = D

    def update_cluster(self):
        for cluster in self.clusters:
            sum_h = sum_w = number = 0
            for p in cluster.pixels:
                sum_h += p[0]
                sum_w += p[1]
                number += 1
                _h = int(sum_h / number)
                _w = int(sum_w / number)
                cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])

    def save_current_image(self, name):
        image_arr = np.copy(self.data)
        for cluster in self.clusters:
            for p in cluster.pixels:
                image_arr[p[0]][p[1]][0] = cluster.l
                image_arr[p[0]][p[1]][1] = cluster.a
                image_arr[p[0]][p[1]][2] = cluster.b
            image_arr[cluster.h][cluster.w][0] = 0
            image_arr[cluster.h][cluster.w][1] = 0
            image_arr[cluster.h][cluster.w][2] = 0
        self.save_lab_image(name, image_arr)

    def iterate_10times(self):
        self.init_clusters()
        self.move_clusters()
        for i in trange(20):
            self.assignment()
            self.update_cluster()
            name = path
        self.save_current_image(name)


if __name__ == '__main__':
    load_path = r'P:\GAN\CycleGAN-liuye-master\CycleGAN-liuye-master\datasets\crack\img_dir\test_Positive/'
    save_path = r'P:\GAN\CycleGAN-liuye-master\CycleGAN-liuye-master\datasets\crack\img_dir\test_Positive_SP/'
    for file in os.listdir(load_path):
        load_path_ = load_path + file
        path = save_path + file[:-4] + '.png'
        for k in [16, 32, 64, 128]:
            p = SLICProcessor(load_path_, k, 30)
            p.iterate_10times()

# https://github.com/LarkMi/SLIC/blob/main/SLIC.py

import skimage
from skimage.segmentation import slic, mark_boundaries
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import numpy as np
import cv2

#
# np.set_printoptions(threshold=np.inf)
path = r'P:\GAN\CycleGAN-liuye-master\CycleGAN-liuye-master\datasets\MCFF_crack\test_Positive/'
img_name = 'CRACK500_20160306_104527_1281_721.jpg'
img = io.imread(path + img_name, as_gray=True)  # as_gray是灰度读取，得到的是归一化值
segments = slic(img, n_segments=50, compactness=0.2, start_label=1)  # 进行SLIC分割
out = mark_boundaries(img, segments)
out = out * 255  # io的灰度读取是归一化值，若读取彩色图片去掉该行
img3 = Image.fromarray(np.uint8(out))
img3.show()
seg_img_name = 'seg.png'
img3.save(path + '\\' + seg_img_name)  # 显示并保存加上分割线后的图片

maxn = max(segments.reshape(int(segments.shape[0] * segments.shape[1]), ))

for i in range(1, maxn + 1):
    a = np.array(segments == i)
    b = img * a
    w, h = [], []
    for x in range(b.shape[0]):
        for y in range(b.shape[1]):
            if b[x][y] != 0:
                w.append(x)
                h.append(y)

    c = b[min(w):max(w), min(h):max(h)]
    c = c * 255
    d = c.reshape(c.shape[0], c.shape[1], 1)
    e = np.concatenate((d, d), axis=2)
    e = np.concatenate((e, d), axis=2)
    img2 = Image.fromarray(np.uint8(e))
    img2.save(path + '\\' + str(i) + '.png')
    print('已保存第' + str(i) + '张图片')

wid, hig = [], []
img = io.imread(path + '\\' + seg_img_name)

for i in range(1, maxn + 1):
    w, h = [], []
    for x in range(segments.shape[0]):
        for y in range(segments.shape[1]):
            if segments[x][y] == i:
                w.append(x)
                h.append(y)

    font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
    # print((min(w),min(h)))
    img = cv2.putText(img, str(i), (h[int(len(h) / (2))], w[int(len(w) / 2)]), font, 1, (255, 255, 255),
                      2)  # 添加文字，1.2表示字体大小，（0,40）是初始的位置，(255,255,255)表示颜色，2表示粗细
img = Image.fromarray(np.uint8(img))
img.show()
img.save(r'P:\GAN\CycleGAN-liuye-master\CycleGAN-liuye-master\datasets\crack\img_dir\test_Positive_SP/' + '\\' + seg_img_name + '_label.png')


# 将所有图像进行取反
