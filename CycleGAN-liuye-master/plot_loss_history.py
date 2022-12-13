# -*- coding: utf-8 -*-
# @Time    : 2022/10/28 14:20
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : plot_loss_history.py
# @Software: PyCharm
import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Users\liuye\Downloads\id_loss.csv')

X = data['Step']
Y = data['Value']
plt.figure()
plt.plot(X, Y, label='Identity Loss', color='#3979F2')
plt.xlabel('Iteration Steps')
plt.ylabel('Loss Value')
plt.ylim((0.0, 0.45))
plt.xlim((0, 18000))
plt.legend(loc='upper right')
plt.savefig('M:\CycleGAN(WSSS)\File\Figure\id_loss.png', dpi=800)

data = pd.read_csv(r'C:\Users\liuye\Downloads\cyclegan_loss.csv')

X = data['Step']
Y = data['Value']
plt.figure()
plt.plot(X, Y, label='Cycle Loss', color='#3979F2')
plt.xlabel('Iteration Steps')
plt.ylabel('Loss Value')
plt.ylim((0.0, 0.45))
plt.xlim((0, 18000))
plt.legend(loc='upper right')
plt.savefig('M:\CycleGAN(WSSS)\File\Figure\cyclegan_loss.png', dpi=800)

data = pd.read_csv(r'C:\Users\liuye\Downloads\ssim_loss.csv')

X = data['Step']
Y = data['Value']
plt.figure()
plt.plot(X, Y, label='SSIM Loss', color='#3979F2')
plt.xlabel('Iteration Steps')
plt.ylabel('Loss Value')
plt.ylim((0.0, 1.65))
plt.xlim((0, 18000))
plt.legend(loc='upper right')
plt.savefig('M:\CycleGAN(WSSS)\File\Figure\ssim_loss.png', dpi=800)

data = pd.read_csv(r'C:\Users\liuye\Downloads\G_loss.csv')
# #E4392E
X = data['Step']
Y = data['Value']
plt.figure()
plt.plot(X, Y, label='Generator Loss', color='#3979F2')
plt.plot(X, np.ones_like(X), label='Target Value', color='#E4392E', linestyle='--', linewidth=2.5)
# 最后两百个iter的平均值
plt.plot(X[700 : 1000], np.ones_like(X[700 : 1000]) * 0.9388, label='Convergence value', color='#F47E62', linestyle='--', linewidth=2.5)
plt.xlabel('Iteration Steps')
plt.ylabel('Loss Value')
plt.xlim((0, 18000))
plt.ylim((0.0, 1.6))
plt.legend(loc='upper right')
plt.savefig('M:\CycleGAN(WSSS)\File\Figure\g_loss.png', dpi=800)

data = pd.read_csv(r'C:\Users\liuye\Downloads\D_loss.csv')
# #E4392E
X = data['Step']
Y = data['Value']
plt.figure()
plt.plot(X, Y, label='Discriminator Loss', color='#3979F2')
plt.plot(X, np.ones_like(X) * 0.5, label='Target Value', color='#E4392E', linestyle='--', linewidth=2.5)
# 最后两百个iter的平均值
plt.plot(X[700 : 1000], np.ones_like(X[700 : 1000]) * 0.4947, label='Convergence value', color='#F47E62', linestyle='--', linewidth=2.5)
plt.xlabel('Iteration Steps')
plt.ylabel('Loss Value')
plt.xlim((0, 18000))
plt.ylim((0.0, 1.2))
plt.legend(loc='upper right')
plt.savefig('M:\CycleGAN(WSSS)\File\Figure\d_loss.png', dpi=800)

