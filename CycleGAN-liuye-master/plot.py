import os

import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import numpy as np


def plot_heatmap(save_path, predict_array):
    if predict_array.ndim == 4:
        sns.heatmap(predict_array[:, :, :, 0].reshape((predict_array.shape[1], predict_array.shape[2])),
                    # annot=True,
                    xticklabels=False, yticklabels=False)
    else:
        sns.heatmap(predict_array[:, :, 0].reshape((predict_array.shape[0], predict_array.shape[1])),
                    # annot=True,
                    xticklabels=False, yticklabels=False)
    plt.savefig(save_path)


def plot_heatmap_crop_image(save_path, crop_array):
    plt.figure(figsize=(8, 3))

    sns.heatmap(crop_array[:, :, 0], annot=False, xticklabels=False, yticklabels=False,
                vmin=0, vmax=1,
                square=True, cbar='hot', cbar_kws={"shrink": 0.92})

    plt.savefig(save_path, bbox_inches='tight', pad_inches=-0.000)


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _rang


# 此函数是为了裁剪师姐的数据而创建的，目前已经裁剪完成，但是由于裁剪中心的设置，有一些图片需要进一步调整
def crop_image(image_path, size=227):
    # 这里已经预设为 227
    # 前两个坐标点是左上角坐标
    # 后两个坐标点是右下角坐标
    # width在前， height在后
    image = Image.open(image_path)
    width, height = image.size
    box_box = [(0, 0, size, size), (0, height - size, size, height),
               (width - size, 0, width, size), (width - size, height - size, width, height)]
    i = 1
    for box in box_box:
        region = image.crop(box)
        region.save(r'C:\Users\liuye\Desktop\data\crop_data\{}_crop_{}.jpg'.format(image_path[-7:-4], i))
        i += 1


# 此函数是为了将原始的细长裂缝的裂缝图像227×227填充至512×512，然后调用分割效果极好的模型进行分割
def pad_img(img, pad_size=(512, 512), values=255):
    new_image = np.pad(img, ((512 - img.shape[0], 0), (512 - img.shape[1], 0), (0, 0)), 'constant',
                       constant_values=values)

    return new_image


# path = r'C:\Users\liuye\Desktop\Slender Cracks Positive/'
# for i in os.listdir(path):
#     file_path = path + i
#     img = cv2.imread(file_path)
#     new_img = pad_img(img)
#     plt.imsave(r'C:\Users\liuye\Desktop\Machine_Background\Pad Crack Image\{}.jpg'.format(i[:5]), new_img)

# 柱状图

#
# labels = [1, 5, 10, 15]
# S = [0.7001, 0.7168, 0.7547, 0.7458]
# H = [0.7001, 0.7536, 0.7624, 0.7582]
# N = [0.7001, 0.7643, 0.7754, 0.7646]
#
# x = np.array([1, 5, 10, 15])
# # the label locations
# width = 3  # the width of the bars
#
# fig, ax = plt.subplots()
#
# rects1 = ax.bar(x - width / 3, S, width / 3, label='Standard Knowledge Distillation', color='deepskyblue', alpha=0.8)
# rects2 = ax.bar(x, H, width / 3, label='Hyper-thermal Knowledge Distillation', color='green', alpha=0.7)
# rects3 = ax.bar(x + width / 3, N, width / 3, label='Non-isothermal Knowledge Distillation', color='gold', alpha=0.9)
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('IoU')
# ax.set_xlabel('Temperature $T_0$')
# ax.set_ylim(0.66, 0.82)
# ax.set_xticks([1, 5, 10, 15])
#
# ax.legend()
#
# # ax.bar_label(rects1, padding=3)
# # ax.bar_label(rects2, padding=3)
#
# fig.tight_layout()
#
# plt.show()
#
# labels = [5, 10, 15]
# S = [0.7353, 0.7547, 0.7458]
# H = [0.7536, 0.7624, 0.7582]
# N = [0.7643, 0.7754, 0.7646]
#
# fig, ax = plt.subplots()
# y = np.array([0])
# rects0 = ax.bar(y, 0.7001, 1 + 0.1, label='No Knowledge Distillation', color='salmon', alpha=0.8)
# plt.text(0, 0.7003, '70.0%', ha='center', va='bottom')
# x = np.array([5, 10, 15])
# # the label locations
# width = 3  # the width of the bars
#
# rects1 = ax.bar(x - width / 3 - 0.38, S, width / 3 + 0.1, label='Standard Knowledge Distillation', color='deepskyblue', alpha=0.8)
# rects2 = ax.bar(x, H, width / 3 + 0.1, label='Hyperthermal Knowledge Distillation', color='green', alpha=0.7)
# rects3 = ax.bar(x + width / 3 + 0.38, N, width / 3 + 0.1, label='Non-isothermal Knowledge Distillation', color='gold', alpha=0.9)
# plt.text(4 - 0.38, 0.7355, '73.5%', ha='center', va='bottom')
# plt.text(9 - 0.38, 0.7549, '75.5%', ha='center', va='bottom')
# plt.text(14 - 0.38, 0.7460, '74.6%', ha='center', va='bottom')
#
# plt.text(5, 0.7538, '75.4%', ha='center', va='bottom')
# plt.text(10, 0.7626, '76.2%', ha='center', va='bottom')
# plt.text(15, 0.7584, '75.8%', ha='center', va='bottom')
#
# plt.text(6 + 0.38, 0.7645, '76.4%', ha='center', va='bottom')
# plt.text(11 + 0.38, 0.7756, '77.5%', ha='center', va='bottom')
# plt.text(16 + 0.38, 0.7648, '76.5%', ha='center', va='bottom')
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('IoU', fontsize=14)
# ax.set_xlabel('Temperature of soft label ($T_1$)', fontsize=14)
# ax.set_ylim(0.66, 0.84)
# ax.set_xticks([0, 5, 10, 15])
#
# ax.legend()

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)

# fig.tight_layout()
#
# plt.show()
# fig.savefig('test.png', dpi=800)
#
#
# labels = [5, 10, 15]
# S = [0.7515, 0.7627, 0.7535]
# H = [0.7621, 0.7739, 0.7721]
# N = [0.7756, 0.7840, 0.7802]
#
# fig, ax = plt.subplots()
# y = np.array([0])
# rects0 = ax.bar(y, 0.7116, 1 + 0.1, label='No Knowledge Distillation', color='salmon', alpha=0.8)
# plt.text(0, 0.7118, '71.2%', ha='center', va='bottom')
# x = np.array([5, 10, 15])
# # the label locations
# width = 3  # the width of the bars
#
# rects1 = ax.bar(x - width / 3 - 0.38, S, width / 3 + 0.1, label='Standard Knowledge Distillation', color='deepskyblue', alpha=0.8)
# rects2 = ax.bar(x, H, width / 3 + 0.1, label='Hyperthermal Knowledge Distillation', color='green', alpha=0.7)
# rects3 = ax.bar(x + width / 3 + 0.38, N, width / 3 + 0.1, label='Non-isothermal Knowledge Distillation', color='gold', alpha=0.9)
# plt.text(4 - 0.38, 0.7517, '75.2%', ha='center', va='bottom')
# plt.text(9 - 0.38, 0.7627, '76.3%', ha='center', va='bottom')
# plt.text(14 - 0.38, 0.7535, '75.4%', ha='center', va='bottom')
#
# plt.text(5, 0.7623, '76.2%', ha='center', va='bottom')
# plt.text(10, 0.7741, '77.4%', ha='center', va='bottom')
# plt.text(15, 0.7723, '77.2%', ha='center', va='bottom')
#
# plt.text(6 + 0.38, 0.7758, '77.6%', ha='center', va='bottom')
# plt.text(11 + 0.38, 0.7842, '78.4%', ha='center', va='bottom')
# plt.text(16 + 0.38, 0.7804, '78.0%', ha='center', va='bottom')
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('IoU', fontsize=14)
# ax.set_xlabel('Temperature of soft label ($T_1$)', fontsize=14)
# ax.set_ylim(0.66, 0.84)
# ax.set_xticks([0, 5, 10, 15])
#
# ax.legend()
#
# # ax.bar_label(rects1, padding=3)
# # ax.bar_label(rects2, padding=3)
#
# fig.tight_layout()
#
# plt.show()
# fig.savefig('val.png', dpi=800)
