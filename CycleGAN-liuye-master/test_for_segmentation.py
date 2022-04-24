# -*- coding: utf-8 -*-
# @Time    : 2022/3/26 11:23
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : test_for_segmentation.py
# @Software: PyCharm

# scheme 1
# Step 1: 导入模型观察分割标签输出结果
# Step 2：导入分割标签查看最终结果

import os
import cv2
import module
import numpy as np
import tf2lib as tl
import tensorflow as tf
import tensorflow.keras as keras


path = r'P:\GAN\CycleGAN-liuye-master\CycleGAN-liuye-master\datasets\crack\Positive/'
file_list = os.listdir(path)


hyper_params = {'crop_size': 224,
                'epochs': 5,
                'epoch_decay': 2,
                'learning_rate_G': 0.0002,
                'learning_rate_D': 0.00002,
                }
len_dataset = 19000


# define the model

G_A2B = module.AttentionCycleGAN_v1_Generator(input_shape=(hyper_params['crop_size'], hyper_params['crop_size'], 3),
                                              attention=True)
G_B2A = module.AttentionCycleGAN_v1_Generator(input_shape=(hyper_params['crop_size'], hyper_params['crop_size'], 3),
                                              attention=False)

D_A = module.ConvDiscriminator(input_shape=(hyper_params['crop_size'], hyper_params['crop_size'], 3), dim=32,
                               n_downsamplings=3)
D_B = module.ConvDiscriminator(input_shape=(hyper_params['crop_size'], hyper_params['crop_size'], 3), dim=32,
                               n_downsamplings=3)


G_lr_scheduler = module.LinearDecay(hyper_params['learning_rate_G'], hyper_params['epochs'] * len_dataset,
                                    hyper_params['epoch_decay'] * len_dataset)
D_lr_scheduler = module.LinearDecay(hyper_params['learning_rate_D'], hyper_params['epochs'] * len_dataset,
                                    hyper_params['epoch_decay'] * len_dataset)

G_optimizer = keras.optimizers.RMSprop(learning_rate=G_lr_scheduler)
D_optimizer = keras.optimizers.RMSprop(learning_rate=D_lr_scheduler)


ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
checkpoint_path = r'P:\GAN_CheckPoint\2022-03-26-10-18-38.504991/'


checkpoint = tl.Checkpoint(dict(G_A2B=G_A2B,
                                G_B2A=G_B2A,
                                D_A=D_A,
                                D_B=D_B,
                                G_optimizer=G_optimizer,
                                D_optimizer=D_optimizer,
                                ep_cnt=ep_cnt),
                           checkpoint_path,
                           max_to_keep=5)

try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
    print('Load Model Success!')
except Exception as e:
    print(e)


G_A2B = checkpoint.checkpoint.G_A2B
model = keras.models.Model(inputs=G_A2B.inputs, outputs=[G_A2B.outputs[1]])
for i in file_list:
    test_data = cv2.imread(path + i)
    test_data = cv2.resize(test_data, dsize=(hyper_params['crop_size'], hyper_params['crop_size']))
    test_data = test_data.astype(np.float32)
    test_data = tf.clip_by_value(test_data, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
    test_data = test_data * 2 - 1
    test_data = tf.convert_to_tensor(test_data, dtype=tf.float32)
    test_data = tf.reshape(test_data, (1, 224, 224, 3))
    seg_mask = G_A2B(test_data, training=False)[1]
    seg_mask = tf.clip_by_value(seg_mask, -1, 1)
    seg_mask = np.reshape(seg_mask.numpy(), (224, 224, 3))
    # seg_mask = (seg_mask * 255).astype(np.int16)
    seg_mask = ((seg_mask + 1) * 0.5 * 255).astype(np.int8)
    cv2.imwrite(r'P:\GAN\CycleGAN-liuye-master\CycleGAN-liuye-master\Mask\2022-03-26-10-18-38.504991\int8_0.5_1/' + i,
                seg_mask)



