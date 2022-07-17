# -*- coding: utf-8 -*-
# @Time    : 2022/3/19 9:25
# @Author  : Liuyee
# @Email   : SY2113205@buaa.edu.cn / csu1704liuye@163.com
# @File    : train_comet.py
# @Software: PyCharm

import json
import shutil
import random
import os.path
import datetime
import functools
from comet_ml import Experiment

import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras

import tqdm
import Cycle_data as data
import module
import Metrics
import tf2lib as tl
import tf2gan as gan

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

experiment_button = False
training = True
experiment = object

if experiment:
    pass


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.random.set_seed(seed)  # tf cpu fix seed
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first


setup_seed(42)


if experiment_button:
    experiment = Experiment(
        api_key="sDV9A5CkoqWZuJDeI9JbJMRvp",
        project_name="CycleGAN_for_Crack",
        workspace="lovingthresh",
    )

hyper_params = {
    'ex_number': 'A2A_ssim_4_Bridge_3080Ti',
    'device': '3080Ti',
    'data_type': 'crack',
    'datasets_dir': r'datasets',
    'load_size': 224,
    'crop_size': 224,
    'batch_size': 1,
    'epochs': 8,
    'epoch_decay': 4,
    'learning_rate_G': 0.0002,
    'learning_rate_D': 0.00002,
    'learning_rate_D_B_weight': 1.0,
    'beta_1': 0.5,
    'adversarial_loss_mode': 'lsgan',
    'gradient_penalty_mode': 'none',
    'gradient_penalty_weight': 10.0,
    'g_loss_weight': 2.0,
    'cycle_loss_weight': 10.0,
    'identity_loss_weight': 10.0,
    'ssim_loss_weight': 1.0,
    'ssim_Fake_True_weight': 5.0,
    'std_loss_weight': 50.0,
    'pool_size': 50,
    'lambda_reg': 1e-6,
    'starting_rate': 0.01
}

# output_dir
a = str(datetime.datetime.now())
b = list(a)
b[10] = '-'
b[13] = '-'
b[16] = '-'
output_dir = ''.join(b)
repeat_num = 12

if hyper_params['device'] == 'A40':
    output_dir = r'/root/autodl-tmp/Cycle_GAN/{}'.format(output_dir)
    hyper_params['batch_size'] = 2
    repeat_num = 4
elif hyper_params['device'] == '3080Ti' or '3090':
    output_dir = r'E:/Cycle_GAN/output/{}'.format(output_dir)
    if hyper_params['device'] == '3080Ti':
        hyper_params['batch_size'] = 1
        repeat_num = 1
    else:
        hyper_params['batch_size'] = 2
        repeat_num = 4
py.mkdir(output_dir)
hyper_params['repeat_num'] = repeat_num
hyper_params['output_dir'] = output_dir
hyper_params['ex_date'] = a[:10]

if experiment_button:
    hyper_params['ex_key'] = experiment.get_key()
    experiment.log_parameters(hyper_params)
    experiment.set_name('{}-{}'.format(hyper_params['ex_date'], hyper_params['ex_number']))
    experiment.add_tag('AttentionGAN')
    experiment.add_tag('Base')
    experiment.add_tag('A2A')
    experiment.add_tag('RMSprop')
    experiment.add_tag('Weak-D')

with open('{}/hyper_params.json'.format(output_dir), 'w') as fp:
    json.dump(hyper_params, fp)
# ==============================================================================
# =                                    data                                    =
# ==============================================================================

# Dataset制作
A_img_paths = py.glob(py.join(hyper_params['datasets_dir'], hyper_params['data_type'], 'train_Positive'), '*.jpg')
B_img_paths = py.glob(py.join(hyper_params['datasets_dir'], hyper_params['data_type'], 'train_Negative'), '*.jpg')
A_B_dataset, len_dataset = data.make_zip_dataset(A_img_paths, B_img_paths,
                                                 hyper_params['batch_size'],
                                                 hyper_params['load_size'],
                                                 hyper_params['crop_size'],
                                                 training=True,
                                                 shuffle=False,
                                                 repeat=repeat_num,
                                                 )

# Segmentation数据制作
A_img_val_paths = py.glob(py.join(hyper_params['datasets_dir'], hyper_params['data_type'], 'test_Positive'), '*.jpg')
A_mask_paths = py.glob(py.join(hyper_params['datasets_dir'], hyper_params['data_type'], 'test_Positive_mask'), '*.jpg')
A_mask_dataset, len_mask_dataset = data.make_zip_dataset(A_img_val_paths, A_mask_paths,
                                                         hyper_params['batch_size'],
                                                         hyper_params['load_size'],
                                                         hyper_params['crop_size'],
                                                         training=False,
                                                         shuffle=False,
                                                         repeat=1,
                                                         random_fn=False,
                                                         mask=True)

# 用来保存假样本
A2B_pool = data.ItemPool(hyper_params['pool_size'])
B2A_pool = data.ItemPool(hyper_params['pool_size'])

# 测试样本，可以略过
A_img_paths_test = py.glob(py.join(hyper_params['datasets_dir'], hyper_params['data_type'], 'val_Positive'), '*.jpg')
B_img_paths_test = py.glob(py.join(hyper_params['datasets_dir'], hyper_params['data_type'], 'val_Negative'), '*.jpg')
A_B_dataset_test, _ = data.make_zip_dataset(
    A_img_paths_test, B_img_paths_test, hyper_params['batch_size'], hyper_params['load_size'],
    hyper_params['crop_size'],
    training=False, shuffle=False,
    repeat=repeat_num)

# ==============================================================================
# =                                   models                                   =
# ==============================================================================

G_A2B = module.AttentionCycleGAN_v1_Generator(input_shape=(hyper_params['crop_size'], hyper_params['crop_size'], 3),
                                              attention=True)
G_B2A = module.AttentionCycleGAN_v1_Generator(input_shape=(hyper_params['crop_size'], hyper_params['crop_size'], 3),
                                              attention=False)

D_A = module.ConvDiscriminator(input_shape=(hyper_params['crop_size'], hyper_params['crop_size'], 3), dim=32,
                               n_downsamplings=4)
D_B = module.ConvDiscriminator(input_shape=(hyper_params['crop_size'], hyper_params['crop_size'], 3), dim=32,
                               n_downsamplings=4)


# ==============================================================================
# =                                   losses                                   =
# ==============================================================================

def ssim_loss(y_pred, y_true):
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    return -1 * tf.reduce_sum(-1 + tf.image.ssim(y_pred, y_true, max_val=1.0, filter_size=11,
                                                 filter_sigma=1.5, k1=0.01, k2=0.03))


def contentFunc():
    cnn = tf.keras.applications.vgg19.VGG19(include_top=False, input_shape=(hyper_params['crop_size'], hyper_params['crop_size'], 3))
    input = cnn.inputs
    output = cnn.layers[9].output
    model = tf.keras.models.Model(input, output)
    model.trainable = False
    return model


class PerceptualLoss:

    def __init__(self, loss):
        self.criterion = loss
        self.contentFunc = contentFunc()

    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc(fakeIm)
        f_real = self.contentFunc(realIm)
        loss = self.criterion(f_fake, f_real)
        return loss


def perceptual_loss(y_pred, y_true):
    P = PerceptualLoss(tf.keras.losses.MeanSquaredError())
    return 0.2 * P.get_loss(y_pred, y_true)


d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(hyper_params['adversarial_loss_mode'])
cycle_loss_fn = tf.losses.MeanAbsoluteError()
identity_loss_fn = tf.losses.MeanAbsoluteError()

# ==============================================================================
# =                             lr_scheduler                                   =
# ==============================================================================
G_lr_scheduler = module.LinearDecay(hyper_params['learning_rate_G'], hyper_params['epochs'] * len_dataset,
                                    hyper_params['epoch_decay'] * len_dataset)
D_lr_scheduler = module.LinearDecay(hyper_params['learning_rate_D'], hyper_params['epochs'] * len_dataset,
                                    hyper_params['epoch_decay'] * len_dataset)

G_optimizer = keras.optimizers.RMSprop(learning_rate=G_lr_scheduler)
D_optimizer = keras.optimizers.RMSprop(learning_rate=D_lr_scheduler)


# G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler)
# D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler)

# ==============================================================================
# =                                 train step                                 =
# ==============================================================================


# @tf.function
def train_G(A_True, B_True):
    with tf.GradientTape() as t:
        A2B_Fake = G_A2B(A_True, training=True)[0]
        A2B_m = G_A2B(A_True, training=True)[2]
        A2B_n = G_A2B(A_True, training=True)[3]
        A2B_content = G_A2B(A_True, training=True)[4]
        B2A_Fake = G_B2A(B_True, training=True)
        A2B2A_Fake = G_B2A(A2B_Fake, training=True)
        B2A2B_Fake = G_A2B(B2A_Fake, training=True)[0]
        B2A2B_m = G_A2B(B2A_Fake, training=True)[2]
        B2A2B_n = G_A2B(B2A_Fake, training=True)[3]
        B2A2B_content = G_A2B(B2A_Fake, training=True)[4]
        A2A = G_B2A(A_True, training=True)
        B2B = G_A2B(B_True, training=True)[0]
        B2B_m = G_A2B(B_True, training=True)[2]
        B2B_n = G_A2B(B_True, training=True)[3]
        B2B_content = G_A2B(B_True, training=True)[4]
        # A2B, mask_B, temp_B = G_A2B(A, training=True)
        # B2A, mask_A, temp_A = G_B2A(B, training=True)
        # A2B2A, _, _ = G_B2A(A2B, training=True)
        # B2A2B, _, _ = G_A2B(B2A, training=True)
        # A2A, _, _ = G_B2A(A, training=True)
        # B2B, _, _ = G_A2B(B, training=True)

        A2B_d_logits = D_B(A2B_Fake, training=True)
        B2A_d_logits = D_A(B2A_Fake, training=True)

        A2B_g_loss = g_loss_fn(A2B_d_logits)
        B2A_g_loss = g_loss_fn(B2A_d_logits)
        A2B2A_cycle_loss = cycle_loss_fn(A_True, A2B2A_Fake)
        B2A2B_cycle_loss = cycle_loss_fn(B_True, B2A2B_Fake)
        A2A_id_loss = identity_loss_fn(A_True, A2A)
        B2B_id_loss = identity_loss_fn(B_True, B2B)

        # std_loss_1 = tf.math.reduce_std(A2B_Fake) - tf.math.reduce_std(B_True)

        # loss_reg_A = args.lambda_reg * (
        #         K.sum(K.abs(mask_A[:, :-1, :, :] - mask_A[:, 1:, :, :])) +
        #         K.sum(K.abs(mask_A[:, :, :-1, :] - mask_A[:, :, 1:, :])))
        #
        # loss_reg_B = args.lambda_reg * (
        #         K.sum(K.abs(mask_B[:, :-1, :, :] - mask_B[:, 1:, :, :])) +
        #         K.sum(K.abs(mask_B[:, :, :-1, :] - mask_B[:, :, 1:, :])))

        # rate = args.starting_rate

        s_loss_1 = ssim_loss(A2B_m, A2B_n)
        s_loss_2 = ssim_loss(B2A2B_m, B2A2B_n)
        s_loss_3 = ssim_loss(B2B_m, B2B_n)
        s_loss_4 = ssim_loss(A2B_Fake, B_True)

        Perceptual_loss_1 = perceptual_loss(A2B_content, B_True)
        Perceptual_loss_2 = perceptual_loss(B2A2B_content, B_True)
        Perceptual_loss_3 = perceptual_loss(B2B_content, B_True)

        G_loss = \
        hyper_params['g_loss_weight'] * (A2B_g_loss + B2A_g_loss) + \
        hyper_params['cycle_loss_weight'] * (A2B2A_cycle_loss + B2A2B_cycle_loss) + \
        hyper_params['identity_loss_weight'] * (A2A_id_loss + B2B_id_loss) + \
        hyper_params['ssim_loss_weight'] * (s_loss_1 + s_loss_2 + s_loss_3) + \
        hyper_params['ssim_Fake_True_weight'] * s_loss_4 + \
        (Perceptual_loss_1 + Perceptual_loss_2 + Perceptual_loss_3)
        # hyper_params['std_loss_weight'] * std_loss_1

    G_grad = t.gradient(G_loss, G_A2B.trainable_variables + G_B2A.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables + G_B2A.trainable_variables))

    return A2B_Fake, B2A_Fake, {'A2B_g_loss': A2B_g_loss,
                                'B2A_g_loss': B2A_g_loss,
                                'A2B2A_cycle_loss': A2B2A_cycle_loss,
                                'B2A2B_cycle_loss': B2A2B_cycle_loss,
                                'A2A_id_loss': A2A_id_loss,
                                'B2B_id_loss': B2B_id_loss,
                                's_loss': s_loss_1 + s_loss_2 + s_loss_3,
                                'ssim_Fake_True_weight': s_loss_4,
                                'Perceptual_loss': Perceptual_loss_1 + Perceptual_loss_2 + Perceptual_loss_3,
                                # 'std_loss': std_loss_1
                                }
    # 'loss_reg_A': loss_reg_A,
    # 'loss_reg_B': loss_reg_B


@tf.function
def train_D(A_True, B_True, A2B_Fake, B2A_Fake):
    with tf.GradientTape() as t:
        A_d_logits = D_A(A_True, training=True)
        B2A_d_logits = D_A(B2A_Fake, training=True)
        B_d_logits = D_B(B_True, training=True)
        A2B_d_logits = D_B(A2B_Fake, training=True)

        A_d_loss, B2A_d_loss = d_loss_fn(A_d_logits, B2A_d_logits)
        B_d_loss, A2B_d_loss = d_loss_fn(B_d_logits, A2B_d_logits)
        D_A_gp = gan.gradient_penalty(functools.partial(D_A, training=True), A_True, B2A_Fake,
                                      mode=hyper_params['gradient_penalty_mode'])
        D_B_gp = gan.gradient_penalty(functools.partial(D_B, training=True), B_True, A2B_Fake,
                                      mode=hyper_params['gradient_penalty_mode'])

        D_loss = (A_d_loss + B2A_d_loss) + hyper_params['learning_rate_D_B_weight'] * (B_d_loss + A2B_d_loss) + \
        hyper_params['gradient_penalty_weight'] * (D_A_gp + D_B_gp)

    D_grad = t.gradient(D_loss, D_A.trainable_variables + D_B.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, D_A.trainable_variables + D_B.trainable_variables))

    return {'A_d_loss': A_d_loss + B2A_d_loss,
            'B_d_loss': B_d_loss + A2B_d_loss,
            'D_A_gp': D_A_gp,
            'D_B_gp': D_B_gp}


def train_step(A_True, B_True):
    A2B_Fake, B2A_Fake, G_Loss_dict = train_G(A_True, B_True)

    # cannot autograph `A2B_pool`
    A2B_Fake = A2B_pool(A2B_Fake)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower
    B2A_Fake = B2A_pool(B2A_Fake)  # because of the communication between CPU and GPU

    D_Loss_dict = train_D(A_True, B_True, A2B_Fake, B2A_Fake)

    return G_Loss_dict, D_Loss_dict


@tf.function
def sample(A_Test, B_Test):
    A2B_Test = G_A2B(A_Test, training=False)[0]
    m_Test = G_A2B(A_Test, training=False)[2]
    n_Test = G_A2B(A_Test, training=False)[3]
    B2A_Test = G_B2A(B_Test, training=False)
    A2B_mask_Test = G_A2B(A_Test, training=False)[1]
    A2B2A_Test = G_B2A(A2B_Test, training=False)
    B2A2B_Test = G_A2B(B2A_Test, training=False)[0]
    B2A2B_mask_Test = G_A2B(B2A_Test, training=False)[1]
    A2A_Fake = G_B2A(A_Test, training=False)
    A2A_mask_Test = G_A2B(A2A_Fake, training=False)[1]

    return A2B_Test[0:1, :, :, :], B2A_Test[0:1, :, :, :], A2B_mask_Test[0:1, :, :, :], A2B2A_Test[0:1, :, :, :], \
    B2A2B_Test[0:1, :, :, :], B2A2B_mask_Test[0:1, :, :, :], m_Test[0:1, :, :, :], n_Test[0:1, :, :, :], \
    A2A_Fake[0:1, :, :, :], A2A_mask_Test[0:1, :, :, :]


# ==============================================================================
# =                             checkpoint                                     =
# ==============================================================================
# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(G_A2B=G_A2B,
                                G_B2A=G_B2A,
                                D_A=D_A,
                                D_B=D_B,
                                G_optimizer=G_optimizer,
                                D_optimizer=D_optimizer,
                                ep_cnt=ep_cnt),
                           py.join(output_dir, 'checkpoints'),
                           max_to_keep=5)
try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
    print('Load Model Success!')
except Exception as e:
    print(e)

# ==============================================================================
# =                           copy & upload                                    =
# ==============================================================================
module_dir = py.join(output_dir, 'module_code')
py.mkdir(module_dir)
os.mkdir(os.path.join(output_dir, 'save_model'))

# 本地复制源代码，便于复现(模型文件、数据文件、训练文件、测试文件)
# 冷代码
shutil.copytree('imlib', '{}/{}'.format(module_dir, 'imlib'))
shutil.copytree('pylib', '{}/{}'.format(module_dir, 'pylib'))
shutil.copytree('tf2gan', '{}/{}'.format(module_dir, 'tf2gan'))
shutil.copytree('tf2lib', '{}/{}'.format(module_dir, 'tf2lib'))

# 个人热代码
shutil.copy('module.py', module_dir)
shutil.copy('Cycle_data.py', module_dir)
shutil.copy('train_comet.py', module_dir)
shutil.copy('test.py', module_dir)

# 云端上次源代码
if experiment_button:
    experiment.log_asset_folder('imlib', log_file_name=True)
    experiment.log_asset_folder('pylib', log_file_name=True)
    experiment.log_asset_folder('tf2gan', log_file_name=True)
    experiment.log_asset_folder('tf2lib', log_file_name=True)

    experiment.log_code('module.py')
    experiment.log_code('data.py')
    experiment.log_code('train_comet.py')
    experiment.log_code('test.py')


# ==============================================================================
# =                     Weak Supervision Val                                   =
# ==============================================================================
def Validation(model, dataset):
    model = keras.models.Model(inputs=model.inputs, outputs=[model.outputs[1][:, :, :, 0:1]])
    initial_learning_rate = 5e-5
    optimizer = keras.optimizers.RMSprop(initial_learning_rate)
    model.compile(optimizer=optimizer,
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=Metrics.METRICS)
    return model.evaluate(dataset), model


# ==============================================================================
# =                     summary and train                                      =
# ==============================================================================
def train(Step=0):
    # sample
    test_iter = iter(A_B_dataset_test)
    sample_dir = py.join(output_dir, 'samples_training')
    py.mkdir(sample_dir)

    train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'train'))
    with train_summary_writer.as_default():
        for ep in tqdm.trange(hyper_params['epochs'], desc='Epoch Loop'):
            if ep < ep_cnt:
                continue

            # update epoch counter
            ep_cnt.assign_add(1)

            # train for an epoch
            for A, B in tqdm.tqdm(A_B_dataset, desc='Inner Epoch Loop', total=len_dataset):

                G_loss_dict, D_loss_dict = train_step(A, B)

                # # summary
                tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
                tl.summary(D_loss_dict, step=G_optimizer.iterations, name='D_losses')
                tl.summary({'learning rate': G_lr_scheduler.current_learning_rate}, step=G_optimizer.iterations,
                           name='learning rate')
                if Step:
                    Step += 1

                # sample
                sampling = True
                if sampling:
                    num = 50
                    if hyper_params['device'] == 'A100':
                        num = 10
                    if G_optimizer.iterations.numpy() % num == 0:
                        A, B = next(test_iter)
                        A2B, B2A, A2B_mask, A2B2A, B2A2B, B2A2B_mask, m, n, A2A, A2A_mask = sample(A, B)
                        metrics_info, model = Validation(G_A2B, A_mask_dataset)
                        m_iou = metrics_info[-1]
                        tl.summary({'Metrics/m_IoU': tf.convert_to_tensor(m_iou)}, step=G_optimizer.iterations, name='m_IoU')
                        tl.summary({'Metrics/acc': tf.convert_to_tensor(metrics_info[1])}, step=G_optimizer.iterations,
                                   name='acc')
                        tl.summary({'Metrics/m_Pr': tf.convert_to_tensor(metrics_info[2])}, step=G_optimizer.iterations,
                                   name='m_Pr')
                        tl.summary({'Metrics/m_Re': tf.convert_to_tensor(metrics_info[3])}, step=G_optimizer.iterations,
                                   name='m_Re')
                        tl.summary({'Metrics/m_F1': tf.convert_to_tensor(metrics_info[4])}, step=G_optimizer.iterations,
                                   name='m_F1')
                        if m_iou > 0.4:

                            model.save(os.path.join(output_dir, 'save_model',
                                                    '{}-{}-{}/'.format(ep, G_optimizer.iterations.numpy(), m_iou)))
                        img = im.immerge(np.concatenate(
                            [A[0:1, :, :, :], A2B, A2B_mask, A2B2A, m, A2A,
                             B[0:1, :, :, :], B2A, B2A2B_mask, B2A2B, n, A2A_mask],
                            axis=0),
                            n_rows=2)
                        im.imwrite(img, py.join(sample_dir, 'iter-%09d.jpg' % G_optimizer.iterations.numpy()))
                        if hyper_params['device'] == 'A100':
                            experiment.log_image(img, 'iter-%09d.jpg' % G_optimizer.iterations.numpy())
                        print(G_loss_dict, D_loss_dict)

            # save checkpoint
            checkpoint.save(ep)


def validation():
    model = keras.models.load_model(r'P:\GAN_CheckPoint\G_A2B/')
    initial_learning_rate = 5e-5
    optimizer = keras.optimizers.RMSprop(initial_learning_rate)
    model = keras.models.Model(inputs=model.inputs, outputs=model.outputs[0][:, :, :, 0:1])
    model.compile(optimizer=optimizer,
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy', Metrics.M_Precision, Metrics.M_Recall, Metrics.M_F1, Metrics.M_IOU])

    model.evaluate(A_mask_dataset)


if training:

    if experiment_button:
        step = 1
        with experiment.train():
            train(Step=step)
        experiment.end()
    else:
        train()
