import functools
import os
import shutil
import cv2
import datetime

import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import tqdm
import matplotlib.pyplot as plt
import data
import module

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--dataset', default='crack')
py.arg('--datasets_dir', default='datasets')
py.arg('--load_size', type=int, default=227)  # load image to this size
py.arg('--crop_size', type=int, default=224)  # then crop to this size
py.arg('--batch_size', type=int, default=3)
py.arg('--epochs', type=int, default=10)
py.arg('--epoch_decay', type=int, default=5)  # epoch to start decaying learning rate
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--adversarial_loss_mode', default='lsgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])
py.arg('--gradient_penalty_weight', type=float, default=10.0)
py.arg('--cycle_loss_weight', type=float, default=10.0)
py.arg('--identity_loss_weight', type=float, default=10.0)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
py.arg('--lambda_reg', type=float, default=1e-6)
py.arg('--starting_rate', type=float, default=0.01, help='Set the lambda weight between GAN loss and '
                                                         'Recon loss during curriculum period at the '
                                                         'beginning. We used the 0.01 weight.')
py.arg("--mode", default='client')
py.arg("--port", default=52162)
args = py.args()

# output_dir
a = str(datetime.datetime.now())
b = list(a)
b[10] = '-'
b[13] = '-'
b[16] = '-'
output_dir = ''.join(b)
output_dir = r'E:/Cycle_GAN/output/{}'.format(output_dir)
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

# ==============================================================================
# =                                    data                                    =
# ==============================================================================

# 这位大佬Dataset制作好复杂哦
A_img_paths = py.glob(py.join(args.datasets_dir, args.dataset, 'trainA'), '*.jpg')
B_img_paths = py.glob(py.join(args.datasets_dir, args.dataset, 'trainB'), '*.jpg')
A_B_dataset, len_dataset = data.make_zip_dataset(A_img_paths, B_img_paths, args.batch_size, args.load_size,
                                                 args.crop_size, training=True, repeat=False)

# 用来保存假样本，但是有什么用呢？
A2B_pool = data.ItemPool(args.pool_size)
B2A_pool = data.ItemPool(args.pool_size)

# 测试样本，可以略过
A_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'Positive_mini'), '*.jpg')
B_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'Negative_mini'), '*.jpg')
A_B_dataset_test, _ = data.make_zip_dataset(
    A_img_paths_test, B_img_paths_test, args.batch_size, args.load_size, args.crop_size, training=False, repeat=True)

# ==============================================================================
# =                                   models                                   =
# ==============================================================================

G_A2B = module.AttentionCycleGAN_v1_Generator(input_shape=(args.crop_size, args.crop_size, 3), attention=True)
G_B2A = module.AttentionCycleGAN_v1_Generator(input_shape=(args.crop_size, args.crop_size, 3), attention=False)

# G_A2B = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3), attention=True)
# G_B2A = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3), attention=False)
# G_A2B = tf.keras.models.load_model(r'A2B.h5')
# G_B2A = tf.keras.models.load_model(r'B2A.h5')

D_A = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, 3))
D_B = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, 3))

d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)
cycle_loss_fn = tf.losses.MeanAbsoluteError()
identity_loss_fn = tf.losses.MeanAbsoluteError()


def ssim_loss(y_pred, y_true):
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    return -1 * tf.reduce_sum(-1 + tf.image.ssim(y_pred, y_true, max_val=1.0, filter_size=11,
                                                 filter_sigma=1.5, k1=0.01, k2=0.03))


G_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
D_lr_scheduler = module.LinearDecay(args.lr * 0.1, args.epochs * len_dataset, args.epoch_decay * len_dataset)
G_optimizer = keras.optimizers.RMSprop(learning_rate=G_lr_scheduler)
D_optimizer = keras.optimizers.RMSprop(learning_rate=D_lr_scheduler)


# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

@tf.function
def train_G(A_Generator, B_Generator):
    with tf.GradientTape() as t:
        A2B_Result = G_A2B(A_Generator, training=True)[0]
        A2B_m = G_A2B(A_Generator, training=True)[2]
        A2B_n = G_A2B(A_Generator, training=True)[3]
        B2A_Result = G_B2A(B_Generator, training=True)
        A2B2A_Result = G_B2A(A2B_Result, training=True)
        B2A2B_Result = G_A2B(B2A_Result, training=True)[0]
        B2A2B_m = G_A2B(B2A_Result, training=True)[2]
        B2A2B_n = G_A2B(B2A_Result, training=True)[3]
        A2A = G_B2A(A_Generator, training=True)
        B2B = G_A2B(B_Generator, training=True)[0]
        B2B_m = G_A2B(B_Generator, training=True)[2]
        B2B_n = G_A2B(B_Generator, training=True)[3]
        # A2B, mask_B, temp_B = G_A2B(A, training=True)
        # B2A, mask_A, temp_A = G_B2A(B, training=True)
        # A2B2A, _, _ = G_B2A(A2B, training=True)
        # B2A2B, _, _ = G_A2B(B2A, training=True)
        # A2A, _, _ = G_B2A(A, training=True)
        # B2B, _, _ = G_A2B(B, training=True)

        A2B_d_logits = D_B(A2B_Result, training=True)
        B2A_d_logits = D_A(B2A_Result, training=True)

        A2B_g_loss = g_loss_fn(A2B_d_logits)
        B2A_g_loss = g_loss_fn(B2A_d_logits)
        A2B2A_cycle_loss = cycle_loss_fn(A_Generator, A2B2A_Result)
        B2A2B_cycle_loss = cycle_loss_fn(B_Generator, B2A2B_Result)
        A2A_id_loss = identity_loss_fn(A_Generator, A2A)
        B2B_id_loss = identity_loss_fn(B_Generator, B2B)

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

        G_loss = 2 * (A2B_g_loss + B2A_g_loss) + (A2B2A_cycle_loss + B2A2B_cycle_loss) * args.cycle_loss_weight + (
                A2A_id_loss + B2B_id_loss) * args.identity_loss_weight + (s_loss_1 + s_loss_2 + s_loss_3)

    G_grad = t.gradient(G_loss, G_A2B.trainable_variables + G_B2A.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables + G_B2A.trainable_variables))

    return A2B_Result, B2A_Result, {'A2B_g_loss': A2B_g_loss,
                                    'B2A_g_loss': B2A_g_loss,
                                    'A2B2A_cycle_loss': A2B2A_cycle_loss,
                                    'B2A2B_cycle_loss': B2A2B_cycle_loss,
                                    'A2A_id_loss': A2A_id_loss,
                                    'B2B_id_loss': B2B_id_loss,
                                    's_loss': s_loss_1 + s_loss_2 + s_loss_3
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
        D_A_gp = gan.gradient_penalty(functools.partial(D_A, training=True), A_True, B2A_Fake, mode=args.gradient_penalty_mode)
        D_B_gp = gan.gradient_penalty(functools.partial(D_B, training=True), B_True, A2B_Fake, mode=args.gradient_penalty_mode)

        D_loss = (A_d_loss + B2A_d_loss) + (B_d_loss + A2B_d_loss) + (D_A_gp + D_B_gp) * args.gradient_penalty_weight

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
    return A2B_Test[0:1, :, :, :], B2A_Test[0:1, :, :, :], A2B_mask_Test[0:1, :, :, :], A2B2A_Test[0:1, :, :, :], \
           B2A2B_Test[0:1, :, :, :], B2A2B_mask_Test[0:1, :, :, :], m_Test[0:1, :, :, :], n_Test[0:1, :, :, :]


# ==============================================================================
# =                                    run                                     =
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

module_dir = py.join(output_dir, 'module_code')
py.mkdir(module_dir)

# 本地复制源代码，便于复现(模型文件、数据文件、训练文件、测试文件)
# 冷代码
shutil.copytree('imlib', module_dir)
shutil.copytree('pylib', module_dir)
shutil.copytree('tf2gan', module_dir)
shutil.copytree('tf2gan', module_dir)

# 个人热代码
shutil.copy('module.py', module_dir)
shutil.copy('data.py', module_dir)
shutil.copy('train_comet.py', module_dir)
shutil.copy('train_comet.py', module_dir)


# summary
training = True
if training:
    train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'train'))

    # sample
    test_iter = iter(A_B_dataset_test)
    sample_dir = py.join(output_dir, 'samples_training')
    py.mkdir(sample_dir)

    # main loop
    with train_summary_writer.as_default():
        for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):
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

                # sample
                sampling = True
                if sampling:
                    if G_optimizer.iterations.numpy() % 100 == 0:
                        A, B = next(test_iter)
                        A2B, B2A, A2B_mask, A2B2A, B2A2B, B2A2B_mask, m, n = sample(A, B)
                        img = im.immerge(np.concatenate(
                            [A[0:1, :, :, :], A2B, A2B_mask, A2B2A, m, B[0:1, :, :, :], B2A, B2A2B_mask, B2A2B, n],
                            axis=0),
                            n_rows=2)
                        im.imwrite(img, py.join(sample_dir, 'iter-%09d.jpg' % G_optimizer.iterations.numpy()))
                        print(G_loss_dict, D_loss_dict)

            # save checkpoint
            checkpoint.save(ep)

# %%
test = False
if test:
    G_A2B = checkpoint.checkpoint.G_A2B

    mask_layer = G_A2B.output
    mask_model = tf.keras.Model(inputs=G_A2B.input, outputs=mask_layer)
    path = r'C:\Users\liuye\Desktop\Slender Cracks Positive/'
    for i in os.listdir(path):
        filepath = path + i
        o_img_file = cv2.imread(filepath)
        img_file = ((o_img_file - 127.5) / 127.5).reshape((1, 227, 227, 3))
        image = tf.convert_to_tensor(img_file, dtype=tf.float32)
        mask = mask_model.predict(image)[1]
        # result = G_A2B.predict(image)
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        plt.imsave('./result_validation/{}_2.jpg'.format(i[:-4]), mask.reshape(227, 227, 3))
    # print(mask.shape)

    # def plot_without_axis(plot, channel):
    #     plt.imshow(plot.reshape(227, 227, channel))
    #     plt.axis('off')
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.show()
    # plot_without_axis(img_file, 3)
    # plot_without_axis(mask[0], 1)
    # a = 1

# A, B = next(test_iter)
# A2B, B2A, A2B2A, B2A2B = sample(A, B)
# img = im.immerge(np.concatenate([A, A2B, A2B2A, B, B2A, B2A2B], axis=0), n_rows=2)
# im.imwrite(img, py.join(sample_dir, '0111a.jpg'))


# B = cv2.imread(r'I:\Image Processing\Mix_img\80\image\00001_img.png')
# B = B.reshape((1, 227, 227, 3))
# img = tf.image.resize(B, [227, 227])  # or img = tf.image.resize(img, [load_size, load_size]); img = tl.center_crop(
# # img, crop_size)
# img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
# img = img * 2 - 1
# A2B = G_A2B(img, training=False)
# A2B2A = G_B2A(A2B, training=False)
# # B2A2B = G_A2B(B2A, training=False)
# # B2A2B2A = G_B2A(B2A2B, training=False)
# img = im.immerge(np.concatenate([A2B2A], axis=0), n_rows=1)
# im.imwrite(img, py.join(sample_dir, '0011a.jpg'))
