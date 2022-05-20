# -*- coding: utf-8 -*-
# @Time    : 2022/5/2 14:53
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : train_Supervised_comet.py
# @Software: PyCharm
from comet_ml import Experiment

import os
import json
import shutil
import random
import datetime

import keras.losses
import keras.optimizers
import model_profiler

import pylib as py
from Metrics import *
from Callback import *
import Cycle_data as data
from builders import builder

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

experiment_button = False
training = True
experiment = object

if experiment:
    pass

if experiment_button:
    experiment = Experiment(
        api_key="sDV9A5CkoqWZuJDeI9JbJMRvp",
        project_name="crack_supervised",
        workspace="lovingthresh",
        auto_param_logging=True,
    )


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.random.set_seed(seed)  # tf cpu fix seed
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first


setup_seed(42)
# ----------------------------------------------------------------------
#                               parameter
# ----------------------------------------------------------------------

hyper_params = {
    'ex_number': 'Supervised_crack_3080Ti',
    'Model': 'UNet',
    'Base_Model': 'VGG16',
    'loss': 'Binary_Crossentropy',
    'repeat_num': 1,
    'device': '3080Ti',
    'data_type': 'crack',
    'datasets_dir': r'datasets',
    'load_size': 224,
    'crop_size': 224,
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 1e-4,
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
    hyper_params['batch_size'] = 128
    repeat_num = 1
elif hyper_params['device'] == '3080Ti' or '3090':
    output_dir = r'E:/Cycle_GAN/output/{}'.format(output_dir)
    if hyper_params['device'] == '3080Ti':
        hyper_params['batch_size'] = 32
        repeat_num = 1
    else:
        hyper_params['batch_size'] = 32
        repeat_num = 1

hyper_params['repeat_num'] = repeat_num
hyper_params['output_dir'] = output_dir
hyper_params['ex_date'] = a[:10]

# ----------------------------------------------------------------------
#                               dataset
# ----------------------------------------------------------------------

# Train Dataset
A_img_train_paths = py.glob(py.join(hyper_params['datasets_dir'], hyper_params['data_type'], 'train_Positive'), '*.jpg')
A_mask_train_paths = py.glob(py.join(hyper_params['datasets_dir'], hyper_params['data_type'], 'train_Positive_mask'),
                             '*.jpg')
A_train_dataset, len_train_dataset = data.make_zip_dataset(A_img_train_paths, A_mask_train_paths,
                                                           hyper_params['batch_size'],
                                                           hyper_params['load_size'],
                                                           hyper_params['crop_size'],
                                                           repeat=hyper_params['repeat_num'],
                                                           training=True,
                                                           random_fn=True,
                                                           shuffle=False,
                                                           mask=True)

# Validation Dataset
A_img_val_paths = py.glob(py.join(hyper_params['datasets_dir'], hyper_params['data_type'], 'val_Positive'), '*.jpg')
A_mask_val_paths = py.glob(py.join(hyper_params['datasets_dir'], hyper_params['data_type'], 'val_Positive_mask'),
                           '*.jpg')
A_val_dataset, len_val_dataset = data.make_zip_dataset(A_img_val_paths, A_mask_val_paths,
                                                       hyper_params['batch_size'],
                                                       hyper_params['load_size'],
                                                       hyper_params['crop_size'],
                                                       repeat=1,
                                                       training=False,
                                                       random_fn=False,
                                                       shuffle=False,
                                                       mask=True)

# Validation Dataset
A_img_test_paths = py.glob(py.join(hyper_params['datasets_dir'], hyper_params['data_type'], 'test_Positive'), '*.jpg')
A_mask_test_paths = py.glob(py.join(hyper_params['datasets_dir'], hyper_params['data_type'], 'test_Positive_mask'),
                            '*.jpg')
A_test_dataset, len_test_dataset = data.make_zip_dataset(A_img_test_paths, A_mask_test_paths,
                                                         hyper_params['batch_size'],
                                                         hyper_params['load_size'],
                                                         hyper_params['crop_size'],
                                                         repeat=1,
                                                         training=False,
                                                         random_fn=False,
                                                         shuffle=False,
                                                         mask=True)

# ----------------------------------------------------------------------
#                               model
# ----------------------------------------------------------------------

model, base_model = builder(1, input_size=(hyper_params['crop_size'], hyper_params['crop_size']),
                            model=hyper_params['Model'],
                            base_model=hyper_params['Base_Model'])
model.summary()
profile = model_profiler.model_profiler(model, hyper_params['batch_size'])
print(profile)
if experiment_button:
    hyper_params['ex_key'] = experiment.get_key()
    experiment.log_parameters(hyper_params)
    experiment.set_name('{}-{}'.format(hyper_params['ex_date'], hyper_params['ex_number']))
    experiment.add_tag('Supervised')
    experiment.add_tag(hyper_params['Model'])

# ----------------------------------------------------------------------
#                               output
# ----------------------------------------------------------------------
py.mkdir(output_dir)
module_dir = py.join(output_dir, 'module_code')
py.mkdir(module_dir)
os.mkdir(os.path.join(output_dir, 'save_model'))

# 本地复制源代码，便于复现(模型文件、数据文件、训练文件、测试文件)
# 冷代码
shutil.copytree('imlib', '{}/{}'.format(module_dir, 'imlib'))
shutil.copytree('pylib', '{}/{}'.format(module_dir, 'pylib'))
shutil.copytree('models', '{}/{}'.format(module_dir, 'models'))
shutil.copytree('builders', '{}/{}'.format(module_dir, 'builders'))
shutil.copytree('base_models', '{}/{}'.format(module_dir, 'base_models'))

# 个人热代码
shutil.copy('plot.py', module_dir)
shutil.copy('Metrics.py', module_dir)
shutil.copy('Callback.py', module_dir)
shutil.copy('Cycle_data.py', module_dir)
shutil.copy('train_Supervised_comet.py', module_dir)

# 云端上次源代码
if experiment_button:
    experiment.log_asset_folder('imlib', log_file_name=True)
    experiment.log_asset_folder('pylib', log_file_name=True)
    experiment.log_asset_folder('models', log_file_name=True)
    experiment.log_asset_folder('builders', log_file_name=True)
    experiment.log_asset_folder('base_models', log_file_name=True)

    experiment.log_code('plot.py')
    experiment.log_code('Metrics.py')
    experiment.log_code('Callback.py')
    experiment.log_code('Cycle_data.py')
    experiment.log_code('train_Supervised_comet.py')

# ----------------------------------------------------------------------
#                                Callback
# ----------------------------------------------------------------------

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='{}/tensorboard/'.format(output_dir))
checkpoint = tf.keras.callbacks.ModelCheckpoint('{}/checkpoint/'.format(output_dir) +
                                                'ep{epoch:03d}-val_loss{val_loss:.3f}/',
                                                monitor='val_accuracy', verbose=0,
                                                save_best_only=False, save_weights_only=False,
                                                mode='auto', period=1)
os.makedirs('{}/plot/'.format(output_dir))
plot_path = '{}/plot/'.format(output_dir)
checkpointplot = CheckpointPlot(generator=A_val_dataset, path=plot_path)

with open('{}/hyper_params.json'.format(output_dir), 'w') as fp:
    json.dump(hyper_params, fp)

# ----------------------------------------------------------------------
#                               train
# ----------------------------------------------------------------------


model.compile(optimizer=keras.optimizers.Adam(hyper_params['learning_rate']),
              loss=keras.losses.BinaryCrossentropy(),
              metrics=['accuracy', 'mse', M_Precision, M_Recall, M_F1, M_IOU])


def train():
    model.fit(A_train_dataset,
              epochs=hyper_params['epochs'],
              validation_data=A_val_dataset,
              initial_epoch=0,
              verbose=1,
              callbacks=[tensorboard, checkpoint, EarlyStopping, checkpointplot,
                         DynamicLearningRate])


if experiment_button:
    with experiment.train():
        train()
    experiment.end()
else:
    train()


# ----------------------------------------------------------------------
#                             val_test
# ----------------------------------------------------------------------
def Val_Test(model_path, val_dataset: list):
    val_model = keras.models.load_model(model_path,
                                        custom_objects={'M_Precision': M_Precision,
                                                        'M_Recall': M_Recall,
                                                        'M_F1': M_F1,
                                                        'M_IOU': M_IOU,
                                                        }
                                        )
    for dataset in val_dataset:
        val_model.evaluate(dataset)
