import tensorflow.keras as keras
import matplotlib.pyplot as plt
from plot import plot_heatmap


# checkpoint
# 目前存在的问题是只能输出最后一个checkpoints
# 2021/9/20 问题已经解决，目前可以输出指定的checkpoint了
# 测试一下git的合并功能
class CheckpointSaver(keras.callbacks.Callback):
    def __init__(self, manager):
        super(CheckpointSaver, self).__init__()
        self.manager = manager

    def on_epoch_end(self, epoch, logs=None):
        self.manager.save()


lr_schedule_E = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.96,
)

# 2、PiecewiseConstantDecay(分段常数衰减)
boundaries = [100000, 110000]
values = [1.0, 0.5, 0.1]
lr_schedule_P = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=boundaries,
    values=values
)

# 3、PolynomialDecay(多项式衰减)
starter_learning_rate = 0.1
end_learning_rate = 0.01
decay_steps = 10000
lr_schedule_Pi = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=starter_learning_rate,
    decay_steps=decay_steps,
    end_learning_rate=end_learning_rate,
    power=0.5
)

# 4、InverseTimeDecay(逆时间衰减)
initial_learning_rate = 0.1
decay_steps = 10000
decay_rate = 0.5
lr_schedule_I = keras.optimizers.schedules.InverseTimeDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate
)
#
# 监视验证集损失函数动态调整

DynamicLearningRate = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='auto',
    min_delta=0.0001, cooldown=0, min_lr=1e-8
)

EarlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')


# 2、在每个Epoch后输出X张预测图片
class CheckpointPlot(keras.callbacks.Callback):
    def __init__(self, generator, path, num_img=1):
        super(CheckpointPlot, self).__init__()
        self.generator = generator
        self.father_path = path
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        for i in range(self.num_img):
            raw_tuples = self.generator.__next__()
            raw_image = raw_tuples[0]
            raw_label = raw_tuples[1]
            predict_array = self.model.predict(raw_image)
            save_predict_path = self.father_path + '{}_{}_Predict_'.format(str(epoch), i) + '.png'
            save_true_path = self.father_path + '{}_{}_True_'.format(str(epoch), i) + '.png'
            plt.figure()
            plot_heatmap(save_path=save_predict_path, predict_array=predict_array[-1])
            plt.figure()
            plot_heatmap(save_path=save_true_path, predict_array=raw_label[-1])
