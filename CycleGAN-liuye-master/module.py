import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model


# ==============================================================================
# =                                  networks                                  =
# ==============================================================================
class SELayer(keras.layers.Layer):
    def __init__(self, channels, reduction=16):
        super(SELayer, self).__init__()
        self.channels = channels
        self.reduction = reduction
        self.avg_poll = GlobalAvgPool2D()
        self.sigmoid = tf.keras.activations.sigmoid

    def fc(self, x):
        x = Dense(self.channels // self.reduction, use_bias=False)(x),
        x = ReLU()(x),
        x = Dense(self.channels, use_bias=False)(x)

        return x

    def call(self, inputs, *args, **kwargs):
        b, _, _, c = inputs.shape
        y = self.avg_poll(inputs)
        y = tf.reshape(y, (y.shape[0], 1, 1, y.shape[1]))
        y = self.fc(y)
        y = self.sigmoid(y)
        # mask.shape = (B, 1, 1, C)
        return inputs * y, y


def CBAM_conv_block(inputs, filter_num, reduction_ratio, stride=1, name=None):
    x = inputs
    x = Conv2D(filter_num[0], (1, 1), strides=stride, padding='same', name=name + '_conv1')(x)
    x = BatchNormalization(axis=3, name=name + '_bn1')(x)
    x = Activation('relu', name=name + '_relu1')(x)

    x = Conv2D(filter_num[1], (3, 3), strides=1, padding='same', name=name + '_conv2')(x)
    x = BatchNormalization(axis=3, name=name + '_bn2')(x)
    x = Activation('relu', name=name + '_relu2')(x)

    x = Conv2D(filter_num[2], (1, 1), strides=1, padding='same', name=name + '_conv3')(x)
    x = BatchNormalization(axis=3, name=name + '_bn3')(x)

    # Channel Attention
    avgpool = GlobalAveragePooling2D(name=name + '_channel_avgpool')(x)
    maxpool = GlobalMaxPool2D(name=name + '_channel_maxpool')(x)
    # Shared MLP
    Dense_layer1 = Dense(filter_num[2] // reduction_ratio, activation='relu', name=name + '_channel_fc1')
    Dense_layer2 = Dense(filter_num[2], activation='relu', name=name + '_channel_fc2')
    avg_out = Dense_layer2(Dense_layer1(avgpool))
    max_out = Dense_layer2(Dense_layer1(maxpool))

    channel = keras.layers.add([avg_out, max_out])
    channel = Activation('sigmoid', name=name + '_channel_sigmoid')(channel)
    channel = Reshape((1, 1, filter_num[2]), name=name + '_channel_reshape')(channel)
    channel_out = tf.multiply(x, channel)

    # Spatial Attention
    avgpool = tf.reduce_mean(channel_out, axis=3, keepdims=True, name=name + '_spatial_avgpool')
    maxpool = tf.reduce_max(channel_out, axis=3, keepdims=True, name=name + '_spatial_maxpool')
    spatial = Concatenate(axis=3)([avgpool, maxpool])

    spatial = Conv2D(1, (7, 7), strides=1, padding='same', name=name + '_spatial_conv2d')(spatial)
    spatial_out = Activation('sigmoid', name=name + '_spatial_sigmoid')(spatial)

    CBAM_out = tf.multiply(channel_out, spatial_out)

    # residual connection
    r = Conv2D(filter_num[2], (1, 1), strides=stride, padding='same', name=name + '_residual')(inputs)
    x = keras.layers.multiply([CBAM_out, r])

    return x, CBAM_out



def _get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return keras.layers.BatchNormalization
    elif norm == 'instance_norm':
        return tfa.layers.InstanceNormalization
    elif norm == 'layer_norm':
        return keras.layers.LayerNormalization


def ResnetGenerator(input_shape=(227, 227, 3),
                    output_channels=3,
                    dim=64,
                    n_downsamplings=2,
                    n_blocks=9,
                    norm='instance_norm',
                    attention=False):
    if attention:
        output_channels = output_channels + 1
        # output_channels = output_channels
    Norm = _get_norm_layer(norm)

    # 受保护的用法
    def _residual_block(x):
        dim = x.shape[-1]
        h = x

        # 为什么这里不用padding参数呢？使用到了‘REFLECT’
        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

        h = keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = Norm()(h)

        return keras.layers.add([x, h])

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(dim, 7, padding='valid', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.relu(h)

    # 2
    for _ in range(n_downsamplings):
        dim *= 2
        h = keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 3
    for _ in range(n_blocks):
        h = _residual_block(h)

    # 4
    for _ in range(n_downsamplings):
        dim //= 2
        h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
        h = keras.layers.Conv2D(dim, 3, strides=1, padding='valid', use_bias=False)(h)
        h = keras.layers.Conv2D(dim, 5, strides=1, padding='valid', use_bias=False)(h)
        h = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)
        h = keras.layers.Conv2DTranspose(dim, 3, strides=1, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 5
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(output_channels, 8, padding='valid')(h)
    if not attention:
        h = tf.tanh(h)
        return keras.Model(inputs=inputs, outputs=h)
    # 假如我不添加tanh的话，又会出现报错
    if attention:
        # h = keras.layers.Conv2D(3, (4, 4), (1, 1), "valid")(h)
        # h, attention_mask = CBAM_conv_block(h, [64, 64, 256], 16, name="CBAM")
        # h = keras.layers.Conv2DTranspose(3, (4, 4), (1, 1), "valid")(h)
        # attention_mask = keras.layers.Conv2DTranspose(3, (4, 4), (1, 1), "valid")(attention_mask)
        # attention_mask = keras.layers.Softmax()(attention_mask)
        # h = keras.layers.multiply([h, attention_mask])
        # h = keras.layers.Activation('tanh')(h)
        # attention_mask = keras.layers.Activation('tanh')(attention_mask)
        attention_mask = h[:, :, :, 0:1]
        attention_mask = tf.pad(attention_mask, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        attention_mask = Conv2D(64, (3, 3), (1, 1), 'same', use_bias=False)(attention_mask)
        attention_mask = Norm()(attention_mask)
        attention_mask = tf.nn.relu(attention_mask)
        attention_mask = Conv2D(3, (3, 3), (1, 1), 'valid', use_bias=False)(attention_mask)
        attention_mask = Norm()(attention_mask)
        attention_mask = tf.sigmoid(attention_mask)
        # attention_mask = attention_mask * attention_mask

        # attention_mask = tf.expand_dims(attention_mask, axis=3)
        # attention_mask = tf.concat([attention_mask, attention_mask, attention_mask], axis=3)

        # 上述式子对应的式attention_v2，在此次实验中可以发现裂缝都被涂上了绿色，可以以此来做无监督学习
        # 应该对上述式子进行更进一步的研究
        # attention_mask = tf.sigmoid(h[:, :, :, 0])  # 91

        content_mask = h[:, :, :, 1:]
        h = tf.tanh(tf.tanh(content_mask * attention_mask) + inputs * (1 - attention_mask))

        return keras.Model(inputs=inputs, outputs=[h, attention_mask])


def squeeze_middle2axes_operator(x4d, C, output_size):
    # keras.backend.set_image_data_format('channels_first')
    shape = x4d.shape  # get dynamic tensor shape
    x4d = tf.reshape(x4d, [-1, shape[1], shape[2] * 2, shape[4] * 2])
    return x4d


def squeeze_middle2axes_shape(output_size):
    in_batch, C, in_rows, _, in_cols, _ = output_size

    if None in [in_rows, in_cols]:
        output_shape = (in_batch, C, None, None)
    else:
        output_shape = (in_batch, C, in_rows, in_cols)
    return output_shape


class pixelshuffle(tf.keras.layers.Layer):
    """Sub-pixel convolution layer.
    See https://arxiv.org/abs/1609.05158
    """

    def compute_output_shape(self, input_size):
        r = self.scale
        rrC, H, W = np.array(input_size[1:])
        assert rrC % (r ** 2) == 0
        height = H * r if H is not None else -1
        width = W * r if W is not None else -1

        return input_size[0], rrC // (r ** 2), height, width

    def __init__(self, scale, trainable=False, **kwargs):
        self.scale = scale
        super().__init__(trainable=trainable, **kwargs)

    @tf.autograph.experimental.do_not_convert
    def call(self, t, *args, **kwargs):
        upscale_factor = self.scale
        input_size = t.shape.as_list()
        dimensionality = len(input_size) - 2
        new_shape = self.compute_output_shape(input_size)
        C = new_shape[1]

        output_size = new_shape[2:]
        x = [upscale_factor] * dimensionality
        old_h = input_size[-2] if input_size[-2] is not None else -1
        old_w = input_size[-1] if input_size[-1] is not None else -1

        shape = t.shape
        t = tf.reshape(t, [-1, C, x[0], x[1], shape[-2], shape[-1]])

        perms = [0, 1, 5, 2, 4, 3]
        t = tf.transpose(t, perm=perms)

        shape = t.shape
        t = Lambda(squeeze_middle2axes_operator, output_shape=squeeze_middle2axes_shape,
                   arguments={'C': C, 'output_size': shape})(t)
        t = tf.transpose(t, [0, 1, 3, 2])
        return t


def AttentionCycleGAN_v1_Generator(input_shape=(227, 227, 3), output_channel=3,
                                   n_downsampling=2, n_ResBlock=9,
                                   norm='batch_norm', attention=False):
    Norm = _get_norm_layer(norm)
    a = keras.Input(shape=input_shape)
    h = tf.pad(a, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')

    def model_layer_1(y):
        y = Conv2D(64, (7, 7), (1, 1), 'valid')(y)
        y = Norm()(y)
        y = ReLU()(y)
        return y

    h = model_layer_1(h)

    n_downsampling = n_downsampling
    n_ResBlock = n_ResBlock

    if attention:
        output_channel = output_channel + 1

    for i in range(n_downsampling):
        mult = 2 ** i

        def model_layer_2(y):
            y = Conv2D(64 * mult * 2, (3, 3), (2, 2), 'same')(y)
            y = Norm()(y)
            y = ReLU()(y)
            return y

        h = model_layer_2(h)

    mult = 2 ** n_downsampling

    for i in range(n_ResBlock):
        x = h

        def model_layer_3(y):
            y = Conv2D(64 * mult, (3, 3), padding='valid')(y)
            y = Norm()(y)
            return y

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = model_layer_3(h)
        h = ReLU()(h)
        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = model_layer_3(h)

        h = keras.layers.add([x, h])

    upsampling = n_downsampling

    for i in range(upsampling):
        mult = 2 ** (n_downsampling - 1)

        def model_layer_4(y):
            y = tf.pad(y, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
            y = Conv2D(int(64 * mult / 2), (5, 5), (1, 1))(y)
            y = Norm()(y)
            y = ReLU()(y)
            y = Conv2D(int(64 * mult / 2) * 4, (3, 3), (1, 1))(y)
            y = tf.transpose(y, [0, 3, 1, 2])
            y = pixelshuffle(2)(y)
            y = Norm()(y)
            y = ReLU()(y)
            y = tf.transpose(y, [0, 2, 3, 1])

            return y

        h = model_layer_4(h)

        # def model_layer_4(y):
        #     y = Conv2DTranspose(64 * mult / 2, (3, 3), (2, 2), 'same')(y)
        #     y = Norm()(y)
        #     y = ReLU()(y)
        #     return y
        #
        # h = model_layer_4(h)
        # mult = mult / 2

    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = Conv2D(output_channel, (8, 8), (1, 1), 'valid')(h)
    h = tf.tanh(h)
    result_layer = h

    if attention:
        attention_mask = h[:, :, :, :1]
        attention_mask = tf.pad(attention_mask, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        attention_mask = Conv2D(64, (3, 3), (1, 1), 'same', use_bias=False)(attention_mask)
        attention_mask = Norm()(attention_mask)
        attention_mask = tf.nn.relu(attention_mask)
        attention_mask = Conv2D(1, (3, 3), (1, 1), 'valid', use_bias=False)(attention_mask)
        attention_mask = Norm()(attention_mask)
        attention_mask = tf.sigmoid(attention_mask)

        content_mask = h[:, :, :, 1:]
        attention_mask = tf.concat([attention_mask, attention_mask, attention_mask], axis=3)
        result_layer = content_mask * attention_mask + a * (1 - attention_mask)

        return keras.Model(inputs=a, outputs=[result_layer, attention_mask, content_mask])
    return keras.Model(inputs=a, outputs=result_layer)


def ConvDiscriminator(input_shape=(256, 256, 3),
                      dim=64,
                      n_downsamplings=3,
                      norm='instance_norm'):
    dim_ = dim
    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 8)
        h = keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.leaky_relu(h, alpha=0.2)

    # 2
    dim = min(dim * 2, dim_ * 8)
    h = keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    # 3
    h = keras.layers.Conv2D(1, 4, strides=1, padding='same')(h)

    return keras.Model(inputs=inputs, outputs=h)


# ==============================================================================
# =                          learning rate scheduler                           =
# ==============================================================================

class LinearDecay(keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate * (
                    1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate
