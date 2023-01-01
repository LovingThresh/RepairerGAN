# -*- coding: utf-8 -*-
# @Time    : 2022/7/20 20:12
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : config.py
# @Software: PyCharm
norm_cfg = dict(type='BN', requires_grad=True)  # 分割框架通常使用 SyncBN
model = dict(
    type='EncoderDecoder',  # 分割器(segmentor)的名字
    # pretrained='open-mmlab://resnet50_v1c',  # 将被加载的 ImageNet 预训练主干网络
    pretrained=None,
    backbone=dict(
        type='ResNetV1c',  # 主干网络的类别。 可用选项请参考 mmseg/models/backbones/resnet.py
        depth=50,  # 主干网络的深度。通常为 50 和 101。
        num_stages=4,  # 主干网络状态(stages)的数目，这些状态产生的特征图作为后续的 head 的输入。
        out_indices=(0, 1, 2, 3),  # 每个状态产生的特征图输出的索引。
        dilations=(1, 1, 2, 4),  # 每一层(layer)的空心率(dilation rate)。
        strides=(1, 2, 1, 1),  # 每一层(layer)的步长(stride)。
        norm_cfg=dict(  # 归一化层(norm layer)的配置项。
            type='BN',  # 归一化层的类别。通常是 SyncBN。
            requires_grad=True),  # 是否训练归一化里的 gamma 和 beta。
        norm_eval=False,  # 是否冻结 BN 里的统计项。
        style='pytorch',  # 主干网络的风格，'pytorch' 意思是步长为2的层为 3x3 卷积， 'caffe' 意思是步长为2的层为 1x1 卷积。
        contract_dilation=True),  # 当空洞 > 1, 是否压缩第一个空洞层。
    decode_head=dict(
        type='PSPHead',  # 解码头(decode head)的类别。 可用选项请参考 mmseg/models/decode_heads。
        in_channels=2048,  # 解码头的输入通道数。
        in_index=3,  # 被选择的特征图(feature map)的索引。
        channels=512,  # 解码头中间态(intermediate)的通道数。
        pool_scales=(1, 2, 3, 6),  # PSPHead 平均池化(avg pooling)的规模(scales)。 细节请参考文章内容。
        dropout_ratio=0.1,  # 进入最后分类层(classification layer)之前的 dropout 比例。
        num_classes=2,  # 分割前景的种类数目。 通常情况下，cityscapes 为19，VOC为21，ADE20k 为150。
        norm_cfg=dict(type='BN', requires_grad=True),  # 归一化层的配置项。
        align_corners=False,  # 解码里调整大小(resize)的 align_corners 参数。
        loss_decode=dict(  # 解码头(decode_head)里的损失函数的配置项。
            type='CrossEntropyLoss',  # 在分割里使用的损失函数的类别。
            use_sigmoid=False,  # 在分割里是否使用 sigmoid 激活。
            loss_weight=1.0)),  # 解码头里损失的权重。
    auxiliary_head=dict(
        type='FCNHead',  # 辅助头(auxiliary head)的种类。可用选项请参考 mmseg/models/decode_heads。
        in_channels=1024,  # 辅助头的输入通道数。
        in_index=2,  # 被选择的特征图(feature map)的索引。
        channels=256,  # 辅助头中间态(intermediate)的通道数。
        num_convs=1,  # FCNHead 里卷积(convs)的数目. 辅助头里通常为1。
        concat_input=False,  # 在分类层(classification layer)之前是否连接(concat)输入和卷积的输出。
        dropout_ratio=0.1,  # 进入最后分类层(classification layer)之前的 dropout 比例。
        num_classes=2,  # 分割前景的种类数目。 通常情况下，cityscapes 为19，VOC为21，ADE20k 为150。
        norm_cfg=dict(type='BN', requires_grad=True),  # 归一化层的配置项。
        align_corners=False,  # 解码里调整大小(resize)的 align_corners 参数。
        loss_decode=dict(  # 辅助头(auxiliary head)里的损失函数的配置项。
            type='CrossEntropyLoss',  # 在分割里使用的损失函数的类别。
            use_sigmoid=False,  # 在分割里是否使用 sigmoid 激活。
            loss_weight=0.4)))  # 辅助头里损失的权重。默认设置为0.4。
train_cfg = dict()  # train_cfg 当前仅是一个占位符。
test_cfg = dict(
    mode='whole')  # 测试模式， 选项是 'whole' 和 'sliding'. 'whole': 整张图像全卷积(fully-convolutional)测试。 'sliding': 图像上做滑动裁剪窗口(sliding crop window)。
dataset_type = 'CrackDataset'  # 数据集类型，这将被用来定义数据集。
data_root = '../datasets/crack/'  # 数据的根路径。
img_norm_cfg = dict(  # 图像归一化配置，用来归一化输入的图像。
    mean=[123.675, 116.28, 103.53],  # 预训练里用于预训练主干网络模型的平均值。
    std=[58.395, 57.12, 57.375],  # 预训练里用于预训练主干网络模型的标准差。
    to_rgb=True)  # 预训练里用于预训练主干网络的图像的通道顺序。
crop_size = (224, 224)  # 训练时的裁剪大小
train_pipeline = [  # 训练流程
    dict(type='LoadImageFromFile'),  # 第1个流程，从文件路径里加载图像。
    dict(type='LoadAnnotations'),  # 第2个流程，对于当前图像，加载它的注释信息。
    dict(type='Resize',  # 变化图像和其注释大小的数据增广的流程。
         img_scale=(224, 224)),  # 图像的最大规模。
    dict(type='RandomCrop',  # 随机裁剪当前图像和其注释大小的数据增广的流程。
         crop_size=(224, 224)),  # 随机裁剪图像生成 patch 的大小。
    dict(
        type='Normalize',  # 归一化当前图像的数据增广的流程。
        mean=[123.675, 116.28, 103.53],  # 这些键与 img_norm_cfg 一致，因为 img_norm_cfg 被
        std=[58.395, 57.12, 57.375],  # 用作参数。
        to_rgb=True),
    dict(type='DefaultFormatBundle'),  # 流程里收集数据的默认格式捆。
    dict(type='Collect',  # 决定数据里哪些键被传递到分割器里的流程。
         keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),  # 第1个流程，从文件路径里加载图像。
    dict(
        type='MultiScaleFlipAug',  # 封装测试时数据增广(test time augmentations)。
        img_scale=(224, 224),  # 决定测试时可改变图像的最大规模。用于改变图像大小的流程。
        flip=False,  # 测试时是否翻转图像。
        transforms=[
            dict(type='Resize',  # 使用改变图像大小的数据增广。
                 img_scale=(224, 224),
                 keep_ratio=True),  # 是否保持宽和高的比例，这里的图像比例设置将覆盖上面的图像规模大小的设置。
            dict(type='RandomFlip'),  # 考虑到 RandomFlip 已经被添加到流程里，当 flip=False 时它将不被使用。
            dict(
                type='Normalize',  # 归一化配置项，值来自 img_norm_cfg。
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor',  # 将图像转为张量
                 keys=['img']),
            dict(type='Collect',  # 收集测试时必须的键的收集流程。
                 keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,  # 单个 GPU 的 Batch size
    workers_per_gpu=1,  # 单个 GPU 分配的数据加载线程数
    train=dict(  # 训练数据集配置
        type='CrackDataset',  # 数据集的类别, 细节参考自 mmseg/datasets/。
        data_root='../datasets/crack/',  # 数据集的根目录。
        img_dir='img_dir/train_Positive/',  # 数据集图像的文件夹。
        ann_dir='ann_dir/train_Positive_CAM_CRFs_mask/',  # 数据集注释的文件夹。
        pipeline=[  # 流程， 由之前创建的 train_pipeline 传递进来。
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(224, 224)),
            dict(type='RandomCrop', crop_size=(224, 224)),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(  # 验证数据集的配置
        type='CrackDataset',
        data_root='../datasets/crack/',
        img_dir='img_dir/val_Positive/',
        ann_dir='ann_dir/val_true/',
        pipeline=[  # 由之前创建的 test_pipeline 传递的流程。
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(224, 224)),
            dict(type='RandomCrop', crop_size=(224, 224)),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(224, 224),
                flip=False,
                transforms=[
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CrackDataset',
        data_root='../datasets/crack/',
        img_dir='img_dir/test_Positive/',
        ann_dir='ann_dir/test_Positive_CAM_mask/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(224, 224)),
            dict(
                type='MultiScaleFlipAug',
                flip=False,
                transforms=[
                    dict(type='Resize', img_scale=(224, 224), keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(  # 注册日志钩 (register logger hook) 的配置文件。
    interval=50,  # 打印日志的间隔
    hooks=[
        # dict(type='TensorboardLoggerHook')  # 同样支持 Tensorboard 日志
        dict(type='TextLoggerHook', by_epoch=False)
    ])

log_level = 'INFO'  # 日志的级别。
load_from = None  # 从一个给定路径里加载模型作为预训练模型，它并不会消耗训练时间。
resume_from = None  # 从给定路径里恢复检查点(checkpoints)，训练模式将从检查点保存的轮次开始恢复训练。
workflow = [('train',
             1)]  # runner 的工作流程。 [('train', 1)] 意思是只有一个工作流程而且工作流程 'train' 仅执行一次。根据 `runner.max_iters` 工作流程训练模型的迭代轮数为40000次。
cudnn_benchmark = True  # 是否是使用 cudnn_benchmark 去加速，它对于固定输入大小的可以提高训练速度。
optimizer = dict(  # 用于构建优化器的配置文件。支持 PyTorch 中的所有优化器，同时它们的参数与PyTorch里的优化器参数一致。
    type='SGD',
    # 优化器种类，更多细节可参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/optimizer/default_constructor.py#L13。
    lr=0.01,  # 优化器的学习率，参数的使用细节请参照对应的 PyTorch 文档。
    momentum=0.9,  # 动量 (Momentum)
    weight_decay=0.0005)  # SGD 的衰减权重 (weight decay)。
optimizer_config = dict()  # 用于构建优化器钩 (optimizer hook) 的配置文件，执行细节请参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py#L8。
lr_config = dict(
    policy='poly',
    # 调度流程的策略，同样支持 Step, CosineAnnealing, Cyclic 等. 请从 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9 参考 LrUpdater 的细节。
    power=0.9,  # 多项式衰减 (polynomial decay) 的幂。
    min_lr=0.0001,  # 用来稳定训练的最小学习率。
    by_epoch=False)  # 是否按照每个 epoch 去算学习率。
runner = dict(
    type='IterBasedRunner',  # 将使用的 runner 的类别 (例如 IterBasedRunner 或 EpochBasedRunner)。
    max_iters=12000)  # 全部迭代轮数大小，对于 EpochBasedRunner 使用 `max_epochs` 。
checkpoint_config = dict(
    # 设置检查点钩子 (checkpoint hook) 的配置文件。执行时请参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py。
    by_epoch=False,  # 是否按照每个 epoch 去算 runner。
    interval=4000)  # 保存的间隔
evaluation = dict(  # 构建评估钩 (evaluation hook) 的配置文件。细节请参考 mmseg/core/evaluation/eval_hook.py。
    interval=400,  # 评估的间歇点
    metric='mIoU')  # 评估的指标

# 训练结果保存路径
work_dir = 'work/'
