model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowFast',
        pretrained=None,
        resample_rate=2,  # tau
        speed_ratio=2,  # alpha
        channel_ratio=8,  # beta_inv
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            norm_eval=False),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            norm_eval=False)),
    cls_head=dict(
        type='MultiTaskHead',
        in_channels=2304,  # 2048+256
        num_classes=[3,13,7,7],
        spatial_type='avg',
        dropout_ratio=0.5,
        multi_task=True,
        loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0)),
    train_cfg = None,
    test_cfg = dict(average_clips='prob'))

dataset_type = 'RawframeDataset'
data_root = '/raid/datasets/public/PETRAW/'
data_root_val = '/raid/datasets/public/PETRAW/'
name = 'multi_task'
split = 1
ann_file_train = f'/raid/datasets/public/PETRAW/mmaction/petraw_{name}_train_split_{split}_rawframes.txt'
ann_file_val = f'/raid/datasets/public/PETRAW/mmaction/petraw_{name}_val_split_{split}_rawframes.txt'
ann_file_test = f'/raid/datasets/public/PETRAW/mmaction/petraw_{name}_val_split_{split}_rawframes.txt'
img_norm_cfg = dict(
    mean=[117.1168392375, 75.27494787, 67.37629650299999], 
    std=[60.241352387999996, 51.261253263, 49.192591569], to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=1, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=1,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=2,
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        filename_tmpl='img_{:06}.jpg',
        multi_task=True,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        filename_tmpl='img_{:06}.jpg',
        multi_task=True,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        filename_tmpl='img_{:06}.jpg',
        multi_task=True,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD', lr=0.1, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=34)
total_epochs = 50
checkpoint_config = dict(interval=5)

evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        #    dict(type='TensorboardLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
# work_dir = f'/raid/results/phase_recognition/mmaction/petraw/{name}/test'
work_dir = '/code/multimodal/logs/multi_task_test'
# load_from = 'https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r152_4x16x1_256e_kinetics400_rgb/slowfast_r152_4x16x1_256e_kinetics400_rgb_20210122-bdeb6b87.pth'
# load_from = '/raid/pretrained_models/mmaction2/slowfast_r50_256p_8x8x1_256e_kinetics400_rgb_20200810-863812c2.pth'

load_from = None
resume_from = None
find_unused_parameters = False
workflow = [('train', 1)]
cudnn_benchmark = True
gpu_ids = range(0, 1)