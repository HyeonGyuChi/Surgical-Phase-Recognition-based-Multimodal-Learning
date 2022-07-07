model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowFast',
        pretrained=None,
        resample_rate=4,  # tau
        speed_ratio=4,  # alpha
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
    # cls_head=dict(
    #     type='SlowFastHead',
    #     in_channels=2304,  # 2048+256
    #     num_classes=27,
    #     spatial_type='avg',
    #     dropout_ratio=0.5,
    #     multi_task=False,

    #     loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0)),
    #     # loss_cls=dict(type='CBLoss', loss_weight=1.0, 
    #     #     samples_per_cls=[
    #     #         [154292, 26405, 6034, 15333, 76558, 10274, 1648],
    #     #     ],
    #     #     no_of_classes=[27])),
    # train_cfg = None,
    # test_cfg = dict(average_clips=None))
    cls_head=dict(
        type='SlowFastHead',
        in_channels=2304,  # 2048+256
        num_classes=27,
        spatial_type='avg',
        dropout_ratio=0.5, #),
        loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0,
        class_weight=[1.1219280406844807, 0.9681475764205234, 2.700095993716775, 
        0.9141240990478067, 0.2630276758636177, 1.3230899984498694, 
        0.7860581656049337, 1.0354759361569805, 0.9021204066360199, 
        0.5556540790991905, 4.018654414391012, 1.2661060557235002, 
        6.645501651139689, 0.7116979830072313, 1.0866479455282185, 
        1.7290175515393105, 0.4124589457336612, 1.2908398737972935, 
        1.9176686603241502, 0.48788876865370095, 8.124719629747148, 
        0.8676360619727526, 0.2317482728305169, 1.0, 0.6669441952076479, 
        0.8789846237615558, 1.495703685202791])),
    train_cfg = None,
    test_cfg = dict(average_clips='prob'))#, max_testing_views=8))

dataset_type = 'RawframeDataset'
data_root = '/dataset3/multimodal/gastric/rawframes'
anno_root = '/dataset3/multimodal/gastric'
split = 1
ann_file_train = f'{anno_root}/gastric_train_{split}_rawframes.txt'
ann_file_val = f'{anno_root}/gastric_val_{split}_rawframes.txt'
ann_file_test = f'{anno_root}/gastric_val_{split}_rawframes.txt'
img_norm_cfg = dict(
    mean=[117.1168392375, 75.27494787, 67.37629650299999], 
    std=[60.241352387999996, 51.261253263, 49.192591569], to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=30, num_clips=1),
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
        clip_len=32,
        frame_interval=30,
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
        clip_len=32,
        frame_interval=30,
        num_clips=1,
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
    videos_per_gpu=32,
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root,
        pipeline=test_pipeline))
# optimizer
# optimizer = dict(
#     type='SGD', lr=0.1, momentum=0.9,
#     weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer = dict(
    type='SGD', lr=0.01, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus

optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=34)

total_epochs = 256
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
work_dir = '/code/multimodal/logs/slowfast_gastric_40_no_pre'
# load_from = 'https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r152_4x16x1_256e_kinetics400_rgb/slowfast_r152_4x16x1_256e_kinetics400_rgb_20210122-bdeb6b87.pth'
# load_from = '/raid/pretrained_models/mmaction2/slowfast_r50_256p_8x8x1_256e_kinetics400_rgb_20200810-863812c2.pth'

load_from = None #'https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/slowfast_r50_8x8x1_256e_kinetics400_rgb_20200716-73547d2b.pth'
resume_from = None
find_unused_parameters = False
workflow = [('train', 1)]
cudnn_benchmark = True
gpu_ids = range(0, 1)
