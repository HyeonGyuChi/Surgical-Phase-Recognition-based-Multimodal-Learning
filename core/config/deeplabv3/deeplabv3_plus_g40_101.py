# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
classes=32
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='ASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        num_classes=classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# dataset settings
dataset_type = 'HsdbDataset'
data_root = '/dataset3/multimodal/'

######## Choose Data Split ########
option = 'real' #MIX_real  #DRreal   #SEANMIX_real  #SEAN_RUSreal TAG_RUSreal, #cpv3_real
split_num, gast_num = '56', '40'  
###################################

train_img_dir = 'gastrectomy-'+gast_num+'/images'
train_ann_dir = 'gastrectomy-'+gast_num+'/annotations/semantic_segmentation' #/gastrec'+gast_num+'_semantic_mask_train' + split_num
if gast_num == '100':
    prefix = '100_'
else:
    prefix = ''

base_split_path = 'gastrectomy-'+gast_num+'/'
train_split_path = base_split_path+prefix+option+'_train'+split_num+'.txt'
valid_split_path = base_split_path+prefix+'valid'+split_num+'.txt'

work_dir = '/code/multimodal/logs/deeplabv3_g{}_{}'.format(gast_num, split_num)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.0),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=6*3,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=train_img_dir,
        ann_dir=train_ann_dir,
        split=train_split_path,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=train_img_dir,
        ann_dir=train_ann_dir,
        split=valid_split_path,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=train_img_dir,
        ann_dir=train_ann_dir,
        split=valid_split_path,
        pipeline=test_pipeline))



# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
# optimizer = dict(
#     type='AdamW',
#     lr=1e-02,
#     betas=(0.9, 0.999),
#     weight_decay=0.01,)
optimizer_config = dict()
# learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1,
    min_lr_ratio=1e-07,
    by_epoch=True,
)

# runtime settings
# runner = dict(type='IterBasedRunner', max_iters=80000)
# checkpoint_config = dict(by_epoch=False, interval=8000)
# evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)
runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(by_epoch=True, interval=100)
evaluation = dict(interval=100, metric='mIoU')

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = False
