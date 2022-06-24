# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
classes=32
model = dict(
    type='CascadeEncoderDecoder',
    num_stages=2,
    pretrained='open-mmlab://msra/hrnetv2_w48',
    # pretrained='/code/multimodal/accessory/mmsegmentation/pretrain/ocrnet_hr48_512x1024_80k_cityscapes.pth',
    # pretrained='https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr48_512x1024_160k_cityscapes/ocrnet_hr48_512x1024_160k_cityscapes_20200602_191037-dfbf1b0c.pth',
    # pretrained='https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr48_512x1024_80k_cityscapes/ocrnet_hr48_512x1024_80k_cityscapes_20200601_222752-9076bcdf.pth',
    backbone=dict(
        type='HRNet',
        norm_cfg=norm_cfg,
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                # num_channels=(18, 36)
                num_channels=(48, 96)
                ),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                # num_channels=(18, 36, 72)
                num_channels=(48, 96, 192)
                ),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                # num_channels=(18, 36, 72, 144)
                num_channels=(48, 96, 192, 384)
                ))),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[48, 96, 192, 384],
            channels=sum([48, 96, 192, 384]),
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            kernel_size=1,
            num_convs=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=classes,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='OCRHead',
            in_channels=[48, 96, 192, 384],
            channels=512,
            ocr_channels=256,
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            norm_cfg=norm_cfg,
            dropout_ratio=-1,
            num_classes=classes,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    # decode_head=[
    #     dict(
    #         type='FCNHead',
    #         in_channels=[18, 36, 72, 144],
    #         channels=sum([18, 36, 72, 144]),
    #         in_index=(0, 1, 2, 3),
    #         input_transform='resize_concat',
    #         kernel_size=1,
    #         num_convs=1,
    #         concat_input=False,
    #         dropout_ratio=-1,
    #         num_classes=19,
    #         norm_cfg=norm_cfg,
    #         align_corners=False,
    #         loss_decode=dict(
    #             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    #     dict(
    #         type='OCRHead',
    #         in_channels=[18, 36, 72, 144],
    #         in_index=(0, 1, 2, 3),
    #         input_transform='resize_concat',
    #         channels=512,
    #         ocr_channels=256,
    #         dropout_ratio=-1,
    #         num_classes=19,
    #         norm_cfg=norm_cfg,
    #         align_corners=False,
    #         loss_decode=dict(
    #             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ],
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
work_dir = '/code/multimodal/logs/ocr_g40_56'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
scale_size = (786, 786)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=scale_size, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.0),
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
        img_scale=crop_size,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
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

log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
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

runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(by_epoch=True, interval=100)
evaluation = dict(interval=100, metric='mIoU')
gpu_ids = range(0, 1)