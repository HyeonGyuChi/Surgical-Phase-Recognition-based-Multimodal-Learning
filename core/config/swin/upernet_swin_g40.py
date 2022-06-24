# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
classes=32
model = dict(
    type='EncoderDecoder',
    pretrained='https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192340-593b0e13.pth',
    # pretrained='/code/multimodal/accessory/mmsegmentation/pretrain/upernet_swin_base_patch4_window7_mmseg.pth',
    # pretrained='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth',
    # pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        # pretrain_img_size=224,
        # embed_dims=96,
        embed_dims=128,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        # depths=[2, 2, 6, 2],
        depths=[2, 2, 18, 2],
        # num_heads=[3, 6, 12, 24],
        num_heads=[4, 8, 16, 32],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg,
        ),
    decode_head=dict(
        type='UPerHead',
        # in_channels=[96, 192, 384, 768],
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        # in_channels=384,
        in_channels=512,
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
work_dir = '/code/multimodal/logs/swin_g40_56'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
scale_size = (769, 769)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=scale_size, ratio_range=(0.5, 2.0)),
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
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict()
lr_config = dict(
    # policy='CosineAnnealing',
    policy='poly',
    warmup='linear',
    # warmup_iters=1000,
    # warmup_ratio=0.1,
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    # min_lr_ratio=1e-07,
    min_lr=0.0,
    by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(by_epoch=True, interval=100)
evaluation = dict(interval=100, metric='mIoU')
gpu_ids = range(0, 1)