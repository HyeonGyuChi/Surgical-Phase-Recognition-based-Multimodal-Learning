#!/usr/bin/env bash

cd '../accessory/mmsegmentation/tools'
N_GPUS=4
GPUS='4,5,6,7'
PORT=22271

CONFIG='../../../core/config/deeplabv3/deeplabv3_plus_g40_101.py'
CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train.py $CONFIG --launcher pytorch

CONFIG='../../../core/config/swin/upernet_swin_g40.py'
PORT=22272
CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train.py $CONFIG --launcher pytorch

CONFIG='../../../core/config/ocr/ocr-hrnet_g40.py'
PORT=22273
CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train.py $CONFIG --launcher pytorch
