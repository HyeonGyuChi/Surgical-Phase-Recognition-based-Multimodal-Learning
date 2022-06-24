#!/usr/bin/env bash

cd '../accessory/mmsegmentation/tools'

CONFIG='../../../core/config/deeplabv3/deeplabv3_plus_g40_101.py'
# N_GPUS=1
# GPUS='6'
N_GPUS=4
GPUS='4,5,6,7'
PORT=24531

CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train.py $CONFIG --launcher pytorch

