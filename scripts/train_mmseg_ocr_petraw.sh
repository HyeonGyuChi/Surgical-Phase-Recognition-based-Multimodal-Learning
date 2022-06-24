#!/usr/bin/env bash

cd '../accessory/mmsegmentation/tools'

CONFIG='../../../core/config/ocr/ocr-hrnet.py'
N_GPUS=1
GPUS='6'
PORT=29531

CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train.py $CONFIG --launcher pytorch
