#!/usr/bin/env bash
cd '../accessory/mmaction2/tools'

CONFIG='../../../core/config/slowfast/slowfast_multi_task.py'

N_GPUS=1
GPUS='7'
PORT=29539

CUDA_VISIBLE_DEVICES=$GPUS python train.py $CONFIG