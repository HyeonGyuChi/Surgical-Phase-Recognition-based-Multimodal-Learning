#!/usr/bin/env bash

cd '../accessory/mmsegmentation/tools'

N_GPUS=1

GPUS='5'
PORT=25220
# CONFIG='/code/multimodal/core/config/deeplabv3_g40_12/deeplabv3_plus_g40_101.py'
CONFIG='/code/multimodal/logs/deeplabv3_g40_12/deeplabv3_plus_g40_101.py'
CHECKPOINT='/code/multimodal/logs/deeplabv3_g40_12/epoch_300.pth'
SAVE_PATH='/code/multimodal/logs/deeplabv3_g40_12/results'


# GPUS='6'
# PORT=25320
# CONFIG='/code/multimodal/logs/swin_g40_12/upernet_swin_g40.py'
# CHECKPOINT='/code/multimodal/logs/swin_g40_12/epoch_300.pth'
# SAVE_PATH='/code/multimodal/logs/swin_g40_12/results'


# GPUS='7'
# PORT=25320
# CONFIG='/code/multimodal/logs/ocr_g40_12/ocr-hrnet.py'
# CHECKPOINT='/code/multimodal/logs/ocr_g40_12/epoch_300.pth'
# SAVE_PATH='/code/multimodal/logs/ocr_g40_12/results'


# for idx in {1..508}; do
for ((idx=1;idx<=508;idx++)); do
    # echo $idx
    CUDA_VISIBLE_DEVICES=$GPUS python test.py $CONFIG $CHECKPOINT --show-dir \
    $SAVE_PATH --opacity 1 --test_idx ${idx}
done

