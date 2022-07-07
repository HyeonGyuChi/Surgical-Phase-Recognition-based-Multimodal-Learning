#!/usr/bin/env bash
cd '../accessory/mmaction2/tools'

CONFIG='/code/multimodal/logs/slowfast_gastric_40_4/slowfast_g40_hsb.py'
CHECKPOINT='/code/multimodal/logs/slowfast_gastric_40_4/best_top1_acc_epoch_185.pth'
N_GPUS=1
GPUS='7'
PORT=29666

CUDA_VISIBLE_DEVICES=$GPUS python test.py $CONFIG $CHECKPOINT \
    --eval "top_k_accuracy" "mean_class_accuracy"