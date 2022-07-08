#!/usr/bin/env bash

cd ..

N_GPUS=1
GPUS='7'
PORT=25320
CONFIG='/code/multimodal/logs/slowfast_gastric_40_4/slowfast_g40_hsb.py'
CHECKPOINT='/code/multimodal/logs/slowfast_gastric_40_4/best_top1_acc_epoch_185.pth'


CUDA_VISIBLE_DEVICES=6 python infer_test.py --model 'slowfast' \
                    --slowfast_depth 50 --batch_size 16 \
                    --dataset 'gast_mm' \
                    --data_base_path '/dataset3/multimodal' \
                    --data_type 'vd' --optimizer 'adam' \
                    --target_metric 'val_loss' \
                    --overlap_ratio 0.5 \
                    --clip_size 32 \
                    --subsample_ratio 30 \
                    --fold 4