#!/usr/bin/env bash

cd '../accessory/mmsegmentation/tools'

# CONFIG='../../../core/config/swin/upernet_swin.py'
# N_GPUS=4
# GPUS='4,5,6,7'
# PORT=29531

# CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train.py $CONFIG --launcher pytorch

N_GPUS=1



GPUS='5'
PORT=25220
CONFIG='/code/multimodal/core/config/deeplabv3/deeplabv3_plus_petraw_18.py'
CHECKPOINT='/code/multimodal/logs-bak/bak/deeplabv3_petraw/epoch_100.pth'
SAVE_PATH='/code/multimodal/logs/deeplab_petraw_test/results_test'


# GPUS='6'
# PORT=25320
# CONFIG='/code/multimodal/core/config/swin/upernet_swin_petraw.py'
# CHECKPOINT='/code/multimodal/logs-bak/bak/swin_petraw/epoch_100.pth'
# SAVE_PATH='/code/multimodal/logs/swin_petraw_test/results_test'


# GPUS='7'
# PORT=25320
# CONFIG='/code/multimodal/core/config/ocr/ocr-hrnet.py'
# CHECKPOINT='/code/multimodal/logs-bak/bak/ocr_petraw/epoch_100.pth'
# SAVE_PATH='/code/multimodal/logs/ocr_petraw_test/results_test'



for idx in 39 40; do
    CUDA_VISIBLE_DEVICES=$GPUS python test.py $CONFIG $CHECKPOINT --show-dir \
    $SAVE_PATH --opacity 1 --test_idx ${idx}
done


# for idx in 1 2 3 4 5 6 7 8 9 10; do
#     CUDA_VISIBLE_DEVICES=$GPUS python test.py $CONFIG $CHECKPOINT --show-dir \
#     $SAVE_PATH --opacity 1 --test_idx ${idx}
# done

# for idx in 11 12 13 14 15 16 17 18 19 20; do
#     CUDA_VISIBLE_DEVICES=$GPUS python test.py $CONFIG $CHECKPOINT --show-dir \
#     $SAVE_PATH --opacity 1 --test_idx ${idx}
# done

# for idx in 21 22 23 24 25 26 27 28 29 30; do
#     CUDA_VISIBLE_DEVICES=$GPUS python test.py $CONFIG $CHECKPOINT --show-dir \
#     $SAVE_PATH --opacity 1 --test_idx ${idx}
# done

# for idx in 31 32 33 34 35 36 37 38; do
# # for idx in 31 32 33 34 35 36 37 38 39 40; do
#     CUDA_VISIBLE_DEVICES=$GPUS python test.py $CONFIG $CHECKPOINT --show-dir \
#     $SAVE_PATH --opacity 1 --test_idx ${idx}
# done

# for idx in 41 42 43 44 45 46 47 48 49 50; do
#     CUDA_VISIBLE_DEVICES=$GPUS python test.py $CONFIG $CHECKPOINT --show-dir \
#     $SAVE_PATH --opacity 1 --test_idx ${idx}
# done

# for idx in 51 52 53 54 55 56 57 58 59 60; do
#     CUDA_VISIBLE_DEVICES=$GPUS python test.py $CONFIG $CHECKPOINT --show-dir \
#     $SAVE_PATH --opacity 1 --test_idx ${idx}
# done
