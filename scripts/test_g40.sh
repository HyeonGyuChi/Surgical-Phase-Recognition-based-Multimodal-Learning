#!/usr/bin/env bash
cd ..


# CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --num_gpus 4 \
#                         --model 'resnet3d' --model_depth 50 \
#                         --batch_size 4 --dataset 'gast' \
#                         --data_base_path '/dataset3/multimodal' \
#                         --data_type 'vd' --optimizer 'adam' \
#                         --lr_scheduler 'step_lr' \
#                         --loss_fn 'cb' \
#                         --target_metric 'Percentage-Acc' \
#                         --inference_per_frame \
#                         --subsample_ratio 30 \
#                         --fold 4 \
#                         --max_epoch 2

CUDA_VISIBLE_DEVICES=4,5 python main.py --num_gpus 2 \
                        --model 'resnet3d' --model_depth 50 \
                        --batch_size 4 --dataset 'gast' \
                        --data_base_path '/dataset3/multimodal' \
                        --data_type 'vd' --optimizer 'adam' \
                        --lr_scheduler 'step_lr' \
                        --loss_fn 'cb' \
                        --target_metric 'Percentage-Acc' \
                        --inference_per_frame \
                        --subsample_ratio 30 \
                        --fold 4 \
                        --max_epoch 2
