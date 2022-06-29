#!/usr/bin/env bash
cd ..


# CUDA_VISIBLE_DEVICES=4 python main.py --num_gpus 1 \
#                         --model 'resnet3d' --model_depth 101 \
#                         --batch_size 2 --dataset 'gast' \
#                         --data_base_path '/dataset3/multimodal' \
#                         --data_type 'vd' --optimizer 'adam' \
#                         --lr_scheduler 'step_lr' \
#                         --loss_fn 'cb' \
#                         --target_metric 'Percentage-Acc' \
#                         --inference_per_frame \
#                         --subsample_ratio 30 \
#                         --fold 4 \
#                         --max_epoch 200


CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --num_gpus 4 \
                        --model 'resnet3d' --model_depth 50 \
                        --batch_size 128 --dataset 'gast' \
                        --data_base_path '/dataset3/multimodal' \
                        --data_type 'vd' --optimizer 'adam' \
                        --lr_scheduler 'step_lr' \
                        --loss_fn 'cb' \
                        --target_metric 'Percentage-Acc' \
                        --inference_per_frame \
                        --subsample_ratio 30 \
                        --fold 1 \
                        --max_epoch 300


# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --num_gpus 4 \
#                         --model 'slowfast' --slowfast_depth 50 \
#                         --batch_size 128 --dataset 'gast' \
#                         --data_base_path '/dataset3/multimodal' \
#                         --data_type 'vd' --optimizer 'adam' \
#                         --lr_scheduler 'cosine_lr' \
#                         --loss_fn 'cb' \
#                         --target_metric 'Percentage-Acc' \
#                         --inference_per_frame \
#                         --subsample_ratio 30 \
#                         --fold 1 \
#                         --max_epoch 300