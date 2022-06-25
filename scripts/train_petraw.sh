#!/usr/bin/env bash
cd ..

CUDA_VISIBLE_DEVICES=5 python main.py --num_gpus 1 \
                        --model 'lstm' \
                        --input_size 28 --hidden_size 256 \
                        --linear_dim 256 --n_layer 1 \
                        --batch_size 128 --dataset 'petraw' \
                        --data_base_path '/dataset3/multimodal' \
                        --data_type 'ki' --optimizer 'adam' \
                        --lr_scheduler 'step_lr' \
                        --loss_fn 'cb' \
                        --target_metric 'Balance-Acc' \
                        --inference_per_frame \
                        --fold 1 \
                        --max_epoch 100

# CUDA_VISIBLE_DEVICES=4 python main.py --num_gpus 1 \
#                         --model 'lstm' \
#                         --input_size 28 --hidden_size 256 \
#                         --linear_dim 256 --n_layer 1 \
#                         --batch_size 128 --dataset 'petraw' \
#                         --data_base_path '/dataset3/multimodal' \
#                         --data_type 'ki' --optimizer 'adam' \
#                         --lr_scheduler 'cosine_lr' \
#                         --loss_fn 'cb' \
#                         --target_metric 'Balance-Acc' \
#                         --inference_per_frame \
#                         --fold 1 \
#                         --max_epoch 100