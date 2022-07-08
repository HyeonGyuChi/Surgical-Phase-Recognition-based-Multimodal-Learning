#!/usr/bin/env bash
cd ..

# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --num_gpus 4 \
#                         --model 'resnet3d' --model_depth 50 \
#                         --batch_size 64 --dataset 'gast' \
#                         --data_base_path '/dataset3/multimodal' \
#                         --data_type 'vd' --optimizer 'adam' \
#                         --init_lr 1e-2 \
#                         --lr_scheduler 'step_lr' \
#                         --loss_fn 'cb' \
#                         --target_metric 'Total_P-Acc' \
#                         --n_classes 27 \
#                         --inference_per_frame \
#                         --clip_size 32 \
#                         --subsample_ratio 30 \
#                         --fold 1 \
#                         --max_epoch 256

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --num_gpus 4 \
                        --model 'slowfast' --slowfast_depth 50 \
                        --batch_size 64 --dataset 'gast' \
                        --data_base_path '/dataset3/multimodal' \
                        --data_type 'vd' --optimizer 'adam' \
                        --init_lr 1e-2 \
                        --lr_scheduler 'cosine_lr' \
                        --loss_fn 'cb' \
                        --target_metric 'Total_P-Acc' \
                        --n_classes 27 \
                        --inference_per_frame \
                        --clip_size 32 \
                        --subsample_ratio 30 \
                        --fold 1 \
                        --max_epoch 50
