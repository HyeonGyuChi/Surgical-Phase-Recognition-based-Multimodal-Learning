#!/usr/bin/env bash
cd ..

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --num_gpus 4 \
                        --model 'slowfast' --slowfast_depth 50 \
                        --batch_size 64 --dataset 'gast_mm' \
                        --data_base_path '/dataset3/multimodal' \
                        --data_type 'vd' --optimizer 'adam' \
                        --init_lr 1e-3 \
                        --lr_scheduler 'cosine_lr' \
                        --loss_fn 'cb' \
                        --target_metric 'val_loss' \
                        --inference_per_frame \
                        --clip_size 32 \
                        --subsample_ratio 30 \
                        --fold 1 \
                        --max_epoch 256