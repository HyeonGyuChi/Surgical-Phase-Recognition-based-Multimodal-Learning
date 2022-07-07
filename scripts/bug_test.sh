#!/usr/bin/env bash
cd ..

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --num_gpus 4 \
                        --model 'slowfast' --slowfast_depth 50 \
                        --batch_size 64 --dataset 'gast' \
                        --data_base_path '/dataset3/multimodal' \
                        --data_type 'vd' --optimizer 'adam' \
                        --init_lr 1e-2 \
                        --lr_scheduler 'cosine_lr' \
                        --loss_fn 'ce' \
                        --target_metric 'val_loss' \
                        --n_classes 27 \
                        --inference_per_frame \
                        --overlap_ratio 0.5 \
                        --clip_size 32 \
                        --subsample_ratio 30 \
                        --fold 1 \
                        --max_epoch 50

