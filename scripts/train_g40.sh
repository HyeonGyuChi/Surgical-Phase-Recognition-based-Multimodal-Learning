#!/usr/bin/env bash
cd ..

CUDA_VISIBLE_DEVICES=4,5,6 python main.py --num_gpus 3 \
                        --model 'slowfast' --slowfast_depth 50 \
                        --batch_size 128 --dataset 'gast' \
                        --data_base_path '/dataset3/multimodal' \
                        --data_type 'vd' --optimizer 'adam' \
                        --lr_scheduler 'cosine_lr' \
                        --loss_fn 'cb' \
                        --max_epoch 100