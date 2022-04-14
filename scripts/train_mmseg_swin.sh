#!/usr/bin/env bash

cd '../accessory/mmsegmentation/tools'

CONFIG='../../../core/config/upernet_swin.py'
# N_GPUS=4
# GPUS='4,5,6,7'
# PORT=29531

# CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=$PORT train.py $CONFIG --launcher pytorch



CHECKPOINT='/code/multimodal/logs/swin_petraw/epoch_100.pth'
# N_GPUS=4
# GPUS='4,5,6,7'
N_GPUS=1
GPUS='7'
PORT=29539

CUDA_VISIBLE_DEVICES=$GPUS python test.py $CONFIG $CHECKPOINT --show-dir '/code/multimodal/logs/swin_petraw/results_test' --opacity 1


# CHECKPOINT='/code/multimodal/logs/swin_petraw/epoch_100.pth'
# N_GPUS=1
# GPUS='7'
# PORT=29539

# CONFIG='/code/multimodal/core/config/swin/upernet_swin.py'
# CUDA_VISIBLE_DEVICES=$GPUS python test.py $CONFIG $CHECKPOINT --show-dir '/code/multimodal/logs/swin_petraw/results_test' --opacity 1

# CONFIG='/code/multimodal/core/config/swin/upernet_swin2.py'
# CUDA_VISIBLE_DEVICES=$GPUS python test.py $CONFIG $CHECKPOINT --show-dir '/code/multimodal/logs/swin_petraw/results_test' --opacity 1

# CONFIG='/code/multimodal/core/config/swin/upernet_swin3.py'
# CUDA_VISIBLE_DEVICES=$GPUS python test.py $CONFIG $CHECKPOINT --show-dir '/code/multimodal/logs/swin_petraw/results_test' --opacity 1

# CONFIG='/code/multimodal/core/config/swin/upernet_swin4.py'
# CUDA_VISIBLE_DEVICES=$GPUS python test.py $CONFIG $CHECKPOINT --show-dir '/code/multimodal/logs/swin_petraw/results_test' --opacity 1

# CONFIG='/code/multimodal/core/config/swin/upernet_swin5.py'
# CUDA_VISIBLE_DEVICES=$GPUS python test.py $CONFIG $CHECKPOINT --show-dir '/code/multimodal/logs/swin_petraw/results_test' --opacity 1

# CONFIG='/code/multimodal/core/config/swin/upernet_swin6.py'
# CUDA_VISIBLE_DEVICES=$GPUS python test.py $CONFIG $CHECKPOINT --show-dir '/code/multimodal/logs/swin_petraw/results_test' --opacity 1

# CONFIG='/code/multimodal/core/config/swin/upernet_swin7.py'
# CUDA_VISIBLE_DEVICES=$GPUS python test.py $CONFIG $CHECKPOINT --show-dir '/code/multimodal/logs/swin_petraw/results_test' --opacity 1

# CONFIG='/code/multimodal/core/config/swin/upernet_swin8.py'
# CUDA_VISIBLE_DEVICES=$GPUS python test.py $CONFIG $CHECKPOINT --show-dir '/code/multimodal/logs/swin_petraw/results_test' --opacity 1

# CONFIG='/code/multimodal/core/config/swin/upernet_swin9.py'
# CUDA_VISIBLE_DEVICES=$GPUS python test.py $CONFIG $CHECKPOINT --show-dir '/code/multimodal/logs/swin_petraw/results_test' --opacity 1

# CONFIG='/code/multimodal/core/config/swin/upernet_swin10.py'
# CUDA_VISIBLE_DEVICES=$GPUS python test.py $CONFIG $CHECKPOINT --show-dir '/code/multimodal/logs/swin_petraw/results_test' --opacity 1
