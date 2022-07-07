# #!/usr/bin/env bash
# cd '../accessory/mmaction2/tools'

# # CONFIG='../../../core/config/slowfast/slowfast_multi_task.py'
# # N_GPUS=1
# # GPUS='7'
# # PORT=29539

# # CUDA_VISIBLE_DEVICES=$GPUS python train.py $CONFIG


# CONFIG='../../../core/config/slowfast/slowfast_multi_task.py'
# N_GPUS=2
# GPUS='6,7'
# PORT=29539

# CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=29500 \
#         train.py $CONFIG --launcher pytorch #--validate


#!/usr/bin/env bash
cd '../accessory/mmaction2/tools'

# CONFIG='../../../core/config/slowfast/slowfast_multi_task.py'
# N_GPUS=1
# GPUS='7'
# PORT=29539

# CUDA_VISIBLE_DEVICES=$GPUS python train.py $CONFIG


# CONFIG='../../../core/config/slowfast/slowfast_multi_task.py'
CONFIG='../../../core/config/slowfast/slowfast_g40_hsb.py'
N_GPUS=4
GPUS='4,5,6,7'
PORT=29539

CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.launch --nproc_per_node=$N_GPUS --master_port=29500 \
        train.py $CONFIG --launcher pytorch --validate