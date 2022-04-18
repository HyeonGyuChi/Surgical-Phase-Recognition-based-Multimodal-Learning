import sys 
sys.path.append('../../../accessory/mmaction2')
from mmaction.models import build_model

import torch
import mmcv



def generate_slowfast(args):
    config_path = './core/config/slowfast/slowfast_multi_task.py'

    config = mmcv.Config.fromfile(config_path)
    model = build_model(config['model'])

    return model