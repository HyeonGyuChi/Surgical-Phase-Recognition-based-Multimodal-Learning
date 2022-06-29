import sys 
sys.path.append('../../../accessory/mmaction2')
from mmaction.models import build_model

import torch
import mmcv



def generate_slowfast(args):
    config_path = None

    if args.dataset == 'petraw':
        if args.slowfast_depth == 50:
            config_path = './core/config/slowfast/slowfast_multi_task.py'
        elif args.slowfast_depth == 101:
            config_path = './core/config/slowfast/slowfast_multi_task2.py'
    elif args.dataset == 'gast':
        if args.slowfast_depth == 50:
            config_path = './core/config/slowfast/slowfast_g40.py'
        elif args.slowfast_depth == 101:
            config_path = './core/config/slowfast/slowfast_g40_2.py'

    config = mmcv.Config.fromfile(config_path)
    model = build_model(config['model'])

    return model