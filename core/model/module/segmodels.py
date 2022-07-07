import sys 
sys.path.append('../../../accessory/mmsegmentation')
from mmseg.models import build_segmentor

import torch
import mmcv


def generate_segmodel(args):
    if args.seg == 'swin':
        config_path = './core/config/swin/upernet_swin_g40.py'
    elif args.seg == 'ocr':
        config_path = './core/config/ocr/ocr-hrnet_g40.py'
    elif args.seg == 'deeplabv3':
        config_path = './core/config/deeplabv3/deeplabv3_plus_g40_101.py'

    print('load model config : ', config_path)

    config = mmcv.Config.fromfile(config_path)
    model = build_segmentor(config['model'])

    return model
