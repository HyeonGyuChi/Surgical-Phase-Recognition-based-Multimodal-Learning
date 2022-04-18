import os
import sys 
sys.path.append('./accessory/mmaction2')
import torch
import mmcv
from mmaction.models import build_model

from core.config.set_opts import load_opts
from core.api.trainer import Trainer
from core.dataset import get_dataset
from core.model import get_model
from tqdm import tqdm


def main():
    args = load_opts()
    # trainer = Trainer(args)

    ckpt_path = '/raid/results/phase_recognition/mmaction/petraw/multi_task/slowfast_r50_e50_clip8_split1/latest.pth'
    config_path = './core/config/slowfast_multi_task.py'

    # config = mmcv.Config.fromfile(config_path)
    # model = build_model(config['model'])
    

    model = get_model(args)
    # state_dict = torch.load(ckpt_path)['state_dict']
    # model.load_state_dict(state_dict)

    model = model.cuda()
    model.eval()

    # Load dataset
    print('======= Load Dataset =======')
    train_loader, val_loader = get_dataset(args)
    
    # model.train()
    current_epoch = 1
        

    print('======= Inference Test =======')
    with torch.no_grad():
        for data in tqdm(train_loader, desc='[Epoch {} - Train Phase] : '.format(current_epoch)):
            # optimizer.zero_grad()
            
            # B x n_clips x ch x seq x 256 x 256
            x, y = data
            if args.device == 'cuda':
                for k in x.keys():
                    x[k] = x[k].to(args.device)
                y = y.to(args.device)
                
            x = x['video'].unsqueeze(1)

            feat = model.get_feature(x)
            phase_prob, step_prob, left_prob, right_prob = model(x, return_loss=False, infer_3d=True)

            break

if __name__ == '__main__':
    main()