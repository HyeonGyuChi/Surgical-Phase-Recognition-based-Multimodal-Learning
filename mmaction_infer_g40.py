import sys 
sys.path.append('./accessory/mmaction2')
from mmaction.datasets.pipelines import Compose
from mmaction.models import build_model


import argparse
import os
import os.path as osp
import pickle

import mmcv
import numpy as np
import torch
import natsort
from glob import glob


from core.config.set_opts import load_opts
from core.model import get_model


def set_pipeline(config_path):
    config = mmcv.Config.fromfile(config_path)

    data_pipeline = [
        dict(type='DecordInit'),
        dict(
            type='UntrimmedSampleFrames',#'UntrimmedSampleFrames',
            clip_len=1, #args.clip_len,
            frame_interval=1, #args.frame_interval,
            start_index=0),
        dict(type='DecordDecode'),#''FrameSelector' RawFrameDecode),
        dict(type='Resize', scale=(-1, 256)),
        dict(type='CenterCrop', crop_size=256),
        dict(type='Normalize', **config['img_norm_cfg']),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['imgs'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
    data_pipeline = Compose(data_pipeline)

    return data_pipeline

def forward_data(model, data, batch_size, frame_interval):
    # chop large data into pieces and extract feature from them
    results = []
    start_idx = 0
    num_clip = data.shape[0] - (batch_size*frame_interval - frame_interval)*2
    prog_bar_frames = mmcv.ProgressBar(num_clip)
    while start_idx < num_clip:
        with torch.no_grad():
            inputs = data[start_idx:start_idx + batch_size*frame_interval:frame_interval]
            inputs = inputs.permute(1,2,0,3,4)

            tmps = data[start_idx + batch_size*frame_interval//2:start_idx + batch_size*frame_interval//2 + batch_size*frame_interval:frame_interval]
            tmps = tmps.permute(1,2,0,3,4)
            inputs = torch.cat((inputs, tmps), dim=0)

            tmps = data[start_idx + batch_size*frame_interval - frame_interval:start_idx + batch_size*frame_interval - frame_interval + batch_size*frame_interval:frame_interval]
            tmps = tmps.permute(1,2,0,3,4)
            inputs = torch.cat((inputs, tmps), dim=0)

            inputs = inputs.unsqueeze(0)
            
            print('Sampled img shape : ', inputs.shape)

            phase_prob, step_prob, left_prob, right_prob = model(inputs, return_loss=False, infer_3d=True)

            results.append([phase_prob.cpu().numpy().squeeze(), step_prob.cpu().numpy().squeeze(), left_prob.cpu().numpy().squeeze(), right_prob.cpu().numpy().squeeze()])
   
            prog_bar_frames.update()
            start_idx += 1
    # print(results)
    # print(np.shape(results))
    return np.array(results)#np.concatenate(results)

def main():
    CONFIG='/code/multimodal/logs/slowfast_gastric_40_4/slowfast_g40_hsb.py'
    CHECKPOINT='/code/multimodal/logs/slowfast_gastric_40_4/best_top1_acc_epoch_185.pth'

    args = load_opts()
    model = get_model(args)

    state_dict = torch.load(CHECKPOINT)['state_dict']
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()


    data_pipeline = set_pipeline(config_path=CONFIG)

    data = glob(video_dir + '/*mp4')
    data = natsort.natsorted(data)

    modality = 'RGB'

    prog_bar = mmcv.ProgressBar(len(data))
    if not osp.exists(output_prefix):
        os.system(f'mkdir -p {output_prefix}')

    for video_dir in data:        
        patient_channel = video_dir.split('/')[-1].split('.')[0]
     
  
        
        output_file = patient_channel + '_Results_Task1.txt'
             #osp.basename(frame_dir) + '.pkl'
   
             #osp.basename(frame_dir) + '.pkl'
        output_file = osp.join(output_prefix, output_file)
       
        assert output_file.endswith('.txt')
        # length = int(length)

        # prepare a psuedo sample
        tmpl = dict(
            filename=video_dir,
            label=-1,
            # total_frames=length,
            # filename_tmpl=args.f_tmpl,
            start_index=1,
            modality=modality)
        sample = data_pipeline(tmpl)
        # print(sample)
        imgs = sample['imgs']
        shape = imgs.shape
        # print("======================================")
        # print(shape)
        # print("======================================")
        # the original shape should be N_seg * C * H * W, resize it to N_seg *
        # 1 * C * H * W so that the network return feature of each frame (No
        # score average among segments)
    
        with torch.no_grad():
            imgs = imgs.reshape((shape[0], 1) + shape[1:])

            zero_padding = torch.zeros((batch_size*frame_interval - frame_interval, 1, 3, 256, 256))
            imgs = torch.cat((zero_padding, imgs, zero_padding), dim=0)
            imgs = imgs.cuda()
        
        
            prob = forward_data(model, imgs, batch_size, frame_interval)
            
            prog_bar.update()

    

if __name__ == '__main__':
    main()