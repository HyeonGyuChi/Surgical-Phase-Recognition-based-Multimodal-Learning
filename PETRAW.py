import argparse
import os
import os.path as osp
import pickle

import mmcv
import numpy as np
import torch

import sys 
sys.path.append('/root/')
from mmaction.datasets.pipelines import Compose
from mmaction.models import build_model




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

            # inputs = inputs.unsqueeze(0)
            
            phase_prob, step_prob, left_prob, right_prob = model(inputs, return_loss=False, infer_3d=True)

            results.append([phase_prob.cpu().numpy().squeeze(), step_prob.cpu().numpy().squeeze(), left_prob.cpu().numpy().squeeze(), right_prob.cpu().numpy().squeeze()])
   
            prog_bar_frames.update()
            start_idx += 1
    # print(results)
    # print(np.shape(results))
    return np.array(results)#np.concatenate(results)


def main():
    Path = '/root/checkpoint/slowfast_r50_e50'

    frame_interval = 1
    output_prefix = os.path.join(sys.argv[3], 'PETRAW_hutom_petraw', 'Workflow', 'Task1')
    modality = 'RGB'
    is_rgb = True
    clip_len = 1 if is_rgb else 5
    input_format = 'NCHW' if is_rgb else 'NCHW_Flow'
    
    config = mmcv.Config.fromfile(os.path.join(Path, 'config.py'))
    img_norm_cfg = config['img_norm_cfg']
    flow_norm_cfg = dict(mean=[128, 128], std=[128, 128])
    # args.img_norm_cfg = rgb_norm_cfg if args.is_rgb else flow_norm_cfg
    f_tmpl = 'frame{:06d}.jpg' if is_rgb else 'flow_{}_{:05d}.jpg'
    in_channels = clip_len * (3 if is_rgb else 2)
    # max batch_size for one forward
    batch_size = config['data']['test']['pipeline'][0]['clip_len']

    model = build_model(config['model'])
    # model['test_cfg'] = config['test_cfg']
    # load pretrained weight into the feature extractor
    state_dict = torch.load(os.path.join(Path, 'latest.pth'))['state_dict']
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()

    # define the data pipeline for Untrimmed Videos
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
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format=input_format),
        dict(type='Collect', keys=['imgs'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
    data_pipeline = Compose(data_pipeline)

    # data = open(args.data_list).readlines()
    video_dir = sys.argv[1]
    kinematic_dir = sys.argv[2]
    data = os.listdir(video_dir)
    data = [os.path.join(video_dir, x) for x in data]
    

    # enumerate Untrimmed videos, extract feature from each of them
    prog_bar = mmcv.ProgressBar(len(data))
    if not osp.exists(output_prefix):
        os.system(f'mkdir -p {output_prefix}')

    for video_dir in data:
        # frame_dir, length = item.split()

        # frame_dir = osp.join(args.data_prefix, frame_dir)
        
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
            
            # with open(output_file, 'wb') as fout:
            #     pickle.dump(prob, fout)
            ann_dict = annotation_dict()
            with open(output_file, 'w') as f:
                f.write('Frame Phase Step Verb_Left Verb_Right\n')
                for idx in range(len(prob)):
                    phase_idx = np.argmax(prob[idx][0])
                    phase = ann_dict['Phase'][phase_idx]
                    step_idx = np.argmax(prob[idx][1])
                    step = ann_dict['Step'][step_idx]
                    left_idx = np.argmax(prob[idx][2])
                    left = ann_dict['Verb_Left'][left_idx]
                    right_idx = np.argmax(prob[idx][3])
                    right = ann_dict['Verb_Right'][right_idx]
                    name = '\t'.join([str(idx), phase, step, left, right])
                    f.write(name +  '\n')
            prog_bar.update()

import sys
def annotation_dict():
    ann_dict = dict()
    ann_dict['Phase'] = { # 3
            0: 'Idle',
            1: 'Transfer Left to Right',
            2: 'Transfer Right to Left',
        }
    ann_dict['Step'] = { # 13
            0: 'Idle', 
            1: 'Block 1 L2R', 2: 'Block 2 L2R', 3: 'Block 3 L2R',
            4: 'Block 4 L2R', 5: 'Block 5 L2R', 6: 'Block 6 L2R',
            7: 'Block 1 R2L', 8: 'Block 2 R2L', 9: 'Block 3 R2L',
            10: 'Block 4 R2L', 11: 'Block 5 R2L', 12: 'Block 6 R2L', 
        }
    ann_dict['Verb_Left'] = { # 7
            0: 'Idle', 1: 'Catch', 2: 'Drop', 
            3: 'Extract', 4: 'Hold', 5: 'Insert', 6: 'Touch',
        }
    ann_dict['Verb_Right'] = { # 7
            0: 'Idle', 1: 'Catch', 2: 'Drop', 
            3: 'Extract', 4: 'Hold', 5: 'Insert', 6: 'Touch',
        }
    return ann_dict
    
if __name__ == '__main__':
    main()
    # args = parse_args()
    # config = mmcv.Config.fromfile(args.config_file)
    
    # for key in os.environ.keys():
    #     print(key, os.environ[key])
    # model = build_model(config['model'])
