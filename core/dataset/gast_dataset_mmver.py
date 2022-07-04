import os
import natsort
import numpy as np
from glob import glob
import torch
import json
import pandas as pd
from PIL import Image
import cv2
from numpy import array
from core.utils.augmentor import Augmentor#, SignalAugmentor


class GastrectomyDatasetMM(torch.utils.data.Dataset):
    def __init__(self, args, state='train'):
        self.args = args
        self.state = state
        self.data_path = self.args.data_base_path
        self.dtype = self.args.data_type
        self.task = self.args.task
        self.fold = self.args.fold

        self.patient_dict = {
            'train': {
                1: ['R001', 'R002', 'R005', 'R007', 'R010', 
                'R014', 'R015', 'R019', 'R048', 'R056', 
                'R074', 'R076', 'R084', 'R094', 'R100', 
                'R117', 'R201', 'R202', 'R203', 'R204', 
                'R205', 'R206', 'R207', 'R209', 'R210', 
                'R301', 'R302', 'R304', 'R305', 'R313'],
                2: ['R002', 'R003', 'R004', 'R005', 'R006', 
                'R013', 'R014', 'R015', 'R017', 'R018', 
                'R022', 'R048', 'R076', 'R084', 'R094', 
                'R116', 'R201', 'R202', 'R204', 'R205', 
                'R206', 'R207', 'R208', 'R209', 'R210', 
                'R301', 'R302', 'R303', 'R305', 'R313'],
                3: ['R001', 'R002', 'R003', 'R004', 'R006', 
                'R007', 'R010', 'R013', 'R014', 'R015', 
                'R017', 'R018', 'R019', 'R022', 'R056', 
                'R074', 'R084', 'R100', 'R116', 'R117', 
                'R201', 'R203', 'R205', 'R207', 'R208', 
                'R210', 'R302', 'R303', 'R304', 'R313'],
                4: ['R001'],
            },

            'valid': {
                1: ['R003', 'R004', 'R006', 'R013', 'R017', 'R018', 'R022', 'R116', 'R208', 'R303'],
                2: ['R001', 'R007', 'R010', 'R019', 'R056', 'R074', 'R100', 'R117', 'R203', 'R304'],
                3: ['R005', 'R048', 'R076', 'R094', 'R202', 'R204', 'R206', 'R209', 'R301', 'R305'],
                4: ['R003'],
            }
        }
        
        self.target_list = self.patient_dict[self.state][self.fold]
        self.load_data()

        if self.state == 'train':
            self.aug = Augmentor(self.args.augmentations)
        elif self.state == 'valid':
            self.aug = Augmentor(self.args.val_augmentations)

    def __len__(self):
        return len(self.data_dict['video']) #* 32 

    def __getitem__(self, index):
        index = index

        label, img_path_list = self.data_dict['video'][index]
        clip_len = len(img_path_list)

        # while clip_len < self.args.clip_size:
        #     rand_id = int(np.random.choice(len(self.data_dict['video']), 1))
        #     label, img_path_list = self.data_dict['video'][index]
        #     clip_len = len(img_path_list)

        hf_sz = self.args.clip_size // 2
        sample_ratio = self.args.subsample_ratio

        while True:
            rand_id = int(np.random.choice(list(range(sample_ratio*hf_sz, clip_len-sample_ratio*hf_sz, sample_ratio)), 1))

            if rand_id - 16 >= 0 and rand_id + 15 < clip_len:
                index = rand_id
                break

        data = {}
        
        X = []
        for vpath in img_path_list[rand_id-16:rand_id+16]:
            img = Image.open(vpath) # h, w, ch
            X.append(img)
            # img = cv2.imread(vpath) # h, w, ch
            # X.append(img[:,:,::-1])

        X = self.aug(X)
        
        X = torch.stack([torch.Tensor(_X) for _X in X], dim=0)                
        
        if self.args.model == 'slowfast':
            X = X.permute(1, 0, 2, 3).unsqueeze(0)
        else:
            X = X.permute(1, 0, 2, 3)

        data['video'] = X

        labels = torch.from_numpy(np.array(label)).long()

        return data, labels

    def load_data(self):
        # load specific data
        print('[+] Load data .....')
        self.data_dict = {}

        # load video
        if 'vd' in self.args.data_type:
            print('[+] Load Video data')
            self.load_video()
            print('[-] Load Video data ... done')

        print('[-] Load data ..... done')

    def load_video(self):
        """
            ? x ? (30Hz)
        """
        self.data_dict['video'] = []
        n_classes = 27
        self.class_weights = []
        self.class_cnt = []
        self.class_cnt.append(np.zeros(n_classes))
        self.class_weights.append(np.zeros(n_classes))

        target_path = self.data_path + '/gastric/rawframes'

        patient_list = os.listdir(target_path)
        patient_list = natsort.natsorted(patient_list)

        for tmp_patient in patient_list:
            p = tmp_patient.split('_')[4]
            patient = 'R{:03d}'.format(int(p))

            if patient in self.target_list:
                p_path = target_path + f'/{tmp_patient}'
                video_list = natsort.natsorted(os.listdir(p_path))

                for video in video_list:
                    label = int(video.split('_')[-1])
                    v_path = p_path + f'/{video}'

                    file_list = natsort.natsorted(glob(v_path + '/*.jpg'))
                    if len(file_list) >= self.args.clip_size * self.args.subsample_ratio:
                        self.data_dict['video'].append([label, file_list])
                        self.class_cnt[0][label] += len(file_list)

        # class weight computation
        for idx in range(1):
            if len(self.class_cnt[idx]):
                bot_sum = 0
                n_classes = len(self.class_cnt[idx])

                for idx2 in range(n_classes):
                    bot_sum += self.class_cnt[idx][idx2]

                    if idx >= 2:
                        print(idx2, self.class_cnt[idx][idx2])
                    
                for idx2 in range(n_classes):
                    self.class_weights[idx][idx2] = bot_sum / (n_classes * self.class_cnt[idx][idx2])
                
                # if idx < 2:
                #     self.class_weights[idx] = torch.Tensor(np.ones(len(self.class_cnt[idx]))).cuda()
                # else:
                self.class_weights[idx] = torch.Tensor(self.class_cnt[idx]).cuda()

            print('CLS WEIGHTS - ', idx, ' : ',  self.class_cnt[idx], self.class_weights[idx])
