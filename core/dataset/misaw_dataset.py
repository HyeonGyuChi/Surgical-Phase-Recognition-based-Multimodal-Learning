import os
import cv2
import torch
import pickle
import re
import numpy as np
import pandas as pd
import natsort
from numpy import array
from glob import glob
from PIL import Image

from core.utils.augmentor import Augmentor#, SignalAugmentor


"""
    * in paper
    trainset (17 cases)
    - surgeon 2, 3, student 1, 2 (nb case : 3, 4, 6, 4)
    testset (10 cases)
    - surgeon 1, 3 (nb case : 4, 6)

"""

kinematic_all_columns = [
    'l_x', 'l_y', 'l_z', 'l_a', 'l_b', 'l_c', 'l_v1', 'l_v2',
    'r_x', 'r_y', 'r_z', 'r_a', 'r_b', 'r_c', 'r_v1', 'r_v2',
]


class MISAWDataset(torch.utils.data.Dataset):
    def __init__(self, config, state='train'):
        self.state = state
        self.config = config
        
        self.label_phase_pair = {'Idle':0, 'Suturing':1, 'Knot tying':2}
        self.label_step_pair = {'Idle':0, 
                                'Needle holding':1, 'Suture making':2,  'Suture handling':3,   
                                '1 knot':4, '2 knot':5, '3 knot':6,
                                'Idle Step': 7}
        self.label_activity_pair = {
                                'Idle': 0, 
                                'Hold': 1, 'Catch': 2, 'Position': 3,
                                'Give slack': 4, 'Insert': 5, 'Loosen completely': 6,
                                'Loosen partially': 7, 'Make a loop': 8,
                                'Pass through': 9, 'Pull': 10, 'Push': 11,
                                }         

        self.dtype = self.config.data_type
        self.task = self.config.task

        if self.state == 'train':
            self.data_path = self.config.data_base_path + '/MISAW/train'
            self.aug = Augmentor(self.config.augmentations)
        elif self.state == 'valid':
            self.data_path = self.config.data_base_path + '/MISAW/test'
            self.aug = Augmentor(self.config.val_augmentations)
            
        self.load_data()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data = {}
        
        for dtype in self.data_dict.keys():
            if dtype == 'video':
                vpath_list = self.data_dict[dtype][index]

                X = []
                for vpath in vpath_list:
                    img = Image.open(vpath) # h, w, ch
                    X.append(img)

                    X = self.aug(X)
                    X = torch.stack(X, dim=0)
                    X = torch.stack([torch.Tensor(_X) for _X in X], dim=0)
                    X = X.permute(1, 0, 2, 3)

                data[dtype] = X
                
            elif dtype == 'kinematic':
                signal = self.data_dict[dtype][index]
                data[dtype] = torch.from_numpy(np.array(signal)).float()

        labels = torch.from_numpy(np.array(self.labels[index])).long()

        return data, labels

    def load_data(self):
        # load data
        self.data_dict = {}

        if 'vd' in self.dtype:
            vd_data = self.load_video()
            self.data_dict['video'] = vd_data

        if 'ki' in self.dtype:
            ki_data = self.load_kinematics()
            self.data_dict['kinematic'] = ki_data
        
        self.labels = self.load_labels()

        # preprocessing (subsample, ovelapped data)
        self.preprocessing()

    def preprocessing(self):
        # subsample
        sample_rate = self.config.subsample_ratio
        go_subsample = sample_rate > 1 
        seq_size = self.config.clip_size
        
        if go_subsample:
            for key, _data in self.data_dict.items():
                for dir_name in _data.keys():
                    self.data_dict[key][dir_name] = _data[dir_name][::sample_rate]

            for dir_name in self.labels.keys():
                self.labels[dir_name] = self.labels[dir_name][::sample_rate]

        # overlapping data sequence
        if self.state != 'train':
            stride = int(seq_size)
        else:
            stride = int(self.config.clip_size * self.config.overlap_ratio)

        for key, _data in self.data_dict.items():
            seq_data = []
            
            for dir_name in _data.keys():
                d_len = len(_data[dir_name])
                
                for st in range(0, d_len, stride):
                    if st+self.config.clip_size < d_len:
                        seq_data.append(_data[dir_name][st:st+self.config.clip_size])
                    else:
                        break
                    
            self.data_dict[key] = array(seq_data)
        
        seq_data = []
        
        for d_num in self.labels.keys():
            data = self.labels[d_num]
            d_len = len(data)

            for st in range(0, d_len, stride):
                if st+seq_size < d_len:
                    if self.config.inference_per_frame:
                        seq_data.append(data[st:st+self.config.clip_size])
                    else:
                        seq_data.append(data[st:st+1])
                else:
                    break

        self.labels = array(seq_data)
        
    def standardization(self, x):
        mean_x = np.mean(x, 0)
        std_x = np.std(x, 0)

        return (x-mean_x) / (std_x + 1e-5)

    def load_video(self):
        """
            27 sequences of micro-surgical anastomosis
            - 30Hz, 960x540 resol
            
            need pre-processing
            - 960x540 -> 920x540 (center 40 pixels removed)
        """
        data_path = self.data_path + '/video_capture'

        dir_list = os.listdir(data_path)
        natsort.natsorted(dir_list)
        
        data = {}

        for dir_name in dir_list:
            print('Load Video {} ...'.format(dir_name))

            dpath = data_path + '/{}'.format(dir_name)
            file_list = glob(dpath + '/*.jpg')
            file_list = natsort.natsorted(file_list)

            data[dir_name] = file_list
            print('frame len : ', len(data[dir_name]))
        
        return data

    def load_kinematics(self):
        """
            kinematics - 16 variables (left / right)
            - positions (3)
            - rotation angles (3)
            - voltage of grip and output grip (2)
        """
        data_path = self.data_path + '/Kinematic'

        file_list = glob(data_path + '/*txt')
        file_list = natsort.natsorted(file_list)

        data = {}

        for fpath in file_list:
            print('Load Signal {} ...'.format(fpath))
            data_name = re.sub('.txt','',fpath.split('/')[-1])
            data_name = data_name[:3]

            raw_signal_selected = pd.read_csv(fpath,
                                    names=kinematic_all_columns,
                                    sep='\t').astype('float64')

            data[data_name]= self.standardization(raw_signal_selected.values)
            print('signal len : ', len(data[data_name]))

        return data

    def load_labels(self):
        """
            annotations (8 elements)
            - Idle
            - phase
            - step
            - Suturing
            - Needle holding
            - Suture making
            - Suture handling
            - Knot Tying
            - 1 knot
            - 2 knot
            - 3 knot
            - activity (left / right)
            - action verb
            - target
            - surgical instrument
        """
        label_path = self.data_path + '/Label'

        label_list = glob(label_path + '/*txt')
        label_list = natsort.natsorted(label_list)

        labels = {}
        for lab in label_list:
            label_name = re.sub('.txt','',lab.split('/')[-1])
            print('Label load and masking : ', label_name)
            label_name = label_name[:3]

            label = pd.read_csv(lab)
            
            if self.task == 'phase':
                swap_label = self.label_phase_pair
                t_label = np.zeros((len(label), 1))

                for ri, row in enumerate(label.values):
                    row = row[0].split('\t')
                    t_label[ri, 0] = swap_label[row[0]]

            elif self.task == 'step':
                swap_label = self.label_step_pair
                t_label = np.zeros((len(label), 1))

                for ri, row in enumerate(label.values):
                    row = row[0].split('\t')
                    t_label[ri, 0] = swap_label[row[1]]

            elif self.task == 'action':
                swap_label = self.label_activity_pair
                t_label = np.zeros((len(label), 2))
                
                for ri, row in enumerate(label.values):
                    row = row[0].split('\t')
                    t_label[ri, 0] = swap_label[row[2]]
                    t_label[ri, 1] = swap_label[row[-3]]

            elif self.task == 'all':
                t_label = np.zeros((len(label), 4))
                for ri, row in enumerate(label.values):
                    row = row[0].split('\t')
                    t_label[ri, 0] = self.label_phase_pair[row[0]]
                    t_label[ri, 1] = self.label_step_pair[row[1]]
                    t_label[ri, 2] = self.label_activity_pair[row[2]]
                    t_label[ri, 3] = self.label_activity_pair[row[-3]]


            labels[label_name] = t_label
            print('label len : ', len(labels[label_name]), labels[label_name].shape)
        
        return labels

