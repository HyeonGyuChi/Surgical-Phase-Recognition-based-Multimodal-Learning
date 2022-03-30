import os
import numpy as np
import torch
import natsort
import pandas as pd
from glob import glob
from PIL import Image
import cv2
from numpy import array
import pickle

from core.utils.augmentor import Augmentor#, SignalAugmentor



class PETRAWDataset(torch.utils.data.Dataset):
    def __init__(self, config, state='train'):
        self.config = config
        self.state = state
        self.data_path = config.data_base_path + '/PETRAW'
        self.task = config.task

        self.name_to_phase = { # 3
            'Idle': 0,
            'Transfer Left to Right': 1,
            'Transfer Right to Left': 2,
        }
        self.name_to_step = { # 13
            'Idle': 0, 
            'Block 1 L2R': 1, 'Block 2 L2R': 2, 'Block 3 L2R': 3,
            'Block 4 L2R': 4, 'Block 5 L2R': 5, 'Block 6 L2R': 6,
            'Block 1 R2L': 7, 'Block 2 R2L': 8, 'Block 3 R2L': 9,
            'Block 4 R2L': 10, 'Block 5 R2L': 11, 'Block 6 R2L': 12, 
        }
        self.name_to_verb = { # 7
            'Idle': 0, 'Catch': 1, 'Drop': 2, 
            'Extract': 3, 'Hold': 4, 'Insert': 5, 'Touch': 6,
        }
        colors = [
            [255, 255, 255], # base
            [255, 0, 0], # left_inst
            [0, 255, 0], # right_inst
            [0, 0, 255], # pegs
            [255, 0, 255], # blocks
        ]
        self.seg_to_colors = dict(zip(range(1,6), colors))
        self.data_path += '/Training'
        self.set_load_ids()

        self.load_data()

        if self.state == 'train':
            self.aug = Augmentor(self.config.augmentations)
            self.mk_aug = Augmentor(self.config.mask_augmentations)

        elif self.state == 'valid':
            self.aug = Augmentor(self.config.val_augmentations)
            self.mk_aug = Augmentor(self.config.mask_augmentations)
        
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
                    # img = cv2.imread(vpath) # h, w, ch
                    # X.append(img[:,:,::-1])

                X = self.aug(X)
                
                X = torch.stack([torch.Tensor(_X) for _X in X], dim=0)                
                X = X.permute(1, 0, 2, 3)

                data[dtype] = X
            elif dtype == 'mask':
                vpath_list = self.data_dict[dtype][index]

                X = []
                for vpath in vpath_list:
                    # img = Image.open(vpath) # h, w, ch
                    img = np.load(vpath)['arr_0']
                    X.append(img)
                    
                X = torch.stack([torch.Tensor(_X.copy()) for _X in X], dim=0)
                
                data[dtype] = X
            elif dtype == 'kinematic':
                signal = self.data_dict[dtype][index]
                data[dtype] = torch.from_numpy(np.array(signal)).float()

        labels = torch.from_numpy(np.array(self.labels[index])).long()

        return data, labels

    def set_load_ids(self):
        """
            fold 1 ~ 5
        """
        val_index = self.config.fold
        self.target_list = []
        
        target_path = self.data_path + '/Procedural_description'
        file_list = os.listdir(target_path)
        file_list = natsort.natsorted(file_list)
        
        f_len = len(file_list)
        split = f_len // 5
        
        st = 0 + (val_index-1) * split
        ed = 0 + val_index * split
        if val_index == 5:
            ed = f_len
        
        if self.state == 'train':
            for target in range(f_len):
                if not (st <= target and target < ed):
                    self.target_list.append(file_list[target][:-4])
        else:
            for target in range(f_len):
                if (st <= target and target < ed):
                    self.target_list.append(file_list[target][:-4])

    def load_data(self):
        # load specific data
        print('[+] Load data .....')
        self.data_dict = {}
        self.label_dict = {}

        # load video
        if 'vd' in self.config.data_type:
            print('[+] Load Video data')
            self.load_video()
            print('[-] Load Video data ... done')

        # load segmentation mask
        if 'mk' in self.config.data_type:
            print('[+] Load mask data')
            self.load_masks()
            print('[-] Load mask data ... done')

        # load kinematics
        if 'ki' in self.config.data_type:
            print('[+] Load kinematic data')
            self.load_kinematics()
            print('[-] Load kinematic data ... done')
            
        # load seg-kinematics
        if 'ski' in self.config.data_type:
            print('[+] Load seg-kinematic data')
            self.load_seg_kinematics()
            print('[-] Load seg-kinematic data ... done')

        # load procedure (label)
        print('[+] Load labels')
        self.load_labels()
        print('[-] Load labels ... done')
        print('[-] Load data ..... done')

        # data subsampling and others
        print('[+] Preprocessing data .....')
        self.preprocessing()
        print('[-] Preprocessing data ..... done')

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
            stride = int(seq_size * self.config.overlap_ratio)

        for key, _data in self.data_dict.items():
            seq_data = []
            
            for dir_name in _data.keys():
                d_len = len(_data[dir_name])
                
                for st in range(0, d_len, stride):
                    if st+seq_size < d_len:
                        seq_data.append(_data[dir_name][st:st+seq_size])
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
                        seq_data.append(data[st:st+seq_size])
                    else:
                        seq_data.append(data[st:st+1])
                else:
                    break

        self.labels = array(seq_data)
        
    def load_video(self):
        """
            1920 x 1080 (30Hz)
        """
        self.data_dict['video'] = {}

        target_path = self.data_path + '/Video'
        file_list = glob(target_path + '/*')
        file_list = natsort.natsorted(file_list)

        for fpath in file_list:
            key_val = fpath.split('/')[-1]
            
            if key_val in self.target_list:
                frame_list = glob(fpath + '/*.jpg')
                frame_list = natsort.natsorted(frame_list)

                self.data_dict['video'][key_val] = frame_list

    def load_masks(self):
        """
            Background : (0, 0, 0)
            Base : (255, 255, 255)
            Left_instrument : (255, 0, 0)
            Right_instrument : (0, 255, 0)
            Pegs : (0, 0, 255)
            Blocks : (255, 0, 255)
        """
        self.data_dict['mask'] = {}

        # target_path = self.data_path + '/Segmentation'
        target_path = self.data_path + '/Segmentation2'
        file_list = os.listdir(target_path)
        file_list = natsort.natsorted(file_list)

        for key_val in file_list:
            if key_val in self.target_list:
                dpath = target_path + '/{}'.format(key_val)
                frame_list = glob(dpath + '/*.npz')
                # frame_list = glob(dpath + '/*.png')
                frame_list = natsort.natsorted(frame_list)

                self.data_dict['mask'][key_val] = frame_list

    def load_kinematics(self):
        """
            14 variables (30Hz)
            position : x, y, z (cm)
            rotation : q1, q2, q3, q4
            forceps aperture angle : ape_angle
            linear velocity : lin_velo_x,y,z (cm)
            angular vecocity : ang_velo_x,y,z (deg)
            columns (28 (14/14))
            px, py, pz, q1, q2, q3, q4, ape_angle, lin_velo_x, lin_velo_y, lin_velo_z, ang_velo_x, ang_velo_y, ang_velo_z (left/right)
        """
        self.data_dict['kinematic'] = {}

        target_path = self.data_path + '/Kinematic'
        file_list = glob(target_path + '/*.kinematic')
        file_list = natsort.natsorted(file_list)

        for fpath in file_list:
            key_val = fpath.split('/')[-1][:-10]
            
            if key_val in self.target_list:
                df = pd.read_csv(fpath, sep='\t')
                self.data_dict['kinematic'][key_val] = self.standardization(df.values[:, 1:])
                
    def load_seg_kinematics(self):
        """
            4 variables (30Hz)
            position : x, y (left, right)
        """
        self.data_dict['kinematic'] = {}

        target_path = self.data_path + '/Seg_kine4'
        file_list = glob(target_path + '/*.pkl')
        file_list = natsort.natsorted(file_list)

        for fpath in file_list:
            key_val = fpath.split('/')[-1].split('_')[0]
            
            if key_val in self.target_list:
                with open(fpath, 'rb') as f:
                    data = pickle.load(f)
                    
                self.data_dict['kinematic'][key_val] = self.standardization(data)
                # print(key_val, len(self.data_dict['kinematic'][key_val]))
                

    def load_labels(self):
        """
            Phase : 2
            - Idle, X, X
            Step : 12
            - Idle, X, ...
            Action verbs : 6
            - Idle, Catch, Hold, ...
            Target : 1
            Surgical instrument : 1
            Columns
            Phase, Step, Verb_Left, Verb_right
        """
        self.labels = {}

        target_path = self.data_path + '/Procedural_description'
        file_list = glob(target_path + '/*.txt')
        file_list = natsort.natsorted(file_list)
        
        self.class_weights = []
        self.class_cnt = []
        
        for n_classes in [3, 13, 7, 7]:
            self.class_cnt.append(np.zeros(n_classes))
            self.class_weights.append(np.zeros(n_classes))

        for fpath in file_list:
            key_val = fpath.split('/')[-1][:-4]
            
            if key_val in self.target_list:
                df = pd.read_csv(fpath, sep='\t')
                tmp_labels = df.values[:, 1:]
                
                if self.task == 'phase':
                    labels = np.zeros((len(tmp_labels)))
                    
                    for vi, val in enumerate(tmp_labels[:, 0]):
                        labels[vi] = self.name_to_phase[val]
                        self.class_cnt[0][int(labels[vi])] += 1
                elif self.task == 'step':
                    labels = np.zeros((len(tmp_labels)))

                    for vi, val in enumerate(tmp_labels[:, 1]):
                        labels[vi] = self.name_to_step[val]
                        self.class_cnt[1][int(labels[vi])] += 1
                elif self.task == 'action':
                    labels = np.zeros((len(tmp_labels), 2))

                    for vi, val in enumerate(tmp_labels[:, 2:]):
                        labels[vi, 0] = self.name_to_verb[val[0]]
                        labels[vi, 1] = self.name_to_verb[val[1]]
                        self.class_cnt[2][int(labels[vi, 0])] += 1
                        self.class_cnt[3][int(labels[vi, 1])] += 1

                elif self.task == 'all':
                    labels = np.zeros((len(tmp_labels), 4))

                    for vi, val in enumerate(tmp_labels[:, ]):
                        labels[vi, 0] = self.name_to_phase[val[0]]
                        labels[vi, 1] = self.name_to_step[val[1]]
                        labels[vi, 2] = self.name_to_verb[val[2]]
                        labels[vi, 3] = self.name_to_verb[val[3]]

                        for ci in range(4):
                            self.class_cnt[ci][int(labels[vi, ci])] += 1

                self.labels[key_val] = labels

        # class weight computation
        for idx in range(4):
            if len(self.class_cnt[idx]):
                bot_sum = 0
                n_classes = len(self.class_cnt[idx])

                for idx2 in range(n_classes):
                    bot_sum += self.class_cnt[idx][idx2]

                    if idx >= 2:
                        print(idx2, self.class_cnt[idx][idx2])
                    
                for idx2 in range(n_classes):
                    self.class_weights[idx][idx2] = bot_sum / (n_classes * self.class_cnt[idx][idx2])
                
                if idx < 2:
                    self.class_weights[idx] = torch.Tensor(np.ones(len(self.class_cnt[idx]))).cuda()
                else:
                    self.class_weights[idx] = torch.Tensor(self.class_cnt[idx]).cuda()

            print('CLS WEIGHTS - ', idx, ' : ',  self.class_cnt[idx], self.class_weights[idx])

    def standardization(self, x):
        mean_x = np.mean(x, 0)
        std_x = np.std(x, 0)

        return (x-mean_x) / (std_x + 1e-5)



