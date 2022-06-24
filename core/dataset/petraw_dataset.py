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
from preprocess.recon_kinematic_filter import recon_kinematic_filter



class PETRAWDataset(torch.utils.data.Dataset):
    def __init__(self, args, state='train'):
        self.args = args
        self.state = state
        self.data_path = args.data_base_path + '/PETRAW'
        self.task = args.task

        self.num_of_ski_feature = 0

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

        if self.state == 'train':
            self.data_path += '/Training'
            self.aug = Augmentor(self.args.augmentations)
            self.mk_aug = Augmentor(self.args.mask_augmentations)

        elif self.state == 'valid':
            self.data_path += '/Test'
            self.aug = Augmentor(self.args.val_augmentations)
            self.mk_aug = Augmentor(self.args.mask_augmentations)

        self.set_load_ids()
        self.load_data()
        
    def __len__(self):
        return len(self.labels) #// 10

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
                # X = X.permute(1, 0, 2, 3).unsqueeze(0)
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
        val_index = self.args.fold
        self.target_list = []
        
        target_path = self.data_path + '/Procedural_description'
        file_list = os.listdir(target_path)
        file_list = natsort.natsorted(file_list)
        
        f_len = len(file_list)
        split = f_len // 5
        
        st = 0 + (val_index-1) * split # fold1=[0:29] // 29ea => 1~29 case
        ed = 0 + val_index * split
        if val_index == 5: # fold5=[116:149] // 33ea
            ed = f_len
        
        if self.state == 'train':
            for target in range(f_len):
                if not (st <= target and target < ed):
                    self.target_list.append(file_list[target][:-4]) # split [.jpg, .png]
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
        if 'vd' in self.args.data_type:
            print('[+] Load Video data')
            self.load_video()
            print('[-] Load Video data ... done')

        # load segmentation mask
        if 'mk' in self.args.data_type:
            print('[+] Load mask data')
            self.load_masks()
            print('[-] Load mask data ... done')

        # load kinematics
        if 'ki' in self.args.data_type:
            print('[+] Load kinematic data')
            self.load_kinematics()
            print('[-] Load kinematic data ... done')
            
        # load seg-kinematics
        if 'ski' in self.args.data_type:
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
        self.preprocessing2()
        print('[-] Preprocessing data ..... done')

    def preprocessing(self):
        # subsample
        sample_rate = self.args.subsample_ratio
        go_subsample = sample_rate > 1 
        seq_size = self.args.clip_size
        
        if go_subsample:
            for key, _data in self.data_dict.items(): # 'video', 'mask', 'kinematic'
                for dir_name in _data.keys():
                    self.data_dict[key][dir_name] = _data[dir_name][::sample_rate] # default = 6 == 5fps

            for dir_name in self.labels.keys():
                self.labels[dir_name] = self.labels[dir_name][::sample_rate]

        # overlapping data sequence
        if self.state != 'train': # val
            stride = int(seq_size)
        else: # train
            stride = int(seq_size * self.args.overlap_ratio)
        
        for key, _data in self.data_dict.items(): # 'video', 'mask', 'kinematic'
            seq_data = []
            
            for dir_name in _data.keys(): # '033', '045', ...
                d_len = len(_data[dir_name])
                
                for st in range(0, d_len, stride):
                    if st+seq_size < d_len:
                        seq_data.append(_data[dir_name][st:st+seq_size]) # clips
                    else:
                        break

            self.data_dict[key] = array(seq_data) # include all case clips
        
        seq_data = []
        
        for d_num in self.labels.keys():
            data = self.labels[d_num]
            d_len = len(data)

            for st in range(0, d_len, stride):
                if st+seq_size < d_len:
                    if self.args.inference_per_frame:
                        seq_data.append(data[st:st+seq_size])
                    else:
                        seq_data.append(data[st:st+1])
                else:
                    break

        self.labels = array(seq_data)
    
    def preprocessing2(self):
        # subsample
        sample_rate = self.args.subsample_ratio
        go_subsample = sample_rate > 1 
        seq_size = self.args.clip_size
        
        if go_subsample:
            for key, _data in self.data_dict.items():
                if key == 'kinematic': # pass subsampling on ski
                    print('\t ---> pass subsampling on kinematic')
                    for dir_name in _data.keys():
                        self.data_dict[key][dir_name] = _data[dir_name][:]
                else:
                    for dir_name in _data.keys():
                        self.data_dict[key][dir_name] = _data[dir_name][::sample_rate]


            for dir_name in self.labels.keys():
                self.labels[dir_name] = self.labels[dir_name][::sample_rate]

        # overlapping data sequence
        # if self.state != 'train':
        #     stride = int(seq_size)
        # else:
        #     stride = int(seq_size * self.args.overlap_ratio)
        stride = 1

        hf_seq = self.args.clip_size // 2

        for key, _data in self.data_dict.items():
            seq_data = []
            
            for dir_name in _data.keys():
                d_len = len(_data[dir_name])
                
                for st in range(0, d_len, stride):
                    if st-hf_seq < 0:
                        diff = hf_seq-st

                        if key == 'video':
                            seq = [_data[dir_name][0] for _ in range(diff)] + _data[dir_name][st:st+(self.args.clip_size-diff)]

                            # print(seq)
                        else:
                            pad = np.zeros((diff, *_data[dir_name][0].shape))

                            seq = np.concatenate((pad, _data[dir_name][st:st+(self.args.clip_size-diff)]), 0)
                        seq_data.append(seq)

                    elif st+hf_seq >= d_len:
                        diff = st+hf_seq-d_len

                        if key == 'video':
                            seq = _data[dir_name][st-hf_seq:] + [_data[dir_name][-1] for _ in range(diff)]
                        else:
                            pad = np.zeros((diff, *_data[dir_name][-1].shape))

                            seq = np.concatenate((_data[dir_name][st-hf_seq:], pad), 0)
                        seq_data.append(seq)

                    else:
                        seq_data.append(_data[dir_name][st-hf_seq:st+hf_seq])

                    # if seq_data[-1].shape[0] != 8:
                        # print(dir_name, st, seq_data[-1].shape)
                    if len(seq_data[-1]) != 8:
                        print(dir_name, st, len(seq_data[-1]))

            self.data_dict[key] = array(seq_data)

        seq_data = []
        
        for d_num in self.labels.keys():
            data = self.labels[d_num]
            d_len = len(data)

            for st in range(0, d_len, stride):
                seq_data.append(data[st:st+1])

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

        target_path = self.data_path + '/Seg_kine11-5fps'
        file_list = glob(target_path + '/*.pkl')
        file_list = natsort.natsorted(file_list)

        # filter of visual kinematic data
        rk_filter = recon_kinematic_filter(task='PETRAW')

        for fpath in file_list:
            key_val = fpath.split('/')[-1].split('_')[0]
            
            if key_val in self.target_list:
                '''
                with open(fpath, 'rb') as f:
                    data = pickle.load(f)
                '''
                
                rk_filter.set_src_path(fpath) # set .pkl (pd.Dataframe)
                data = rk_filter.filtering(self.args.ski_methods, extract_objs=['Grasper'], extract_pairs=[('Grasper', 'Grasper')])

                # print('dshape:', data.shape)
                # print('col num: ', len(data.columns))
                # print('col', data.columns)
                # print('nump: ', data.to_numpy().shape)
                data = data.to_numpy() # df to np

                self.data_dict['kinematic'][key_val] = data # no more standradization
                self.num_of_ski_feature = data.shape[1] # num of feature 

                # self.data_dict['kinematic'][key_val] = self.standardization(data[:,:4])
                # self.data_dict['kinematic'][key_val] = self.standardization(data[:,:8])
                # self.data_dict['kinematic'][key_val] = np.concatenate((self.standardization(data[:, :8]), data[:, 8:10]), 1)
                # self.data_dict['kinematic'][key_val] = np.concatenate((self.standardization(data[:, :28]), data[:, 28:]), 1) # 요기서 load
                # self.data_dict['kinematic'][key_val] = np.concatenate((self.standardization(data[:, :4]), data[:, 8:]), 1)

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
        
        for n_classes in [3, 13, 7, 7]: # phase, step, left action, right action
            self.class_cnt.append(np.zeros(n_classes))
            self.class_weights.append(np.zeros(n_classes))

        for fpath in file_list:
            key_val = fpath.split('/')[-1][:-4]
            
            if key_val in self.target_list:
                df = pd.read_csv(fpath, sep='\t')
                tmp_labels = df.values[:, 1:] # numpy, shape(:, 4) // Phase, step, verb_l, verb_r
                
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
        for idx in range(4): # phase, step, verb l, verb r
            
            if len(self.class_cnt[idx]):
                bot_sum = 0
                n_classes = len(self.class_cnt[idx])

                for idx2 in range(n_classes):
                    bot_sum += self.class_cnt[idx][idx2]

                    if idx >= 2: # verb l, verb r
                        # print(idx2, self.class_cnt[idx][idx2])
                        pass
                    
                for idx2 in range(n_classes):
                    self.class_weights[idx][idx2] = bot_sum / (n_classes * self.class_cnt[idx][idx2]) # more cnt to less weight
                
                if idx < 2: # pahse, action
                    self.class_weights[idx] = torch.Tensor(np.ones(len(self.class_cnt[idx]))).cuda()
                else: # verb l, verb r
                    self.class_weights[idx] = torch.Tensor(self.class_cnt[idx]).cuda()

            print('CLS WEIGHTS - ', idx, ' : ',  self.class_cnt[idx], self.class_weights[idx])

    def standardization(self, x):
        
        mean_x = np.mean(x, 0)
        std_x = np.std(x, 0)

        return (x-mean_x) / (std_x + 1e-5)



