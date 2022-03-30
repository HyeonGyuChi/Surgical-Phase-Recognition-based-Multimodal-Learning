import os
import torch
import pickle
import random
import re
import numpy as np
import pandas as pd
import natsort
from numpy import array
from glob import glob
from PIL import Image

from core.utils.augmentor import Augmentor#, SignalAugmentor


selected_columns = ['psm_1_tt_x','psm_l_tt_y','psm_1_tt_z','psm_1_lv_x','psm_1_lv_y','psm_1_lv_z','psm_1_gra',
                    'psm_2_tt_x','psm_2_tt_y','psm_2_tt_z','psm_2_lv_x','psm_2_lv_y','psm_2_lv_z','psm_2_gra']

all_columns = ['mtm_l_tt_x','mtm_l_tt_y','mtm_l_tt_z',
            'mtm_l_R_11','mtm_l_R_12','mtm_l_R_13','mtm_l_R_21','mtm_l_R_22','mtm_l_R_23','mtm_l_R_31','mtm_l_R_32','mtm_l_R_33',
            'mtm_l_lv_x','mtm_l_lv_y','mtm_l_lv_z',
            'mtm_l_rv_x','mtm_l_rv_y','mtm_l_rv_z',
            'mtm_l_gra',
            'mtm_r_tt_x','mtm_r_tt_y','mtm_r_tt_z',
            'mtm_r_R_11','mtm_r_R_12','mtm_r_R_13','mtm_r_R_21','mtm_r_R_22','mtm_r_R_23','mtm_r_R_31','mtm_r_R_32','mtm_r_R_33',
            'mtm_r_lv_x','mtm_r_lv_y','mtm_r_lv_z',
            'mtm_r_rv_x','mtm_r_rv_y','mtm_r_rv_z',
            'mtm_r_gra',
            'psm_1_tt_x','psm_l_tt_y','psm_1_tt_z',
            'psm_1_R_11','psm_1_R_12','psm_1_R_13',
            'psm_1_R_21','psm_1_R_22','psm_1_R_23',
            'psm_1_R_31','psm_1_R_32','psm_1_R_33',
            'psm_1_lv_x','psm_1_lv_y','psm_1_lv_z',
            'psm_1_rv_x','psm_1_rv_y','psm_1_rv_z',
            'psm_1_gra',
            'psm_2_tt_x','psm_2_tt_y','psm_2_tt_z',
            'psm_2_R_11','psm_2_R_12','psm_2_R_13','psm_2_R_21','psm_2_R_22','psm_2_R_23','psm_2_R_31','psm_2_R_32','psm_2_R_33',
            'psm_2_lv_x','psm_2_lv_y','psm_2_lv_z',
            'psm_2_rv_x','psm_2_rv_y','psm_2_rv_z',
            'psm_2_gra'
            ]


class JIGSAWSDataset(torch.utils.data.Dataset):
    def __init__(self, config, state='train'):
        self.config = config
        self.state = state
        self.label_pair = {'G1':1, 'G2':2, 'G3':3,
                              'G4':4, 'G5':5, 'G6':6,
                              'G8':7, 'G9':8, 'G10':9, 'G11':10}
        
        self.data_path = self.config.data_base_path + '/JIGSAWS_all'
        self.dtype = self.config.data_type
        self.task = self.config.task

        self.users = ['B','C','D','E','F','G','I','H']
        self.trial_no = ['001','002','003','004','005']
        self.set_load_ids()

        self.class_weights = []
        self.data_dict = {}
        self.load_data()

        if self.state == 'train':
            self.aug = Augmentor(self.config.augmentations)
        elif self.state == 'valid':
            self.aug = Augmentor(self.config.val_augmentations)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data = {}
        
        for dtype, _data in self.data_dict.items():
            if dtype == 'video':
                vpath_list = _data[index]

                X = []
                for vpath in vpath_list:
                    img = Image.open(vpath) # RGB
                    img = self.aug(img)
                    X.append(img)

                X = torch.stack(X, dim=0)
                X = X.permute(1, 0, 2, 3) # ch x Seq x H x W
                data[dtype] = X
            elif dtype == 'kinematic':
                signal = _data[index]
                signal = torch.from_numpy(np.array(signal)).float()
                
                data[dtype] = signal

        labels = torch.from_numpy(np.array(self.labels[index])).long()

        return data, labels

    def set_load_ids(self):
        """
            fold 1 ~ 5 -> trial based
            -> fold value : 1 ~ 5
            fold b, c, d, e, f, g, i, h -> user based
            -> fold value : -1 ~ -8
        """
        val_index = self.config.fold
        self.target_list = []
        
        if self.state == 'train':
            if val_index < 0:
                val_index = val_index % 5
                                
                for target in self.trial_no:
                    if target != self.trial_no[val_index]:
                        self.target_list.append(target)
            else:
                for target in self.users:
                    if target != self.users[val_index]:
                        self.target_list.append(target)
        else:
            if val_index < 0:
                val_index = val_index % 5
                
                for target in self.trial_no:
                    if target == self.trial_no[val_index]:
                        self.target_list.append(target)
            else:
                for target in self.users:
                    if target == self.users[val_index]:
                        self.target_list.append(target)

    def load_data(self):
        # load data
        if 'ki' in self.dtype:
            ki_data = self.load_kinematics()
            self.data_dict['kinematic'] = ki_data

        if 'vd' in self.dtype:
            vd_data = self.load_video()
            self.data_dict['video'] = vd_data
            
        self.labels = self.load_labels()

        # preprocessing (subsample, ovelapped data)
        self.preprocessing()
     
    def preprocessing(self):
        # subsample
        sample_rate = self.config.subsample_ratio
        go_subsample = sample_rate > 1 
        
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
                if st+self.config.clip_size < d_len:
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
        """
        if 'SU' in self.task:
            task_name = 'Suturing'

        data_path = self.data_path + '/captured1'

        dir_list = glob(data_path + '/{}*'.format(task_name))
        dir_list = natsort.natsorted(dir_list)

        data = {}

        for dir_name in dir_list:
            _dir_name = re.sub('_capture1', '', dir_name.split('/')[-1])
            chk = False
            for target in self.target_list:
                if target in _dir_name[-4:]:
                    chk = True
                    break

            if not chk:
                continue
            
            print('Load Video {} ...'.format(dir_name))

            file_list = glob(dir_name + '/*.jpg')
            file_list = natsort.natsorted(file_list)

            list_len = len(file_list)
            split = list_len - list_len % 1000
            file_list = file_list[:split]
            
            data[_dir_name] = file_list
            print('frame len : ', len(data[_dir_name]))
        
        return data

    def load_kinematics(self):
        """
            kinematics - 14 variables (left / right)
        """
        if 'SU' in self.task:
            task_name = 'Suturing'
        
        data_path = self.data_path + '/log'
        file_list = glob(data_path + '/{}*'.format(task_name))
        file_list = natsort.natsorted(file_list)

        data = {}

        for fpath in file_list:
            data_name = re.sub('.txt','',fpath.split('/')[-1])
            chk = False
            for target in self.target_list:
                if target in data_name[-4:]:
                    chk = True
                    break

            if not chk:
                continue

            print('Load Signal {} ...'.format(fpath))

            raw_signal_selected = pd.read_csv(fpath,
                                        names=all_columns,
                                        sep='    ')[selected_columns].astype('float64')

            list_len = len(raw_signal_selected)
            split = list_len - list_len % 1000
            raw_signal_selected = raw_signal_selected[:split]

            data[data_name]= self.standardization(raw_signal_selected.values)
            print('signal len : ', len(data[data_name]))

        return data

    def load_labels(self):
        """
            annotations 
        """
        if self.task == 'SU':
            task_name = 'Suturing'

        self.class_weights.append(np.zeros(len(self.label_pair)+1))

        label_path = self.data_path + '/labels'

        label_list = glob(label_path + '/{}*'.format(task_name))
        label_list = natsort.natsorted(label_list)

        labels = {}
        for lab in label_list:
            label_name = re.sub('.txt','',lab.split('/')[-1])
            chk = False
            for target in self.target_list:
                if target in label_name[-4:]:
                    chk = True
                    break

            if not chk:
                continue
            
            print('Label load and masking : ', label_name)

            modal = list(self.data_dict.keys())[0]
            label = pd.read_csv(lab)
            t_label = np.zeros((len(self.data_dict[modal][label_name]), 2))
            frames = np.arange(1, len(t_label)+1, dtype=np.int)

            # label masking
            for row in label.values:
                rows = row[0].split()
                st, ed, l = int(rows[0]), int(rows[1]), rows[2]
                mask = (frames >= st) & (frames <= ed)
                t_label[mask] = self.label_pair[l]

            for lb in range(len(self.class_weights[0])):
                self.class_weights[0][lb] += sum(t_label[:,0] == lb)

            # fit the length between video and signal (w/ label)
            # list_len = len(t_label)
            # split = list_len - list_len % 100
            # t_label = t_label[:split]

            labels[label_name] = pd.DataFrame(t_label, columns=['frame', 'gesture']).values[:, 1]

        bot_sum = 0
        for lb in range(len(self.class_weights[0])):
            bot_sum += self.class_weights[0][lb]
        for lb in range(len(self.class_weights[0])):
            if self.class_weights[0][lb] == 0:
                self.class_weights[0][lb] = 0
            else:
                self.class_weights[0][lb] = bot_sum / (len(self.class_weights[0]) * (self.class_weights[0][lb] + 1e-5))
        self.class_weights[0] = torch.Tensor(self.class_weights[0]).cuda()

        print(self.class_weights)

        return labels

    def split_data(self, X, y,
                     users=['B','C','D','E','F','G','I','H'],
                     trial_no=['001','002','003','004','005'],
                     val_index=0):
        """
            val_index < 0 : trial based
            val_index >= 0 : user based
        """

        train_set = {}
        train_labels ={}
        test_set = {}
        test_labels ={}
        
        if val_index < 0:
            val_index = val_index % 5
            for key in X.keys():
                if key in X and key in y:
                    if trial_no[val_index] in key:
                        test_set[key] = X[key]
                        test_labels[key] = y[key]
                    else:
                        train_set[key] = X[key]
                        train_labels[key] = y[key]
        else:
            for key in X.keys():
                if key in X and key in y:
                    if users[val_index] in key:
                        test_set[key] = X[key]
                        test_labels[key] = y[key]
                    else:
                        train_set[key] = X[key]
                        train_labels[key] = y[key]
        
        return train_set, train_labels, test_set, test_labels


