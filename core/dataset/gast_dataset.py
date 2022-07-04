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


class GastrectomyDataset(torch.utils.data.Dataset):
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
        return len(self.labels)

    def __getitem__(self, index):
        # if self.state == 'train':
        init_label = self.labels[index]
        ids = np.where(np.array(self.labels) == init_label)[0]
        
        while True:
            rand_id = int(np.random.choice(ids, 1))

            check = True
            if rand_id - 16 >= 0 and rand_id + 15 < len(self.labels):
                for ri in range(rand_id - 16, rand_id + 16):
                    if self.labels[ri] != init_label:
                        check = False
                        break
                if check:
                    index = rand_id
                    break

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
                
                if self.args.model == 'slowfast':
                    X = X.permute(1, 0, 2, 3).unsqueeze(0)
                else:
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

    def load_data(self):
        # load specific data
        print('[+] Load data .....')
        self.data_dict = {}
        self.label_dict = {}

        # load procedure (label)
        print('[+] Load labels')
        self.load_labels()
        print('[-] Load labels ... done')

        # load video
        if 'vd' in self.args.data_type:
            print('[+] Load Video data')
            self.load_video()
            print('[-] Load Video data ... done')

        # load segmentation mask
        # if 'mk' in self.args.data_type:
        #     print('[+] Load mask data')
        #     self.load_masks()
        #     print('[-] Load mask data ... done')
    
        # load seg-kinematics
        # if 'ski' in self.args.data_type:
        #     print('[+] Load seg-kinematic data')
        #     self.load_seg_kinematics()
        #     print('[-] Load seg-kinematic data ... done')
        
        print('[-] Load data ..... done')

        # data subsampling and others
        print('[+] Preprocessing data .....')
        self.preprocessing2()
        print('[-] Preprocessing data ..... done')
    
    def load_video(self):
        """
            ? x ? (30Hz)
        """
        self.data_dict['video'] = {}

        target_path = self.data_path + '/gastric/imgs'
        # patient_list = glob(target_path + '/*')
        patient_list = os.listdir(target_path)
        patient_list = natsort.natsorted(patient_list)

        for patient in patient_list:
            key_val = 'R' + patient[-3:]
            p_path = target_path + '/{}'.format(patient)
            
            video_list = natsort.natsorted(os.listdir(p_path))

            # check video and channel
            if key_val in self.target_list:
                t_frame_list = []
                t_label_list = []

                for ch in self.labels[key_val].keys():
                    ch_full_ver = '{}_video_{}'.format(ch[:3], ch[4:])

                    for video in video_list:
                        if ch_full_ver == video:
                            fpath = p_path + '/{}'.format(ch_full_ver)
                            frame_list = glob(fpath + '/*.jpg')
                            frame_list = natsort.natsorted(frame_list)

                            min_len = min(len(frame_list), len(self.labels[key_val][ch]))
                            frame_list = frame_list[:min_len]
                            self.labels[key_val][ch] = self.labels[key_val][ch][:min_len]
                            
                            if len(t_frame_list) == 0:
                                t_frame_list = frame_list
                                t_label_list = self.labels[key_val][ch]
                            else:
                                t_frame_list += frame_list
                                t_label_list += self.labels[key_val][ch]

                self.data_dict['video'][key_val] = t_frame_list
                self.labels[key_val] = t_label_list

                # print(key_val, len(t_frame_list), len(t_label_list))

    def load_masks(self):
        a = '/gastrectomy-40'
        pass

    def load_seg_kinematics(self):
        """
            4 variables (30Hz)
            position : x, y (left, right)
        """
        self.data_dict['kinematic'] = {}

        target_path = self.data_path + '/Seg_kine7'
        file_list = glob(target_path + '/*.pkl')
        file_list = natsort.natsorted(file_list)

        for fpath in file_list:
            key_val = fpath.split('/')[-1].split('_')[0]
            
            if key_val in self.target_list:
                with open(fpath, 'rb') as f:
                    data = pickle.load(f)
                    
                # self.data_dict['kinematic'][key_val] = self.standardization(data[:,:4])
                # self.data_dict['kinematic'][key_val] = self.standardization(data[:,:8])
                # self.data_dict['kinematic'][key_val] = np.concatenate((self.standardization(data[:, :8]), data[:, 8:10]), 1)
                self.data_dict['kinematic'][key_val] = np.concatenate((self.standardization(data[:, :8]), data[:, 8:]), 1)
                # self.data_dict['kinematic'][key_val] = np.concatenate((self.standardization(data[:, :4]), data[:, 8:]), 1)
                
    def load_labels(self):
        self.labels = {}
        
        n_classes = 27
        self.class_weights = []
        self.class_cnt = []
        self.class_cnt.append(np.zeros(n_classes))
        self.class_weights.append(np.zeros(n_classes))
        
        target_path = self.data_path + '/gastric/parser/phase/annotations/armes/frames/ann3.json'

        with open(target_path, 'r') as f:
            data = json.load(f)

        li = list(data.keys())

        for l in li:
            src_patient = 'R_{:03d}'.format(int(l.split('_')[-3]))

            chk = False
            for target in self.target_list:
                tar = '{}_{}'.format(target[0], target[1:])

                if src_patient == tar:
                    tokens = l.split('_')
                    ch_name = '{}_{}'.format(tokens[-2], tokens[-1])
                    t_labels = data[l]['label']

                    for label_idx in range(n_classes):
                        self.class_cnt[0][label_idx] += len(np.where(np.array(t_labels) == label_idx)[0])

                    if target not in self.labels:
                        self.labels[target] = {}
                        self.labels[target][ch_name] = t_labels
                    else:
                        self.labels[target][ch_name] = t_labels

                    chk = True
                    break

                if chk:
                    break

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


    def preprocessing2(self):
        # subsample
        sample_rate = self.args.subsample_ratio
        go_subsample = sample_rate > 1 
        seq_size = self.args.clip_size
        
        if go_subsample:
            for key, _data in self.data_dict.items():
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
                    if len(seq_data[-1]) != self.args.clip_size:
                        print(dir_name, st, len(seq_data[-1]))

            self.data_dict[key] = array(seq_data)

        seq_data = []
        
        for d_num in self.labels.keys():
            data = self.labels[d_num]
            d_len = len(data)

            for st in range(0, d_len, stride):
                seq_data.append(data[st:st+1])

        self.labels = array(seq_data)


