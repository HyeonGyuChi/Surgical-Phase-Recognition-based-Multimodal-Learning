import os
from glob import glob
import cv2
import json
import numpy as np
from tqdm import tqdm
from natsort import natsorted
import shutil
import matplotlib.pyplot as plt
from vis import Visualizer

train_list = ['R001', 'R002', 'R005', 'R007', 'R010', 
                'R014', 'R015', 'R019', 'R048', 'R056', 
                'R074', 'R076', 'R084', 'R094', 'R100', 
                'R117', 'R201', 'R202', 'R203', 'R204', 
                'R205', 'R206', 'R207', 'R209', 'R210', 
                'R301', 'R302', 'R304', 'R305', 'R313']
val_list = ['R003', 'R004', 'R006', 'R013', 'R017', 'R018', 'R022', 'R116', 'R208', 'R303']
target_list = train_list + val_list
                


bb = '/dataset3/multimodal/gastric/grastric_train_split_1_rawframes.txt'
bb2 = '/dataset3/multimodal/gastric/grastric_val_split_1_rawframes.txt'

ss = '/dataset3/multimodal/gastric/gastric_train_1_rawframes.txt'
ss2 = '/dataset3/multimodal/gastric/gastric_val_1_rawframes.txt'


with open(bb, 'r') as f:
    data = f.readlines()

with open(bb2, 'r') as f:
    data2 = f.readlines()

di = {}

for d in data:
    p = d.split('_')[4]
    p2 = 'R{:03d}'.format(int(p))

    if p2 not in di and p2 in target_list:
        di[p2] = [d]
    elif p in di:
        di[p2].append(d)


for d in data2:
    p = d.split('_')[4]
    p2 = 'R{:03d}'.format(int(p))

    if p2 not in di and p2 in target_list:
        di[p2] = [d]
    elif p2 in di:
        di[p2].append(d)


for k, v in di.items():
    if k in train_list:
        with open(ss, 'a') as f:
            for v2 in v:
                f.writelines(v2)

    elif k in val_list:
        with open(ss2, 'a') as f:
            for v2 in v:
                f.writelines(v2)



# bpath = '/dataset3/multimodal/gastric/parser/phase/annotations/armes/frames/ann3.json'



# with open(bpath, 'r') as f:
#     data = json.load(f)

# li = list(data.keys())
# t_dict = {}

# for l in li:
#     chk = False
#     for target in target_list:
#         tar = '{}_{}'.format(target[0], target[1:])

#         if tar in l:
#             tokens = l.split('_')
#             ch, ch_no = tokens[-2], tokens[-1]

#             if target not in t_dict:
#                 t_dict[target] = {}
#                 t_dict[target]['{}_{}'.format(ch, ch_no)] = data[l]['label']
#             else:
#                 t_dict[target]['{}_{}'.format(ch, ch_no)] = data[l]['label']

#             chk = True
#             break
#         if chk:
#             break

# vis = Visualizer()

# for patient in t_dict.keys():
#     p_dict = t_dict[patient]

#     for video in p_dict.keys():
#         v_dict = p_dict[video]

#         gt = v_dict
#         pred = gt

#         print(patient, video, len(v_dict))

#         fig, ax = plt.subplots(1,1,figsize=(16, 7))
#         ax = vis.draw_presence_bar(ax, 'yaho', gt, [pred])

#         plt.savefig('./yaho.png')
#         break

#     break