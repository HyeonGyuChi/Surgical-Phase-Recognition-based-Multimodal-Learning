import os
from glob import glob
import cv2
import json
import numpy as np
from tqdm import tqdm
from natsort import natsorted
import shutil


bpath = '/dataset3/multimodal/gastric/parser/phase/annotations/armes/frames/ann3.json'

target_list = ['R001', 'R002', 'R005', 'R007', 'R010', 
                'R014', 'R015', 'R019', 'R048', 'R056', 
                'R074', 'R076', 'R084', 'R094', 'R100', 
                'R117', 'R201', 'R202', 'R203', 'R204', 
                'R205', 'R206', 'R207', 'R209', 'R210', 
                'R301', 'R302', 'R304', 'R305', 'R313']

with open(bpath, 'r') as f:
    data = json.load(f)

li = list(data.keys())
t_dict = {}

for l in li:
    chk = False
    for target in target_list:
        tar = '{}_{}'.format(target[0], target[1:])

        if tar in l:
            tokens = l.split('_')
            ch, ch_no = tokens[-2], tokens[-1]

            if target not in t_dict:
                t_dict[target] = {}
                t_dict[target]['{}_{}'.format(ch, ch_no)] = data[l]['label']
            else:
                t_dict[target]['{}_{}'.format(ch, ch_no)] = data[l]['label']

            chk = True
            break
        if chk:
            break

print(len(t_dict['R100']['ch1_01']))
        




# bpath = '/dataset3/multimodal/gastrectomy-40'

# txt_list = natsorted(glob(bpath + '/*.txt'))

# split_dict = {
#     0: [],
#     1: [],
#     2: [],
#     3: [],
#     4: [],
#     5: [],
# }

# for ti, tpath in enumerate(txt_list):
#     with open(tpath, 'r') as f:
#         data = f.readlines()

#     data = natsorted(data)

#     for d in data:
#         patient = d.split('/')[-1].split('_')[0]
#         if patient not in split_dict[ti]:
#             split_dict[ti].append(patient)

#     print(ti, split_dict[ti])
