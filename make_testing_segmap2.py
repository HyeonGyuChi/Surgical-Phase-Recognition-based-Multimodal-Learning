import os
import natsort
from glob import glob
from tqdm import tqdm
import shutil
import cv2
import numpy as np

NUM_CLASSES = 32
valid_classes = [0, 29, 76, 105, 150, 255] # bg, peg, left, block, ?, base
class_map = dict(zip(valid_classes, range(NUM_CLASSES)))
except_valid_classes = []

for i in range(255):
    if not i in valid_classes:
        except_valid_classes.append(i)
except_class_map = dict(zip(except_valid_classes, [6]*len(except_valid_classes)))

PALETTE = [[0, 0, 0],
                [251, 244, 5], [37, 250, 5],[0, 21, 209],
                [172, 21, 2],[172, 21, 229],
                [6, 254, 249],[141, 216, 23],[96, 13, 13],
                [65, 214, 24],[124, 3, 252],[214, 55, 153],[48, 61, 173],
                [110, 31, 254],[249, 37, 14],[249, 137, 254],
                [34, 255, 113],[169, 52, 14],
                [124, 49, 176],[4, 88, 238],
                [115, 214, 178],[115, 63, 178],
                [115, 214, 235],[63, 63, 178],
                [130, 34, 26],[220, 158, 161],
                [201, 117, 56],[121, 16, 40],
                [15, 126, 0],[224, 224, 224],
                [154, 0, 0],[204, 102, 0]]

def encode_segmap(mask):
    # Put all void classes to zero
    for _validc in except_valid_classes:
        mask[mask == _validc] = except_class_map[_validc]
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]

    return mask


target_list = ['R001', 'R002', 'R003', 'R004', 'R005', 
                'R006', 'R007', 'R010', 'R013',
                'R014', 'R015', 'R017', 'R018', 'R019', 
                'R022', 'R048', 'R056', 
                'R074', 'R076', 'R084', 'R094', 'R100', 
                'R117', 'R201', 'R202', 'R203', 'R204', 
                'R205', 'R206', 'R207', 'R209', 'R210', 'R208',
                'R301', 'R302', 'R303', 'R304', 'R305', 'R313']


base_path = '/dataset3/multimodal/gastric'
img_dir = base_path + '/os_frames'
save_path = base_path + '/segmentation'

patient_list = natsort.natsorted(os.listdir(img_dir))

cnt = 0
idx = 0

for patient in patient_list:
    tp = 'R' + patient[-3:]
    if tp not in target_list:
        continue

    p_path = img_dir + f'/{patient}'

    video_list = natsort.natsorted(os.listdir(p_path))

    for video_name in video_list:
        v_path = p_path + f'/{video_name}'
        img_list = natsort.natsorted(os.listdir(v_path))#[::6]

        print(patient, video_name)

        for img_name in tqdm(img_list):
            if cnt % 40000 == 0:
                cnt = 0
                idx += 1

            save_path = base_path + '/test2/seg_{}'.format(idx)
            os.makedirs(save_path, exist_ok=True)
            
            img_path = v_path + f'/{img_name}'
            save_path2 = save_path + f'/{patient}_{video_name}_{img_name}'
            shutil.copy(img_path, save_path2)

            # img = cv2.imread(img_path, 0) 
            # new_seg = encode_segmap(img)
            # new_seg = cv2.resize(new_seg, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
            
            # cv2.imwrite(spath2, new_seg)
            # shutil.copy(dir_path2 + '/{}'.format(seg_list[cnt]), spath2)

            cnt += 1
    
    # dir_path2 = anno_dir + '/{}'.format(patient)

    print('test patient : {}'.format(patient))


    
