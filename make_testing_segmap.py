import os
import natsort
from glob import glob
from tqdm import tqdm
import shutil
import cv2
import numpy as np


base_path = '/dataset3/multimodal/PETRAW/Test'
img_dir = base_path + '/Img'
anno_dir = base_path + '/Segmentation'


patient_list = natsort.natsorted(os.listdir(img_dir))
split = len(patient_list) // 40

print(patient_list)


NUM_CLASSES = 7
valid_classes = [0, 29, 76, 105, 150, 255] # bg, peg, left, block, ?, base
class_map = dict(zip(valid_classes, range(NUM_CLASSES)))
except_valid_classes = []

for i in range(255):
    if not i in valid_classes:
        except_valid_classes.append(i)
except_class_map = dict(zip(except_valid_classes, [6]*len(except_valid_classes)))

PALETTE = [[0, 0, 255], # pegs -> 
            [255, 0, 0], # left_inst -> 1
            [255, 0, 255], # blocks -> 105 (3)
            [0, 255, 0], # right_inst -> 
            [255, 255, 255], # base -> 
            ]

def encode_segmap(mask):
    # Put all void classes to zero
    for _validc in except_valid_classes:
        mask[mask == _validc] = except_class_map[_validc]
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]

    return mask

def mapping(img):
    h, w, _ = img.shape
    new_img = np.zeros((h, w))

    r = img[:,:,2] > 0
    g = img[:,:,1] > 0
    b = img[:,:,0] > 0

    base = r & b & g
    left = r & ~g & ~b
    right = ~r & g & ~b
    peg = ~r & ~g & b
    block = r & b & ~g
    
    
    ids = np.where(base == True)
    new_img[ids[0], ids[1]] = 1
    ids = np.where(left == True)
    new_img[ids[0], ids[1]] = 2
    ids = np.where(right == True)
    new_img[ids[0], ids[1]] = 3
    ids = np.where(peg == True)
    new_img[ids[0], ids[1]] = 4
    ids = np.where(block == True)
    new_img[ids[0], ids[1]] = 5


    return new_img


for idx, patient in enumerate(patient_list):
    save_path = base_path + '/test/img_{}'.format(idx+1)
    save_path2 = base_path + '/test/seg_{}'.format(idx+1)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path2, exist_ok=True)

    dir_path = img_dir + '/{}'.format(patient)
    dir_path2 = anno_dir + '/{}'.format(patient)

    print('test patient : {}'.format(patient))

    img_list = natsort.natsorted(os.listdir(dir_path))#[::6]
    seg_list = natsort.natsorted(os.listdir(dir_path2))#[::6]

    cnt = 0

    for img_name in tqdm(img_list):
        spath = '{}/{}_{}'.format(save_path, patient, img_name)
        spath2 = '{}/{}_{}'.format(save_path2, patient, img_name[:-4] + '.png')

        shutil.copy(dir_path + '/{}'.format(img_name), spath)

        seg = cv2.imread(dir_path2 + '/{}'.format(seg_list[cnt]), 0) 
        # seg = cv2.imread(dir_path2 + '/{}'.format(seg_list[cnt]))        
        new_seg = encode_segmap(seg)
        # new_seg = mapping(seg)
        # new_seg = cv2.resize(new_seg, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(spath2, new_seg)
        # shutil.copy(dir_path2 + '/{}'.format(seg_list[cnt]), spath2)

        cnt += 1
