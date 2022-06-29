import os
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
from natsort import natsorted
import shutil


# base_path = 'logs/ocr_petraw/results_test'
# base_path = 'logs/deeplab_petraw_test/results_test'
# base_path = 'logs/ocr_petraw_test/results_test'
base_path = 'logs/swin_petraw_test/results_test'
# ref_path = '/dataset3/multimodal/PETRAW/Training/Segmentation2'
# save_path = '/dataset3/multimodal/PETRAW/Training/Segmentation_deeplabv3'
ref_path = '/dataset3/multimodal/PETRAW/Test/Segmentation'
save_path = '/dataset3/multimodal/PETRAW/Test/Segmentation_ocr'
# save_path = '/dataset3/multimodal/PETRAW/Test/Segmentation_deeplabv3'
# save_path = '/dataset3/multimodal/PETRAW/Test/Segmentation_swin'

ref_list = natsorted(os.listdir(ref_path))

tot_len = 0
tot_len2 = 0

for ref in ref_list:
    spath = os.path.join(ref_path, ref)
    spath2 = os.path.join(save_path, ref)
    r = os.listdir(spath)
    r2 = os.listdir(spath2)

    tot_len += len(r)
    tot_len2 += len(r2)

    print(ref, len(r), len(r2), len(r)-len(r2))

print(tot_len, tot_len2)

# for ref in ref_list:
#     spath = os.path.join(save_path, ref)
#     os.makedirs(spath, exist_ok=True)

# img_list = glob(base_path + '/*')

# for img_path in tqdm(img_list):
#     img_name = img_path.split('/')[-1]
#     no = img_name.split('_')[0]

#     spath = os.path.join(save_path, no, img_name)

#     shutil.move(img_path, spath)


