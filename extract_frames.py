import os
import natsort
from glob import glob
from tqdm import tqdm
import shutil
import cv2
import numpy as np


base_path = '/dataset3/multimodal/PETRAW/Test'
img_dir = base_path + '/Video'
anno_dir = base_path + '/Segmentation'
save_path = base_path + '/img'

patient_list = natsort.natsorted(os.listdir(img_dir))

for patient in patient_list:
    v_path = os.path.join(img_dir, patient)
    cap = cv2.VideoCapture(v_path)
    t_len = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    spath = save_path + '/{}'.format(patient[:-4])
    os.makedirs(spath, exist_ok=True)

    cnt = 1
    while True:
        ret, frame = cap.read()

        if ret:
            spath2 = spath + '/{:06d}.jpg'.format(cnt)
            cv2.imwrite(spath2, frame)
            cnt += 1

            if cnt % 1000 == 0:
                print('cur processed : {:.4f}%'.format(cnt / t_len * 100))
        else:
            break

    cap.release()
        