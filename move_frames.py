import os
import numpy as np
import natsort
import shutil
from tqdm import tqdm



target_list = ['R001', 'R002', 'R005', 'R007', 'R010', 
                'R014', 'R015', 'R019', 'R048', 'R056', 
                'R074', 'R076', 'R084', 'R094', 'R100', 
                'R117', 'R201', 'R202', 'R203', 'R204', 
                'R205', 'R206', 'R207', 'R209', 'R210', 
                'R301', 'R302', 'R304', 'R305', 'R313', 
                'R003', 'R004', 'R006', 'R013', 'R017', 
                'R018', 'R022', 'R116', 'R208', 'R303']


base_path = '/dataset3/multimodal/gastric'


src_path = base_path + '/os_frames'
p_list = os.listdir(src_path)

storage = 0.

for patient in p_list:
    tp = 'R'+patient[-3:]

    if tp in target_list:
        p_path = src_path + f'/{patient}'
        v_list = natsort.natsorted(os.listdir(p_path))

        for video in v_list:
            v_path = p_path + f'/{video}'
            f_list = natsort.natsorted(os.listdir(v_path))

            tar_path = base_path + f'/imgs/{patient}/{video}'
            os.makedirs(tar_path, exist_ok=True)

            print('start : ', patient, video)
            for frame in tqdm(f_list):
                s_path = v_path + f'/{frame}'
                t_path = tar_path + f'/{frame}'
                shutil.copy(s_path, t_path)

                storage += float(os.path.getsize(s_path))


print('Storage : {:.4f} GB'.format(storage / (1024*1024*1024)))

